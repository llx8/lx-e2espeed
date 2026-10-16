#!/usr/bin/env python3
"""Global Claude Code hook for lx-e2espeed, installed as ~/.claude/hooks/lx_dispatch.py.

  session-start  In a project that has lx-work/, print where the rules are, the phase and its rules
                 doc, warnings, the project status, and the entry rules, shortest first: Claude Code
                 cuts long hook output to a short preview. In the workspace (the directory holding
                 the rules as lx-e2espeed/ above registered projects), print one line per project
                 and rules/workspace.md instead, so no project is read before the user names it.
                 Elsewhere, print nothing. It also runs after compaction, so the handoff survives it.
  post-edit      After an edit inside lx-e2espeed/ or lx-work/, or a shell command in or naming
                 one of them, run the rules' checker on it.
                 Problems go to stderr with exit code 2, which Claude Code shows to Claude.
  prompt         On every prompt in such a project or the workspace, print the reply naming rule, the
                 reply language, and answering only what was asked, as a reminder.
  stop           When Claude ends its turn there: while STATUS.md marks an autonomous run for this session
                 (a Run line with its id), refuse the end of the turn with exit code 2, so the run goes on;
                 otherwise flag the turn's reply text if it names things by number (circled numbers,
                 experiment numbers), with exit code 2 so Claude corrects it, never twice in a row
                 (stop_hook_active).

It only reads files and runs the project's own checker. Unexpected errors exit 0,
so the hook never blocks unrelated work.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

RULES, WORK = "lx-e2espeed", "lx-work"
LX_NAMES = (RULES, WORK)
# Which rules doc to read first in each STATUS.md phase, so routing does not depend on recall.
PHASE_DOCS = {
    "onboarding": ("Read lx-e2espeed/rules/onboarding.md now, or rules/standard-pass.md when the user hands the project "
                   "over to optimize (CLAUDE.md, Session start)."),
    "optimizing": "Read lx-e2espeed/rules/experiment.md before the next experiment.",
    "wrap-up": "Read lx-e2espeed/rules/wrap-up.md now.",
    "done": "The project is done; ask the user what comes next.",
}
PHASE = re.compile(r"^- Phase: (\S+)", re.M)
SYNCED = re.compile(r"^- Rules synced: ([0-9a-f]{7,40})\s*$", re.M)
WORKSPACE_DOC = "rules/workspace.md"
LINE_MAX = 160  # characters of a STATUS.md item in the workspace overview; the full text is one Read away
# A path to an lx-work directory in a shell command, such as `git -C p3/demo/lx-work commit`.
WORK_PATH = re.compile(r"[^\s'\"=;|&<>()]*lx-work(?![\w-])")
HOOK_MAX_ERRORS = 15  # the rest costs context without helping the next fix
# The user could not follow replies that named versions and steps by circled numbers or experiment numbers, and
# asked for a rule enforced by hooks (2026-10-11). NUMBER_LABEL finds them; the experiment word is matched in
# English and Chinese (escaped: files here are English only). Code blocks, code spans, and link targets are
# blanked first, so paths and identifiers stay allowed.
NUMBER_LABEL = re.compile(r"[\u2460-\u2473\u2776-\u2793]|(?<![A-Za-z])exp(?:eriment)?\s*#?\s*\d+|\u5b9e\u9a8c\s*#?\s*\d+",
                          re.I)
CODE = re.compile(r"```.*?```|`[^`\n]*`|\]\([^)\n]*\)", re.S)
REPLY_RULE = ("lx-e2espeed reply rule (CLAUDE.md, Hard rules): name every version, optimization, step, and scenario "
              "by what it is or changes, in plain words, tables included; never a bare or circled number or an "
              "experiment ID in its place.")
# Replies drifted into English after compactions, pulled by the English rules and files, and grew past the question
# (FEEDBACK.md of several projects, 2026-10-09 to 2026-10-11); the prompt reminder carries both.
PROMPT_RULE = ("Reply in the language the user writes in, not in the rules' English, and answer only what the prompt "
               "asks (rules/reporting.md).")
TRANSCRIPT_TAIL = 4 << 20  # bytes of the transcript read first for the current turn (turn_text)
STAMP = Path.home() / ".claude" / "hooks" / "lx_installed"  # written by setup.sh global
# An autonomous run (rules/standard-pass.md, rules/self-iteration.md) marks itself in STATUS.md under "## In progress"
# with a Run line carrying the session id the SessionStart hook printed. While it names the current session, the Stop
# hook refuses the end of the turn, so a run cannot stop early (the user, 2026-10-13); the run's ending steps replace
# it. Another session's line never holds this one, and a line left by a dead session holds nobody.
RUN_LINE = re.compile(r"^- Run: (standard pass|self-iteration run), session ([0-9A-Za-z-]{8,})\b", re.M)
RUN_DOCS = {"standard pass": "rules/standard-pass.md", "self-iteration run": "rules/self-iteration.md"}
RUN_REFUSALS_MAX = 20  # refusals in a row while STATUS.md stays unchanged; past it the turn may end, so a stuck run cannot loop forever
# Every project shares one rules directory, outside the projects; setup.sh global writes its path here.
LOCATION = Path.home() / ".claude" / "hooks" / "lx_rules"


def rules_dir():
    """The rules directory setup.sh global installed from, or None before it ran."""
    try:
        path = Path(LOCATION.read_text(encoding="utf-8").strip())
    except OSError:
        return None
    return path if (path / "CLAUDE.md").is_file() else None


def project_root(cwd):
    result = subprocess.run(["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
                            capture_output=True, text=True, timeout=10)
    root = Path(result.stdout.strip()) if result.returncode == 0 else cwd
    # lx-work is its own repo; from inside it, the project root is one level up.
    return root.parent if root.name in LX_NAMES else root


def lx_repo_of(path):
    for parent in (path, *path.parents):
        if parent.name in LX_NAMES and parent.is_dir():
            return parent
    return None


def in_project(data):
    """Whether the session runs in a project that uses these rules (lx-work/ at its root), or in the workspace."""
    cwd, rules = Path(data.get("cwd") or "."), rules_dir()
    return rules is not None and ((project_root(cwd) / WORK).is_dir() or bool(workspace_projects(cwd, rules)))


def workspace_projects(cwd, rules):
    """(path relative to cwd, root) of each project setup.sh registered below cwd, when cwd is the workspace: the
    directory that holds the rules as lx-e2espeed/ (a symlink to the shared rules directory) and is no project.
    Elsewhere none, so a session in an unrelated directory gets nothing."""
    if (project_root(cwd) / WORK).is_dir() or (cwd / RULES).resolve() != rules.resolve():
        return []
    try:
        registered = [l.strip() for l in (rules / "local" / "projects").read_text(encoding="utf-8").splitlines()
                      if l.strip()]  # one project root per line, as setup.sh project writes it
    except OSError:
        return []
    found = []
    for root in map(Path, registered):
        try:
            rel = root.resolve().relative_to(cwd.resolve())
        except ValueError:
            continue
        if (root / WORK).is_dir():
            found.append((rel, root))
    return sorted(found)


def section_items(text, title):
    """The bullets of a STATUS.md section, without the skeleton's <placeholder> ones."""
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    items = [l[2:].strip() for l in match.group(1).splitlines() if l.startswith("- ")] if match else []
    return [i for i in items if not i.startswith("<")]


def shorten(text):
    return text if len(text) <= LINE_MAX else text[:LINE_MAX - 3].rstrip() + "..."


def running_notes(work):
    return [note.parent.relative_to(work).as_posix() for note in sorted(work.glob("*/[0-9][0-9][0-9]-*/NOTE.md"))
            if re.search(r"^- Status: running$", note.read_text(encoding="utf-8"), re.M)]


def project_line(rel, root, rules):
    """One overview line for a workspace project, from its STATUS.md: phase and its rules doc, the first Next item,
    how many items wait for the user, and what needs attention before work starts."""
    work = root / WORK
    status = work / "STATUS.md"
    text = status.read_text(encoding="utf-8") if status.is_file() else ""
    match = PHASE.search(text)
    phase = match.group(1) if match else "unknown"
    doc = re.search(r"rules/[\w-]+\.md", PHASE_DOCS.get(phase, ""))
    parts = [f"- {rel.as_posix()}: phase {phase}" + (f" ({doc.group(0)})" if doc else "")]
    nxt, waiting = section_items(text, "Next"), section_items(text, "Waiting for the user")
    if nxt:
        parts.append(f"next: {shorten(nxt[0].rstrip('.'))}")
    if waiting:
        parts.append(f"{len(waiting)} item(s) waiting for the user")
    parts += [f"{name} is still running" for name in running_notes(work)]
    if skeleton_behind(rules, text):
        parts.append("skeleton sync due (rules/naming.md)")
    return "; ".join(parts) + "."


def number_labels(text):
    """The number labels in reply text, outside code and link targets, in order of first appearance."""
    return list(dict.fromkeys(m.group(0).strip() for m in NUMBER_LABEL.finditer(CODE.sub(" ", text))))


def turn_texts(path, tail):
    """The assistant texts after the last typed prompt in the transcript's last `tail` bytes, and whether the tail
    held such a prompt."""
    texts, found = [], False
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - tail))
        raw = f.read().decode("utf-8", "replace")
    for line in raw.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # the first line of the tail may be cut
        content = (entry.get("message") or {}).get("content")
        if entry.get("type") == "user" and not entry.get("isMeta"):
            if isinstance(content, str) or any(isinstance(b, dict) and b.get("type") == "text" for b in content or []):
                texts, found = [], True
        elif entry.get("type") == "assistant" and isinstance(content, list):
            texts += [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return texts, found


def turn_text(data):
    """The assistant text since the user's last typed prompt. The tail read grows until it holds that prompt: tool
    output and attachments can make one turn several megabytes."""
    texts, path, tail = [], Path(data.get("transcript_path") or ""), TRANSCRIPT_TAIL
    if path.is_file():
        texts, found = turn_texts(path, tail)
        while not found and tail < path.stat().st_size:
            tail *= 4
            texts, found = turn_texts(path, tail)
    last = data.get("last_assistant_message")
    if isinstance(last, str) and last not in texts:
        texts.append(last)
    return "\n".join(texts)


def prompt(data):
    if in_project(data):
        print(REPLY_RULE + " " + PROMPT_RULE)
    return 0


def run_of_session(data):
    """(kind, STATUS.md path) of the autonomous run whose Run line names this session: in a project its own STATUS.md,
    in the workspace every registered project's."""
    session = data.get("session_id")
    if not session:
        return None
    cwd, rules = Path(data.get("cwd") or "."), rules_dir()
    root = project_root(cwd)
    roots = [root] if (root / WORK).is_dir() else [r for _, r in workspace_projects(cwd, rules)]
    for root in roots:
        status = root / WORK / "STATUS.md"
        try:
            text = status.read_text(encoding="utf-8")
        except OSError:
            continue
        for kind, sid in RUN_LINE.findall(text):
            if sid == session:
                return kind, status
    return None


def run_refusals(status, session):
    """How many turns in a row the Stop hook refused for this run while STATUS.md stayed unchanged, counted in
    lx-work/local/, which lx-work ignores."""
    marker = status.parent / "local" / f"stop-refusals-{session}"
    stamp, count = str(status.stat().st_mtime_ns), 0
    try:
        old_stamp, old_count = marker.read_text(encoding="utf-8").split()
        if old_stamp == stamp:
            count = int(old_count)
    except (OSError, ValueError):
        pass
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(f"{stamp} {count + 1}\n", encoding="utf-8")
    return count + 1


def stop(data):
    if not in_project(data):
        return 0
    run = run_of_session(data)
    if run:
        kind, status = run
        refusals = run_refusals(status, data["session_id"])
        if refusals <= RUN_REFUSALS_MAX:
            sys.stderr.write(f"{status} marks this session's {kind} as in progress, so the turn does not end (refusal "
                             f"{refusals} of {RUN_REFUSALS_MAX} while STATUS.md is unchanged). Go on with the next item; the "
                             f"run ends only by the ending steps of {RUN_DOCS[kind]}, which replace that Run line.\n")
            return 2
        print(f"STATUS.md unchanged over {RUN_REFUSALS_MAX} refused stops: the turn may end; the {kind} is still marked.")
    if data.get("stop_hook_active"):
        return 0
    found = number_labels(turn_text(data))
    if not found:
        return 0
    sys.stderr.write(f"Your reply names things by number: {', '.join(found[:10])}. {REPLY_RULE} Send a short "
                     "correction: the affected lines or table again, each name replaced by what it is.\n")
    return 2


def install_stamp(rules):
    """Hash of what setup.sh global installs from the rules directory: this hook and every skill."""
    digest = hashlib.sha256()
    skills = list((rules / "vendor" / "skills").rglob("*"))
    files = [rules / "hooks" / "dispatch.py",
             *sorted(p for p in skills if p.is_file() and "__pycache__" not in p.parts)]
    for path in files:
        digest.update(path.relative_to(rules).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def skeleton_behind(rules, status):
    """A note when templates/work-skeleton changed after lx-work's '- Rules synced:' commit, or that commit is gone.
    With one rules directory for every project, no pull marks the moment to sync lx-work (rules/naming.md)."""
    match = SYNCED.search(status)
    if not match:
        return None
    result = subprocess.run(["git", "-C", str(rules), "diff", "--quiet", match.group(1), "HEAD", "--",
                             "templates/work-skeleton"], capture_output=True, timeout=10)
    if result.returncode == 0:
        return None
    return (f"templates/work-skeleton changed after the commit lx-work last synced with ({match.group(1)}), or that "
            "commit is gone: tell the user and propose the sync in lx-e2espeed/rules/naming.md.")


def stamp_note(rules):
    installed = STAMP.read_text(encoding="utf-8").strip() if STAMP.is_file() else ""
    if installed != install_stamp(rules):
        print("The hooks and skills installed on this machine differ from lx-e2espeed's. Tell the user and "
              "propose `lx-e2espeed/setup.sh global`.")


def session_line(session):
    return f"Session id: {session}, for the Run line of an autonomous run (STATUS.md, In progress)." if session else ""


def run_notes(text, session):
    for kind, sid in RUN_LINE.findall(text):
        if sid == session:
            yield (f"Your {kind} is in progress: go on with it ({RUN_DOCS[kind]}). The Stop hook holds the turn until "
                   "its ending steps replace the Run line.")
        else:
            yield (f"A {kind} of session {sid[:8]} is marked in progress. Only if the user says to resume it, put this "
                   "session's id on its Run line first; otherwise leave the line alone.")


def workspace_start(cwd, rules, projects, session=None):
    """The workspace overview: one line per project and the workspace rules, nothing read from inside a project
    beyond its STATUS.md, so the session reads a project only once the user names it."""
    print(f"This is the lx workspace {cwd}: the projects below use {RULES}, the rules directory {rules}; every "
          f"`{RULES}/` path in the rules means that directory. Work on one project at a time, the one the user "
          f"names, as {WORKSPACE_DOC} below says.\n\nProjects, from each {WORK}/STATUS.md:")
    for rel, root in projects:
        print(project_line(rel, root, rules))
    if session:
        print(session_line(session))
    stamp_note(rules)
    print("\n" + (rules / WORKSPACE_DOC).read_text(encoding="utf-8"))
    return 0


def session_start(data):
    cwd, rules = Path(data.get("cwd") or "."), rules_dir()
    root = project_root(cwd)
    work = root / WORK
    if rules is not None and not work.is_dir():
        projects = workspace_projects(cwd, rules)
        if projects:
            return workspace_start(cwd, rules, projects, data.get("session_id"))
    if rules is None or not work.is_dir():
        return 0  # a project starts using the rules with setup.sh project, which creates lx-work/
    # Shortest and most urgent first: past about 10 KB, Claude Code shows only the first 2 KB.
    print(f"This project uses {RULES}, the rules directory {rules}; every `{RULES}/` path below and in the rules "
          f"means that directory. Other paths are relative to {root}. If this output was cut short, read "
          f"{rules}/CLAUDE.md and {WORK}/STATUS.md in full before anything else.\n")
    status = work / "STATUS.md"
    text = status.read_text(encoding="utf-8") if status.is_file() else None
    if text is not None:
        match = PHASE.search(text)
        phase = match.group(1) if match else "unknown"
        print(f"Current phase: {phase}. " + PHASE_DOCS.get(phase, "Fix the Phase line in lx-work/STATUS.md."))
        if data.get("session_id"):
            print(session_line(data["session_id"]))
        for note in run_notes(text, data.get("session_id")):
            print(note)
        for name in running_notes(work):
            print(f"{name} is still running, and its change is in the project tree, so nothing else can start "
                  "until it ends. During a self-iteration run or a standard pass, go on with it (its rules doc); "
                  "otherwise tell the user and propose how to end it (its NOTE.md has the plan).")
        behind = skeleton_behind(rules, text)
        if behind:
            print(behind)
    stamp_note(rules)
    if text is not None:
        print("\n" + text)
    print((rules / "CLAUDE.md").read_text(encoding="utf-8"))
    return 0


def post_edit(data):
    cwd = Path(data.get("cwd") or ".")
    tool_input = data.get("tool_input") or {}
    path = tool_input.get("file_path") or tool_input.get("notebook_path")
    repos = set()
    if path:
        repo = lx_repo_of(cwd / path)
        if repo:
            repos.add(repo)
    elif data.get("tool_name") == "Bash":
        # Shell edits carry no file path: check the lx repo the shell is in, and those the command names.
        repo = lx_repo_of(cwd)
        if repo:
            repos.add(repo)
        command = tool_input.get("command", "")
        if WORK in command and (project_root(cwd) / WORK).is_dir():
            repos.add(project_root(cwd) / WORK)
        for named in WORK_PATH.findall(command):  # from the workspace, a command names its project's lx-work
            work = (cwd / Path(named).expanduser()).resolve()
            if work.name == WORK and work.is_dir():
                repos.add(work)
        if RULES in command and rules_dir():
            repos.add(rules_dir())
    failures = []
    for repo in sorted({r.resolve() for r in repos}):  # one run per repo, however the paths spelled it
        # A rules repo is checked by its own checker, lx-work by that of the rules every project shares.
        rules = repo if repo.name == RULES else rules_dir()
        checker = rules / "hooks" / "check_docs.py" if rules else None
        if not checker or not checker.is_file():
            continue
        result = subprocess.run([sys.executable, str(checker), str(repo)],
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            lines = [l for l in (result.stdout + result.stderr).splitlines() if not l.startswith("warning:")]
            errors, summary = lines[:-1], lines[-1:]
            if len(errors) > HOOK_MAX_ERRORS:
                errors = errors[:HOOK_MAX_ERRORS] + [f"... {len(errors) - HOOK_MAX_ERRORS} more errors not shown"]
            failures.append("\n".join(errors + summary) + "\n")
    if failures:
        sys.stderr.write("lx doc check failed. Fix these before moving on:\n" + "".join(failures))
        return 2
    return 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "stamp":  # run by setup.sh global, not by Claude Code
        print(install_stamp(Path(sys.argv[2])))
        return 0
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        data = {}
    try:
        if mode == "session-start":
            return session_start(data)
        if mode == "post-edit":
            return post_edit(data)
        if mode == "prompt":
            return prompt(data)
        if mode == "stop":
            return stop(data)
    except Exception:  # never block unrelated work because of this hook
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
