#!/usr/bin/env python3
"""Global Claude Code hook for lx-e2espeed, installed as ~/.claude/hooks/lx_dispatch.py.

  session-start  In a project that has lx-e2espeed/, print the phase and its rules doc, warnings,
                 the project status, and the entry rules, shortest first: Claude Code cuts long hook
                 output to a short preview. Without lx-work/ yet, route to onboarding. Elsewhere,
                 print nothing. It also runs after compaction, so the handoff survives it.
  post-edit      After an edit inside lx-e2espeed/ or lx-work/, or a shell command in or naming
                 one of them, run that clone's checker.
                 Problems go to stderr with exit code 2, which Claude Code shows to Claude.
  prompt         On every prompt in such a project, print the reply naming rule as a reminder.
  stop           When Claude ends its turn there, flag the turn's reply text if it names things by
                 number (circled numbers, experiment numbers), with exit code 2 so Claude corrects it;
                 never twice in a row (stop_hook_active).

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
    "onboarding": "Read lx-e2espeed/rules/onboarding.md now.",
    "optimizing": "Read lx-e2espeed/rules/experiment.md before the next experiment.",
    "wrap-up": "Read lx-e2espeed/rules/wrap-up.md now.",
    "done": "The project is done; ask the user what comes next.",
}
PHASE = re.compile(r"^- Phase: (\S+)", re.M)
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
TRANSCRIPT_TAIL = 4 << 20  # bytes of the transcript read first for the current turn (turn_text)
STAMP = Path.home() / ".claude" / "hooks" / "lx_installed"  # written by setup.sh global


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
    """Whether the session runs in a project that uses these rules (lx-e2espeed/ and lx-work/ at its root)."""
    root = project_root(Path(data.get("cwd") or "."))
    return (root / RULES / "CLAUDE.md").is_file() and (root / WORK).is_dir()


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
        print(REPLY_RULE)
    return 0


def stop(data):
    if data.get("stop_hook_active") or not in_project(data):
        return 0
    found = number_labels(turn_text(data))
    if not found:
        return 0
    sys.stderr.write(f"Your reply names things by number: {', '.join(found[:10])}. {REPLY_RULE} Send a short "
                     "correction: the affected lines or table again, each name replaced by what it is.\n")
    return 2


def install_stamp(rules):
    """Hash of what setup.sh global installs from a clone: this hook and every skill."""
    digest = hashlib.sha256()
    skills = [p for d in (rules / "skills", rules / "vendor" / "skills") for p in d.rglob("*")]
    files = [rules / "hooks" / "dispatch.py",
             *sorted(p for p in skills if p.is_file() and "__pycache__" not in p.parts)]
    for path in files:
        digest.update(path.relative_to(rules).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def session_start(data):
    root = project_root(Path(data.get("cwd") or "."))
    entry, work = root / RULES / "CLAUDE.md", root / WORK
    if not entry.is_file():
        return 0
    if not (work.is_dir() or (root / ".git").exists()):
        return 0  # the rules repo opened on its own loads its CLAUDE.md without this hook
    # Shortest and most urgent first: past about 10 KB, Claude Code shows only the first 2 KB.
    print(f"This project uses {RULES}. Paths below are relative to {root}. If this output was cut short, "
          f"read {RULES}/CLAUDE.md and {WORK}/STATUS.md in full before anything else.\n")
    status = work / "STATUS.md"
    text = status.read_text(encoding="utf-8") if work.is_dir() and status.is_file() else None
    if not work.is_dir():
        print(f"{WORK}/ does not exist yet: onboarding starts at step 1 once the user says so. "
              + PHASE_DOCS["onboarding"])
    elif text is not None:
        match = PHASE.search(text)
        phase = match.group(1) if match else "unknown"
        print(f"Current phase: {phase}. " + PHASE_DOCS.get(phase, "Fix the Phase line in lx-work/STATUS.md."))
        for note in sorted(work.glob("*/[0-9][0-9][0-9]-*/NOTE.md")):
            if re.search(r"^- Status: running$", note.read_text(encoding="utf-8"), re.M):
                print(f"{note.parent.relative_to(work)} is still running, and its change is in the project tree, "
                      "so nothing else can start until it ends. During a night run, go on with it (rules/night.md); "
                      "otherwise tell the user and propose how to end it (its NOTE.md has the plan).")
    installed = STAMP.read_text(encoding="utf-8").strip() if STAMP.is_file() else ""
    if installed != install_stamp(root / RULES):
        print("The hooks and skills installed on this machine differ from this clone's. Tell the user and "
              "propose `git -C lx-e2espeed pull --ff-only`, then `lx-e2espeed/setup.sh global` if they still differ.")
    if text is not None:
        print("\n" + text)
    print(entry.read_text(encoding="utf-8"))
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
        if any(n in tool_input.get("command", "") for n in LX_NAMES):
            root = project_root(cwd)
            repos.update(root / n for n in LX_NAMES if (root / n).is_dir())
    failures = []
    for repo in sorted(repos):
        rules = repo if repo.name == RULES else repo.parent / RULES
        checker = rules / "hooks" / "check_docs.py"
        if not checker.is_file():
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
