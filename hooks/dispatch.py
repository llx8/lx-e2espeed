#!/usr/bin/env python3
"""Global Claude Code hook for lx-e2espeed, installed as ~/.claude/hooks/lx_dispatch.py.

  session-start  In a project that has lx-e2espeed/, print the phase and its rules doc, warnings,
                 the project status, and the entry rules, shortest first: Claude Code cuts long hook
                 output to a short preview. Without lx-work/ yet, route to onboarding. Elsewhere,
                 print nothing. It also runs after compaction, so the handoff survives it.
  post-edit      After an edit inside lx-e2espeed/ or lx-work/, or a shell command in or naming
                 one of them, run that clone's checker.
                 Problems go to stderr with exit code 2, which Claude Code shows to Claude.

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
    except Exception:  # never block unrelated work because of this hook
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
