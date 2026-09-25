"""Layout checks for an lx-work repo, called by check_docs.py.

Keeps the handoff honest: STATUS.md, NOTE.md, and the indexes must agree with the directories,
and docs made from templates must keep the template's headings and fields.
"""
import re

PHASES = {"onboarding", "optimizing", "wrap-up", "done"}
STATUSES = {"running", "kept", "dropped"}
# What each finished experiment needs: kept ones must rebuild and rerun, dropped ones reverse cleanly.
ENDED_FILES = {"kept": ("change.patch", "run.sh", "result.json"), "dropped": ("change.patch",)}
EXP_DIR = re.compile(r"(\d{3})(-[a-z0-9]+){1,4}")
INDEX_ROW = re.compile(r"^\|\s*(\d{3}(?:-[a-z0-9]+)+)\s*\|\s*([a-z]*)\s*\|", re.M)
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
WORK_REQUIRED = ("STATUS.md", "PROJECT.md", "DECISIONS.md", "FEEDBACK.md", "BACKLOG.md", "MAP.md",
                 "tools/INDEX.md", "exp/INDEX.md")
IGNORED_DIRS = {"__pycache__"}
SHAPE = re.compile(r"^(## .+|- [A-Z][A-Za-z ]*:)")  # template lines a doc must keep
COMMIT = re.compile(r"[0-9a-f]{7,40}")
PLACEHOLDER = re.compile(r"<[A-Za-z][^<>\n]*>")  # an unfilled template field such as <metric>


def field(text, name):
    match = re.search(rf"^- {name}: (.+)$", text, re.M)
    return match.group(1).strip() if match else None


def shape(text):
    return [m.group(1) for line in text.splitlines() if (m := SHAPE.match(line))]


def check_shape(doc, template, rel, rep):
    """A doc keeps every heading and field line of its template; scripts and readers rely on them."""
    if not (doc.is_file() and template.is_file()):
        return
    have = set(shape(doc.read_text(encoding="utf-8")))
    for line in shape(template.read_text(encoding="utf-8")):
        if line not in have:
            rep.error(rel, f"misses '{line}' from its template; add it, or restore it if it was removed")


def check_work_layout(root, rep, rules_root):
    skeleton = rules_root / "templates" / "work-skeleton"
    ignore = (root / ".gitignore").read_text(encoding="utf-8").split() if (root / ".gitignore").is_file() else []
    for entry in ("local/", "__pycache__/"):
        if entry not in ignore:
            rep.error(".gitignore", f"must ignore {entry}; untrack anything already committed with `git rm -r --cached`")
    for rel in WORK_REQUIRED:
        if not (root / rel).is_file():
            rep.error(rel, "missing; copy it from lx-e2espeed/templates/work-skeleton/")
        check_shape(root / rel, skeleton / rel, rel, rep)
    status_path = root / "STATUS.md"
    status = status_path.read_text(encoding="utf-8") if status_path.is_file() else ""
    phase, updated = field(status, "Phase"), field(status, "Updated") or ""
    if phase not in PHASES:
        rep.error("STATUS.md", f"needs a '- Phase: <phase>' line with one of {sorted(PHASES)}")
    elif phase != "onboarding" and not DATE.fullmatch(updated):
        rep.error("STATUS.md", "'- Updated:' must be a YYYY-MM-DD date once onboarding is over")
    project = (root / "PROJECT.md").read_text(encoding="utf-8") if (root / "PROJECT.md").is_file() else ""
    if phase in PHASES - {"onboarding"} and not COMMIT.fullmatch(field(project, "Baseline commit") or ""):
        rep.error("PROJECT.md", "'- Baseline commit:' must be a commit hash once onboarding is over")
    check_experiments(root, rep, updated if DATE.fullmatch(updated) else None,
                      rules_root / "templates" / "exp-note.md")
    check_tools(root, rep)


def check_experiments(root, rep, status_updated, note_template):
    exp = root / "exp"
    if not exp.is_dir():
        return
    listed = "".join((exp / n).read_text(encoding="utf-8")
                     for n in ("INDEX.md", "ARCHIVE.md") if (exp / n).is_file())
    rows, numbers, running = dict(INDEX_ROW.findall(listed)), {}, []
    folders = sorted(p for p in exp.iterdir() if p.is_dir())
    for folder in folders:
        where = f"exp/{folder.name}/"
        match = EXP_DIR.fullmatch(folder.name)
        if not match:
            rep.error(where, "name must be a three-digit number plus 1 to 4 lowercase words")
            continue
        if numbers.setdefault(match.group(1), folder.name) != folder.name:
            rep.error(where, f"number already used by exp/{numbers[match.group(1)]}/")
        if folder.name not in rows:
            rep.error(where, "has no row in exp/INDEX.md")
        if not (folder / "NOTE.md").is_file():
            rep.error(where, "missing NOTE.md")
            continue
        note = (folder / "NOTE.md").read_text(encoding="utf-8")
        check_shape(folder / "NOTE.md", note_template, where + "NOTE.md", rep)
        state, date = field(note, "Status"), field(note, "Date") or ""
        if state not in STATUSES:
            rep.error(where + "NOTE.md", f"Status must be one of {sorted(STATUSES)}")
        elif folder.name in rows and rows[folder.name] != state:
            rep.error(where, f"NOTE.md says {state} but exp/INDEX.md says '{rows[folder.name]}'")
        if state == "running":
            running.append(folder.name)
        elif state in ENDED_FILES and PLACEHOLDER.search(note):
            rep.error(where + "NOTE.md", f"{state}, but '{PLACEHOLDER.search(note).group(0)}' is still a template placeholder")
        for name in ENDED_FILES.get(state, ()):
            if not (folder / name).is_file():
                rep.error(where, f"{state}, so it needs {name}")
        if status_updated and DATE.fullmatch(date) and date > status_updated:
            rep.error("STATUS.md", f"updated {status_updated}, before exp/{folder.name} ({date}); update the handoff")
    for name in sorted(set(rows) - {p.name for p in folders}):
        rep.error("exp/INDEX.md", f"{name} has no directory")
    if len(running) > 1:
        rep.error("exp/", f"only one experiment runs at a time: {', '.join(running)}")


def check_tools(root, rep):
    tools = root / "tools"
    if not tools.is_dir():
        return
    listed = (tools / "INDEX.md").read_text(encoding="utf-8") if (tools / "INDEX.md").is_file() else ""
    for folder in sorted(p for p in tools.iterdir() if p.is_dir() and p.name not in IGNORED_DIRS):
        if not any(folder.glob("main.*")):
            rep.error(f"tools/{folder.name}/", "missing entry file main.*")
        if f"{folder.name}/" not in listed:
            rep.error(f"tools/{folder.name}/", "not listed in tools/INDEX.md")
