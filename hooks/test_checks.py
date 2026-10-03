#!/usr/bin/env python3
"""Prove that every check in check_docs.py can fail.

Usage: test_checks.py
Builds two clean fixtures in a temporary directory, a copy of this lx-e2espeed repo and an lx-work
inside a small project, and confirms the checker passes on both. Then each probe breaks one thing
in a fresh copy and expects the checker's message for it. A check that never fails looks exactly
like one that works, so every new check gets a probe here. The pre-commit hook runs this file.
Exits 1 if a fixture fails or a probe goes undetected.
"""
import contextlib
import io
import re
import shutil
import sys
import tempfile
from pathlib import Path

HOOKS = Path(__file__).resolve().parent
RULES = HOOKS.parent
sys.path.insert(0, str(HOOKS))
import check_docs  # noqa: E402

PROJECT = "qzxproject"  # the fixture project's directory; the checker treats its name as a private term
COMMIT = "1234567"
PLACEHOLDER = re.compile(r"<[A-Za-z][^<>\n]*>")
APP = '''def load():
    return Pipe()


class Pipe:
    def __call__(self, x):
        return self.run(x)

    def run(self, x):
        return x
'''
PATCH = '''diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -9,2 +9,2 @@ class Pipe:
     def run(self, x):
-        return x
+        return x * 1
'''


def read(path):
    return path.read_text(encoding="utf-8")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def edit(path, old, new):
    """Replace old with new; a probe whose edit misses would pass for the wrong reason, so it fails loudly."""
    text = read(path)
    if old not in text:
        raise AssertionError(f"probe setup: {old!r} not found in {path.name}")
    write(path, text.replace(old, new, 1))


def append(path, text):
    old = read(path)
    write(path, old + ("" if old.endswith("\n") else "\n") + text)


def fill(path, fields):
    """Set the given '- Field:' lines, then replace every remaining placeholder."""
    text = read(path)
    for name, value in fields.items():
        text, n = re.subn(rf"^- {name}: .*$", f"- {name}: {value}", text, flags=re.M)
        if not n:
            raise AssertionError(f"fixture: no '- {name}:' line in {path.name}")
    write(path, PLACEHOLDER.sub("none", text))


def build_rules(base):
    root = base / PROJECT / "lx-e2espeed"
    shutil.copytree(RULES, root, ignore=shutil.ignore_patterns(".git", "__pycache__", "local", ".claude"))
    return root


def build_work(base):
    project = base / PROJECT
    write(project / "app.py", APP)
    work = project / "lx-work"
    shutil.copytree(RULES / "templates" / "work-skeleton", work)
    fill(work / "STATUS.md", {"Phase": "optimizing", "Updated": "2026-09-30"})
    fill(work / "PROJECT.md", {"Baseline commit": COMMIT})
    append(work / "PROJECT.md", "- Bench: `tools/bench/main.py --candidate <file>`\n\n    tools/verify/main.py --project <dir>\n")
    fill(work / "MAP.md", {"Measured on": f"baseline {COMMIT} + exp up to 001"})
    append(work / "MAP.md", "- Model call: `app.py:Pipe.run`, entered from `app.py:load`.\n")
    for rel in ("DECISIONS.md", "FEEDBACK.md", "BACKLOG.md", "tools/INDEX.md", "exp/INDEX.md", "night/INDEX.md"):
        fill(work / rel, {})
    item = work / "exp" / "001-fast-path"
    item.mkdir()
    write(item / "NOTE.md", read(RULES / "templates" / "exp-note.md").replace("# NNN-name", "# 001-fast-path"))
    fill(item / "NOTE.md", {"Date": "2026-09-29", "Base": f"baseline {COMMIT} + kept none", "Status": "kept"})
    write(item / "change.patch", PATCH)
    write(item / "run.sh", "#!/usr/bin/env bash\n")
    write(item / "result.json", "{}\n")
    append(work / "exp" / "INDEX.md", "| 001-fast-path | kept | 1.0 ms | -10% | -10% | L0 | faster |\n")
    return work


def run_checker(root):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = check_docs.main(["check_docs.py", str(root)])
    return code, out.getvalue()


def note(work, name):
    return work / "exp" / name / "NOTE.md"


def add_item(work, name, status, row=True):
    """A second experiment directory like 001, with the given status."""
    shutil.copytree(work / "exp" / "001-fast-path", work / "exp" / name)
    edit(note(work, name), "# 001-fast-path", f"# {name}")
    edit(note(work, name), "- Status: kept", f"- Status: {status}")
    if row:
        append(work / "exp" / "INDEX.md", f"| {name} | {status} | | | | | |\n")


def lines(n):
    return "".join(f"filler line {i}\n" for i in range(n))


# (name, fixture, change to make, text the checker must print)
PROBES = [
    # check_docs.py: names, language, frontmatter, limits
    ("bad file name", "work", lambda w: write(w / "My Notes.md", read(w / "BACKLOG.md")), "file name breaks rules/naming.md"),
    ("bad directory name", "work", lambda w: (w / "Scratch Dir").mkdir(), "directory names are lowercase"),
    ("non-English text", "work", lambda w: append(w / "BACKLOG.md", "- " + chr(0x6D4B) + "\n"), "non-English text"),
    ("no frontmatter", "work", lambda w: write(w / "BACKLOG.md", "# Backlog\n"), "missing frontmatter"),
    ("bad tier", "work", lambda w: edit(w / "BACKLOG.md", "tier: on-demand", "tier: sometimes"), "tier must be one of"),
    ("empty read_when", "work", lambda w: edit(w / "BACKLOG.md", re.search(r"read_when: .*", read(w / "BACKLOG.md")).group(0),
                                               "read_when:"), "read_when is empty"),
    ("STATUS.md over its limit", "work", lambda w: append(w / "STATUS.md", lines(60)), "limit 60"),
    ("STATUS.md over its byte budget", "work", lambda w: append(w / "STATUS.md", "- " + "padding " * 400 + "\n"), "bytes, limit 3000"),
    ("long doc without contents", "work", lambda w: append(w / "BACKLOG.md", lines(100)), "without a '## Contents' list"),
    # check_work.py: layout, handoff, experiments, tools
    ("gitignore misses local/", "work", lambda w: write(w / ".gitignore", "__pycache__/\n"), "must ignore local/"),
    ("required doc missing", "work", lambda w: (w / "BACKLOG.md").unlink(), "BACKLOG.md: missing"),
    ("template heading lost", "work", lambda w: edit(w / "STATUS.md", "## Next\n", ""), "misses '## Next'"),
    ("knowledge index missing", "work", lambda w: (w / "knowledge" / "INDEX.md").unlink(), "knowledge/INDEX.md: missing"),
    ("knowledge heading lost", "work", lambda w: edit(w / "knowledge" / "contract.md", "## Output format\n", ""),
     "knowledge/contract.md: misses '## Output format'"),
    ("unknown phase", "work", lambda w: edit(w / "STATUS.md", "- Phase: optimizing", "- Phase: resting"), "needs a '- Phase:"),
    ("Updated not a date", "work", lambda w: edit(w / "STATUS.md", "- Updated: 2026-09-30", "- Updated: today"),
     "'- Updated:' must be a YYYY-MM-DD date"),
    ("baseline not a commit", "work", lambda w: edit(w / "PROJECT.md", f"- Baseline commit: {COMMIT}", "- Baseline commit: head"),
     "'- Baseline commit:' must be a commit hash"),
    ("placeholder after onboarding", "work", lambda w: append(w / "STATUS.md", "- <next step>\n"), "still a template placeholder after onboarding"),
    ("placeholder beside a code span", "work", lambda w: append(w / "PROJECT.md", "- Profilers: `nsys --version`, <working ncu command>\n"),
     "'<working ncu command>' is still a template placeholder"),
    ("bad experiment name", "work", lambda w: (w / "exp" / "1-fast").mkdir(), "name must be a three-digit number"),
    ("number reused", "work", lambda w: add_item(w, "001-other-path", "dropped"), "number already used"),
    ("experiment without a row", "work", lambda w: add_item(w, "002-other-path", "dropped", row=False), "has no row in exp/INDEX.md"),
    ("experiment without NOTE.md", "work", lambda w: (w / "exp" / "001-fast-path" / "NOTE.md").unlink(), "missing NOTE.md"),
    ("unknown status", "work", lambda w: (add_item(w, "002-other-path", "resting")), "Status must be one of"),
    ("NOTE.md and index disagree", "work", lambda w: edit(w / "exp" / "INDEX.md", "| 001-fast-path | kept |", "| 001-fast-path | dropped |"),
     "NOTE.md says kept but exp/INDEX.md says 'dropped'"),
    ("placeholder in a finished note", "work", lambda w: append(note(w, "001-fast-path"), "<what was learned>\n"),
     "kept, but '<what was learned>'"),
    ("kept without run.sh", "work", lambda w: (w / "exp" / "001-fast-path" / "run.sh").unlink(), "kept, so it needs run.sh"),
    ("STATUS.md older than an experiment", "work", lambda w: edit(note(w, "001-fast-path"), "- Date: 2026-09-29", "- Date: 2026-10-01"),
     "update the handoff"),
    ("row without a directory", "work", lambda w: append(w / "exp" / "INDEX.md", "| 009-ghost-path | dropped | | | | | |\n"), "has no directory"),
    ("two items running", "work", lambda w: (add_item(w, "002-one-path", "running"), add_item(w, "003-two-path", "running")),
     "only one experiment or night trial runs at a time"),
    ("tool without main", "work", lambda w: (w / "tools" / "bench").mkdir(), "missing entry file main.*"),
    ("tool not listed", "work", lambda w: write(w / "tools" / "bench" / "main.py", ""), "not listed in tools/INDEX.md"),
    # check_work.py: MAP.md
    ("pointer to a missing file", "work", lambda w: append(w / "MAP.md", "- `gone.py:run`\n"), "gone.py is not in the project"),
    ("pointer to a missing function", "work", lambda w: append(w / "MAP.md", "- `app.py:Pipe.fly`\n"), "does not define fly"),
    ("malformed Measured on", "work", lambda w: edit(w / "MAP.md", f"- Measured on: baseline {COMMIT} + exp up to 001",
                                                     "- Measured on: last week"), "'- Measured on:' must read"),
    ("breakdown on another baseline", "work", lambda w: edit(w / "MAP.md", f"baseline {COMMIT} +", "baseline abcdef0 +"),
     "but PROJECT.md's baseline is"),
    ("breakdown older than a kept change", "work", lambda w: edit(w / "MAP.md", f"baseline {COMMIT} + exp up to 001", f"baseline {COMMIT}"),
     "time breakdown predates exp/001-fast-path (kept), which changed app.py"),
    ("breakdown older than a delivered change", "work", lambda w: (
        edit(note(w, "001-fast-path"), "- Status: kept", "- Status: delivered"),
        edit(w / "exp" / "INDEX.md", "| 001-fast-path | kept |", "| 001-fast-path | delivered |"),
        edit(w / "MAP.md", f"baseline {COMMIT} + exp up to 001", f"baseline {COMMIT}")),
     "time breakdown predates exp/001-fast-path (delivered)"),
    # check_rules.py: entry doc and doc map
    ("CLAUDE.md lost a section", "rules", lambda r: edit(r / "CLAUDE.md", "## Hard rules\n", "## Rules\n"), "lost its '## Hard rules' section"),
    ("CLAUDE.md over its limit", "rules", lambda r: append(r / "CLAUDE.md", lines(100)), "limit 100"),
    ("CLAUDE.md over its byte budget", "rules", lambda r: append(r / "CLAUDE.md", "padding " * 400 + "\n"), "bytes, limit 6000"),
    ("rules doc not in the doc map", "rules", lambda r: write(r / "rules" / "extra.md", read(r / "rules" / "naming.md")),
     "not listed in the CLAUDE.md doc map"),
    ("doc map names a missing doc", "rules", lambda r: append(r / "CLAUDE.md", "- `rules/missing.md`: never.\n"), "`rules/missing.md` does not exist"),
    ("skill without SKILL.md", "rules", lambda r: (r / "skills" / "empty-skill").mkdir(), "missing SKILL.md"),
    ("skill no rules doc routes to", "rules", lambda r: shutil.copytree(r / "skills" / "gpu-inference-delivery", r / "skills" / "gpu-inference-extra"),
     "no rules doc says when to load it"),
    ("phase routed to a missing doc", "rules", lambda r: edit(r / "hooks" / "dispatch.py", "rules/wrap-up.md", "rules/wrap-down.md"),
     "routes a phase to rules/wrap-down.md"),
    ("skill name differs from its directory", "rules", lambda r: edit(r / "skills" / "gpu-inference-delivery" / "SKILL.md",
                                                                      "name: gpu-inference-delivery", "name: delivery"),
     "frontmatter name must equal"),
    # check_rules.py: playbook
    ("playbook folder not a category", "rules", lambda r: shutil.copytree(r / "playbook" / "env", r / "playbook" / "extra"),
     "not a category heading"),
    ("card not in the index", "rules", lambda r: shutil.copy(r / "playbook" / "env" / "trt-plugin-build-hygiene.md",
                                                             r / "playbook" / "env" / "plugin-hygiene-copy.md"), "not listed in playbook/INDEX.md"),
    ("same technique twice", "rules", lambda r: shutil.copy(r / "playbook" / "env" / "trt-plugin-build-hygiene.md",
                                                            r / "playbook" / "host" / "trt-plugin-build-hygiene.md"), "same technique as"),
    ("card lost a section", "rules", lambda r: edit(r / "playbook" / "env" / "trt-plugin-build-hygiene.md", "## Side effects\n", ""),
     "missing section '## Side effects'"),
    ("index links a missing card", "rules", lambda r: append(r / "playbook" / "INDEX.md", "- [ghost](env/ghost.md): never | ✓0 ✗0\n"),
     "env/ghost.md does not exist"),
    ("Evidence line without a verdict", "rules", lambda r: append(r / "playbook" / "env" / "trt-plugin-build-hygiene.md", "- 2026-09: unmarked\n"),
     "Evidence line must start with"),
    ("index count differs from Evidence", "rules", lambda r: append(r / "playbook" / "env" / "trt-plugin-build-hygiene.md", "- ✗ 2026-09: failed\n"),
     "trt-plugin-build-hygiene.md says ✓1 ✗0, but its Evidence has ✓1 ✗1"),
    # check_rules.py: repetition and private terms
    ("line repeated across rules", "rules", lambda r: append(r / "rules" / "night.md", next(
        l for l in read(r / "rules" / "experiment.md").splitlines() if l.startswith("- ") and len(l) > 80) + "\n"),
     "repeats a line from"),
    ("private term in the rules", "rules", lambda r: (write(r.parent / "lx-work" / "PROJECT.md", "- Private terms: Frobnicator\n"),
                                                      append(r / "rules" / "naming.md", "Frobnicator\n")), "private term 'Frobnicator'"),
]


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clean = {"rules": build_rules(tmp / "clean-rules"), "work": build_work(tmp / "clean-work")}
        for kind, root in clean.items():
            code, out = run_checker(root)
            if code != 0:
                failures.append(f"clean {kind} fixture fails the checker:\n{out}")
        if failures:  # probes on a failing fixture prove nothing
            print("\n".join(failures))
            return 1
        for n, (name, kind, change, expect) in enumerate(PROBES):
            base = tmp / f"probe-{n}"
            shutil.copytree(clean[kind].parent.parent, base)
            root = base / PROJECT / clean[kind].name
            try:
                change(root)
            except AssertionError as e:
                failures.append(f"{name}: {e}")
                continue
            code, out = run_checker(root)
            if code != 1 or expect not in out:
                failures.append(f"{name}: expected exit 1 with '{expect}', got exit {code}:\n{out.strip() or '(no output)'}")
            shutil.rmtree(base)
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"{len(PROBES) - len(failures)} of {len(PROBES)} probes caught; both clean fixtures pass." if not failures
          else f"{len(failures)} of {len(PROBES)} probes failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
