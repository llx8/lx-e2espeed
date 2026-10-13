#!/usr/bin/env python3
"""Check an lx-e2espeed or lx-work repo against rules/writing.md and rules/naming.md.

Usage: check_docs.py <repo-root>
Prints one line per problem. Exits 1 on any error.
The repo kind comes from its layout: CLAUDE.md + rules/ (lx-e2espeed) or STATUS.md (lx-work).
"""
import os
import re
import sys
from pathlib import Path

from check_rules import (check_duplicates, check_entry, check_playbook, check_private,
                         private_terms)
from check_work import check_work_layout

TIERS = {"always", "once", "on-demand"}
UNLIMITED_DOCS = {"ARCHIVE.md"}  # grepped, never read whole: no line limit
# local/ is untracked scratch space; .claude/ is Claude Code's own local config; vendor/ holds
# files copied unchanged from upstream, which follow their own format (vendor/skills/UPSTREAM.md).
SKIP_DIRS = {".git", "local", "__pycache__", ".claude", "vendor"}
TEXT_MAX = 2_000_000  # bytes; larger files are data, not docs or code
CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef]")  # CJK, kana, hangul, fullwidth forms
KEBAB = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
SNAKE = re.compile(r"[a-z0-9]+(_[a-z0-9]+)*")
FIXED = re.compile(r"[A-Z]+")
PROJECT_PATCH = re.compile(r"change(\.[A-Za-z0-9_][A-Za-z0-9_.-]*)?\.patch")  # change.patch, change.<linked repo>.patch
DOC_MAX = 150
TOC_OVER = 100  # docs longer than this need a "## Contents" list near the top
# The SessionStart hook injects both into every session; Claude Code cuts hook output above about 10 KB.
BYTE_LIMITS = {("rules", "CLAUDE.md"): 6000, ("work", "STATUS.md"): 3000}


class Report:
    def __init__(self):
        self.errors = []

    def error(self, where, msg):
        self.errors.append(f"{where}: {msg}")


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        base = Path(dirpath)
        for name in dirnames + sorted(filenames):
            path = base / name
            yield path, path.relative_to(root).as_posix()


def line_limit(rel, kind):
    name = rel.rsplit("/", 1)[-1]
    if name in UNLIMITED_DOCS:
        return None
    if kind == "rules":
        if rel.startswith("templates/work-skeleton/"):
            return line_limit(rel[len("templates/work-skeleton/"):], "work")
        fixed = {"CLAUDE.md": 100, "CHANGELOG.md": 100, "templates/card.md": 40, "templates/exp-note.md": 50}
        if rel in fixed:
            return fixed[rel]
        return 40 if rel.startswith("playbook/") and rel != "playbook/INDEX.md" else DOC_MAX
    if rel in ("STATUS.md", "FEEDBACK.md"):
        return 60
    return 50 if rel.split("/")[0] in ("exp", "trials") and name == "NOTE.md" else DOC_MAX


def byte_limit(rel, kind):
    if kind == "rules" and rel.startswith("templates/work-skeleton/"):
        return byte_limit(rel[len("templates/work-skeleton/"):], "work")
    return BYTE_LIMITS.get((kind, rel))


def split_frontmatter(lines):
    """Return (fields, index of the first body line), or (None, 0) without frontmatter."""
    if not lines or lines[0].strip() != "---":
        return None, 0
    fields = {}
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return fields, i + 1
        key, _, value = lines[i].partition(":")
        fields[key.strip()] = value.strip()
    return None, 0


def outside_fences(lines):
    out, fenced = [], False
    for line in lines:
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            out.append(line)
    return out


def check_name(path, rel, kind, rep):
    name = path.name
    if name.startswith("."):
        return
    if path.is_dir():
        if kind == "work" and rel.split("/")[0] in ("exp", "trials") and rel.count("/") == 1:
            return  # experiment and trial directories are checked in check_work_layout
        if not (KEBAB.fullmatch(name) or SNAKE.fullmatch(name)):
            rep.error(rel + "/", "directory names are lowercase kebab-case (snake_case for Python packages)")
        return
    stem, suffix = path.stem, path.suffix
    if suffix == ".py":
        ok = SNAKE.fullmatch(stem)
    elif suffix == ".md":
        ok = FIXED.fullmatch(stem) or KEBAB.fullmatch(stem)
    elif kind == "rules" or suffix == ".sh":
        ok = KEBAB.fullmatch(stem)
    else:
        return  # data files in lx-work are named by the tools that write them
    if not ok:
        rep.error(rel, "file name breaks rules/naming.md")


def is_project_code(rel):
    parts = rel.split("/")
    return (parts[0] in ("exp", "trials") and len(parts) > 2
            and (parts[2] == "candidate" or bool(PROJECT_PATCH.fullmatch(parts[2]))))


def check_file(path, rel, kind, rep, prose, texts):
    suffix = path.suffix
    try:
        if path.stat().st_size > TEXT_MAX:
            return
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return  # binary data
    texts.append((rel, lines))
    if kind == "work" and is_project_code(rel):
        return  # the project's own code keeps its language: patches and candidate copies
    for n, line in enumerate(lines, 1):
        if CJK.search(line):
            rep.error(f"{rel}:{n}", "non-English text; the lx repos are English only, and non-English model data goes to lx-work/local/")
            break
    if suffix != ".md":
        return
    fields, body = split_frontmatter(lines)
    if fields is None:
        rep.error(rel, "missing frontmatter with tier and read_when")
    else:
        if fields.get("tier") not in TIERS:
            rep.error(rel, f"tier must be one of {sorted(TIERS)}")
        if not fields.get("read_when"):
            rep.error(rel, "read_when is empty")
    size, size_limit = path.stat().st_size, byte_limit(rel, kind)
    if size_limit and size > size_limit:
        rep.error(rel, f"{size} bytes, limit {size_limit}; the SessionStart hook injects it, and longer hook output is cut")
    limit = line_limit(rel, kind)
    if limit is None:
        return
    if len(lines) > limit:
        rep.error(rel, f"{len(lines)} lines, limit {limit}; split or cut it")
    if len(lines) > TOC_OVER and "## Contents" not in lines[body:body + 20]:
        rep.error(rel, f"over {TOC_OVER} lines without a '## Contents' list near the top")
    if kind == "rules" and not rel.startswith("templates/") and rel != "CHANGELOG.md":
        prose.append((rel, outside_fences(lines[body:])))


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    root = Path(argv[1]).resolve()
    if (root / "CLAUDE.md").is_file() and (root / "rules").is_dir():
        kind = "rules"
    elif (root / "STATUS.md").is_file():
        kind = "work"
    else:
        print(f"{root}: not an lx-e2espeed or lx-work repo")
        return 1
    rep, prose, texts = Report(), [], []
    for path, rel in walk(root):
        check_name(path, rel, kind, rep)
        if path.is_file():
            check_file(path, rel, kind, rep, prose, texts)
    if kind == "rules":
        check_entry(root, rep)
        check_playbook(root, rep)
        check_duplicates(prose, rep)
        check_private(texts, private_terms(root), rep)
    else:
        check_work_layout(root, rep, Path(__file__).resolve().parent.parent)
    for msg in rep.errors:
        print(f"error: {msg}")
    if rep.errors:
        print(f"{len(rep.errors)} error(s) in {root.name}. See lx-e2espeed/rules/writing.md and naming.md.")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
