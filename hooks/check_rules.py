"""Checks for the lx-e2espeed repo as a whole, called by check_docs.py.

Keeps the entry doc, the doc map, the hook's phase routing, and the playbook consistent, stops
rules from repeating each other, and keeps every project's private terms out of the rules.
"""
import re
from pathlib import Path

MAP_REF = re.compile(r"`((?:rules|hardware|playbook|templates|hooks|vendor|skills)/[^`]*)`")
ROUTED_DOC = re.compile(r"lx-e2espeed/(rules/[a-z0-9-]+\.md)")
CARD_LINK = re.compile(r"\]\(([a-z0-9-]+/[a-z0-9-]+\.md)\)")
# An index line's counts restate the card's Evidence, so they are recomputed from it on every check.
CARD_COUNTS = re.compile(r"\]\(([a-z0-9-]+/[a-z0-9-]+\.md)\).*\| ✓(\d+) ✗(\d+)\s*$", re.M)
VERDICT = re.compile(r"- ([✓✗]) ")  # an Evidence line: worked or failed
CATEGORY = re.compile(r"^## ([a-z0-9-]+)\s*$", re.M)
LIST_MARKER = re.compile(r"^\s*(?:[-*>]|\d+\.)\s+")
CARD_SECTIONS = ("## Symptom", "## Technique", "## Effect", "## Side effects",
                 "## Does not apply", "## Evidence")
# CLAUDE.md is injected every session; losing one of these sections silently changes every session.
ENTRY_SECTIONS = ("## Layout", "## Session start", "## Hard rules", "## Defaults", "## Working loop", "## Doc map")
DUP_MIN = 60    # repeated lines at least this long count as duplication
TERM_MIN = 3    # shorter private terms match too many ordinary words


def check_entry(root, rep):
    entry = (root / "CLAUDE.md").read_text(encoding="utf-8")
    lines = set(entry.splitlines())
    for section in ENTRY_SECTIONS:
        if section not in lines:
            rep.error("CLAUDE.md", f"lost its '{section}' section")
    # CLAUDE.md names the two maps; rules/INDEX.md names every rules and hardware doc, so docs can be added or
    # split without growing the file injected every session.
    rules_map = root / "rules" / "INDEX.md"
    index = rules_map.read_text(encoding="utf-8") if rules_map.is_file() else ""
    for sub in ("rules", "hardware"):
        for path in sorted((root / sub).glob("*.md")):
            rel = f"{sub}/{path.name}"
            if rel != "rules/INDEX.md" and f"`{rel}`" not in index:
                rep.error(rel, "not listed in rules/INDEX.md")
    for ref in ("rules/INDEX.md", "playbook/INDEX.md"):
        if f"`{ref}`" not in entry:
            rep.error("CLAUDE.md", f"doc map must list `{ref}`")
    for name, text in (("CLAUDE.md", entry), ("rules/INDEX.md", index)):
        for ref in MAP_REF.findall(text):
            if not (root / ref).exists():
                rep.error(name, f"`{ref}` does not exist")
    for skill in sorted(p for p in (root / "vendor" / "skills").glob("*") if p.is_dir()):
        if not (skill / "SKILL.md").is_file():
            rep.error(f"vendor/skills/{skill.name}/", "missing SKILL.md")
        if not any(f"`{skill.name}`" in p.read_text(encoding="utf-8") for p in (root / "rules").glob("*.md")):
            rep.error(f"vendor/skills/{skill.name}/", "no rules doc says when to load it")
    dispatch = root / "hooks" / "dispatch.py"
    for ref in ROUTED_DOC.findall(dispatch.read_text(encoding="utf-8")):
        if not (root / ref).is_file():
            rep.error("hooks/dispatch.py", f"routes a phase to {ref}, which does not exist")


def check_playbook(root, rep):
    index_path = root / "playbook" / "INDEX.md"
    if not index_path.is_file():
        rep.error("playbook/INDEX.md", "missing")
        return
    index = index_path.read_text(encoding="utf-8")
    categories, links = set(CATEGORY.findall(index)), set(CARD_LINK.findall(index))
    counts = {link: (int(ok), int(bad)) for link, ok, bad in CARD_COUNTS.findall(index)}
    first_seen = {}
    for folder in sorted(p for p in (root / "playbook").iterdir() if p.is_dir()):
        if folder.name not in categories:
            rep.error(f"playbook/{folder.name}/", "not a category heading in playbook/INDEX.md")
        for card in sorted(folder.glob("*.md")):
            rel = f"{folder.name}/{card.name}"
            if rel not in links:
                rep.error(f"playbook/{rel}", "not listed in playbook/INDEX.md")
            if first_seen.setdefault(card.stem, rel) != rel:
                rep.error(f"playbook/{rel}", f"same technique as playbook/{first_seen[card.stem]}; merge them")
            text = card.read_text(encoding="utf-8")
            for section in CARD_SECTIONS:
                if section not in text:
                    rep.error(f"playbook/{rel}", f"missing section '{section}'")
            check_evidence(rel, text, counts, rep)
    for link in sorted(links):
        if not (root / "playbook" / link).is_file():
            rep.error("playbook/INDEX.md", f"{link} does not exist")


def check_evidence(rel, text, counts, rep):
    """Each Evidence line starts with ✓ or ✗, and the card's index line counts exactly those."""
    lines = text.split("\n## Evidence\n", 1)[-1].split("\n## ", 1)[0].splitlines() if "\n## Evidence\n" in text else []
    verdicts = []
    for line in (l for l in lines if l.startswith("- ")):
        match = VERDICT.match(line)
        if not match:
            rep.error(f"playbook/{rel}", f"Evidence line must start with '- ✓ ' (worked) or '- ✗ ' (failed): \"{line[:50]}\"")
        else:
            verdicts.append(match.group(1))
    have = (verdicts.count("✓"), verdicts.count("✗"))
    if rel not in counts:
        rep.error("playbook/INDEX.md", f"the line for {rel} must end with '| ✓{have[0]} ✗{have[1]}'")
    elif counts[rel] != have:
        rep.error("playbook/INDEX.md", f"{rel} says ✓{counts[rel][0]} ✗{counts[rel][1]}, "
                                       f"but its Evidence has ✓{have[0]} ✗{have[1]}")


def check_duplicates(prose, rep):
    first_seen = {}
    for rel, lines in prose:
        for line in lines:
            text = LIST_MARKER.sub("", line).strip()  # a copied bullet still counts
            if len(text) >= DUP_MIN and first_seen.setdefault(text, rel) != rel:
                rep.error(rel, f"repeats a line from {first_seen[text]}: \"{text[:60]}...\"")


def private_terms(root):
    """Every registered project's private terms and directory name. The rules directory is shared, so a term of any
    project must stay out; setup.sh project registers each project in local/projects, one path per line."""
    registry = root / "local" / "projects"
    terms = []
    for line in registry.read_text(encoding="utf-8").splitlines() if registry.is_file() else []:
        if not line.strip():
            continue
        project = Path(line.strip())
        terms.append(project.name)
        doc = project / "lx-work" / "PROJECT.md"
        match = re.search(r"^- Private terms: (.+)$", doc.read_text(encoding="utf-8"), re.M) if doc.is_file() else None
        if match and not match.group(1).startswith("<"):
            # A parenthetical note, such as who confirmed the list and when, is not a term; left in, it would also
            # glue itself to the term before it, which then matched nothing.
            terms += re.sub(r"\([^)]*\)", ",", match.group(1)).split(",")
    return sorted({t.strip() for t in terms if len(t.strip()) >= TERM_MIN})


def check_private(texts, terms, rep):
    """texts: (rel, lines) for every text file in the rules repo."""
    if not terms:
        return
    pattern = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, terms)) + r")(?![\w-])", re.I)
    for rel, lines in texts:
        for n, line in enumerate(lines, 1):
            match = pattern.search(line)
            if match:
                rep.error(f"{rel}:{n}", f"contains the private term '{match.group(1)}'; "
                          "nothing about the project leaves lx-work")
                break
