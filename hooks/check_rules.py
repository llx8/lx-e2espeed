"""Checks for the lx-e2espeed repo as a whole, called by check_docs.py.

Keeps the entry doc, the doc map, the hook's phase routing, and the playbook consistent, stops
rules from repeating each other, and keeps a project's private terms out of a project clone.
"""
import re

MAP_REF = re.compile(r"`((?:rules|hardware|playbook|templates|hooks|vendor|skills)/[^`]*)`")
SKILL_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
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
# Agent Skills limits: the description is what every session loads, SKILL.md what each use loads.
SKILL_DESC_MAX, SKILL_MD_MAX, SKILL_REF_MAX = 1024, 500, 300
TERM_MIN = 3    # shorter private terms match too many ordinary words


def check_entry(root, rep):
    entry = (root / "CLAUDE.md").read_text(encoding="utf-8")
    lines = set(entry.splitlines())
    for section in ENTRY_SECTIONS:
        if section not in lines:
            rep.error("CLAUDE.md", f"lost its '{section}' section")
    for sub in ("rules", "hardware"):
        for path in sorted((root / sub).glob("*.md")):
            rel = f"{sub}/{path.name}"
            if f"`{rel}`" not in entry:
                rep.error(rel, "not listed in the CLAUDE.md doc map")
    if "`playbook/INDEX.md`" not in entry:
        rep.error("CLAUDE.md", "doc map must list `playbook/INDEX.md`")
    for ref in MAP_REF.findall(entry):
        if not (root / ref).exists():
            rep.error("CLAUDE.md", f"`{ref}` does not exist")
    for skill in sorted(p for p in (root / "skills").glob("*") if p.is_dir()):
        if not (skill / "SKILL.md").is_file():
            rep.error(f"skills/{skill.name}/", "missing SKILL.md")
        if not any(f"`{skill.name}`" in p.read_text(encoding="utf-8") for p in (root / "rules").glob("*.md")):
            rep.error(f"skills/{skill.name}/", "no rules doc says when to load it")
    dispatch = root / "hooks" / "dispatch.py"
    for ref in ROUTED_DOC.findall(dispatch.read_text(encoding="utf-8")):
        if not (root / ref).is_file():
            rep.error("hooks/dispatch.py", f"routes a phase to {ref}, which does not exist")


def check_skill_doc(rel, lines, fields, rep):
    """Skills follow the Agent Skills format instead of the doc tiers: SKILL.md carries name and description."""
    parts = rel.split("/")
    if parts[-1] == "SKILL.md":
        if fields is None:
            rep.error(rel, "needs frontmatter with name and description")
        else:
            if fields.get("name") != parts[1] or not SKILL_NAME.fullmatch(parts[1]):
                rep.error(rel, "frontmatter name must equal its kebab-case directory name")
            description = fields.get("description", "")
            if not description or description.startswith((">", "|")):
                rep.error(rel, "description must be one non-empty line")
            elif ": " in description and not description.startswith(('"', "'")):
                rep.error(rel, "quote the description: an unquoted ': ' breaks the YAML frontmatter")
            elif len(description) > SKILL_DESC_MAX:
                rep.error(rel, f"description is {len(description)} characters, limit {SKILL_DESC_MAX}")
        if len(lines) > SKILL_MD_MAX:
            rep.error(rel, f"{len(lines)} lines, limit {SKILL_MD_MAX}; move detail into references/")
    elif len(lines) > SKILL_REF_MAX:
        rep.error(rel, f"{len(lines)} lines, limit {SKILL_REF_MAX}; split it")
    if len(lines) > 100 and "## Contents" not in lines[:30]:
        rep.error(rel, "over 100 lines without a '## Contents' list near the top")


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
    """The project's private terms, when this clone sits in a project next to lx-work."""
    project = root.parent / "lx-work" / "PROJECT.md"
    if not project.is_file():
        return []
    match = re.search(r"^- Private terms: (.+)$", project.read_text(encoding="utf-8"), re.M)
    terms = [] if not match or match.group(1).startswith("<") else match.group(1).split(",")
    terms.append(root.parent.name)
    return sorted({t.strip() for t in terms if len(t.strip()) >= TERM_MIN})


def check_private(texts, terms, rep):
    """texts: (rel, lines) for every text file in the clone."""
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
