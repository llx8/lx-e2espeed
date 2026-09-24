---
tier: on-demand
read_when: the user corrects how the work is done (write), or the rules review at wrap-up (read)
---
# Feedback on the rules

One line per correction, written when it happens: what the AI did, what the user wanted, and which rule was missing or unclear.
The wrap-up rules review starts here, so later sessions do not have to remember it.

- Adding: grep this file for a similar line first. If one exists, raise its count; otherwise append a new line. Neither needs the whole file read.
- A repeated correction is the strongest sign that a rule is wrong, so the count matters more than the wording.
- At 60 lines, hold the rules review with the user early instead of waiting for wrap-up.
- Once an entry has become a rule change, delete it here; CHANGELOG.md and git history keep it.

Entry format: `- <YYYY-MM-DD>: <what happened> → <what the user wanted> (rule: <file or "missing">) x<count>`
