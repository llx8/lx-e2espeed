---
tier: on-demand
read_when: asking why something was decided, or recording a new decision
---
# Decisions

Every decision made with the user, newest last. A superseded decision stays, marked and pointing to its replacement, until this file nears its limit; then delete it (git history keeps it).

Entry format:

    ## D-NNN: <title>
    - Date: <YYYY-MM-DD>
    - Status: accepted | superseded by D-NNN
    - Decision: <what>
    - Reason: <why>
