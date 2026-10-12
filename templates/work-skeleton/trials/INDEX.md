---
tier: on-demand
read_when: during a self-iteration run, and when reporting one to the user
---
# Trials

One row per trial of a self-iteration run (rules/self-iteration.md). `lx-e2espeed/exp.sh try` adds each row as running; park, drop, and adopt set the status. Fill in the main scenario's A/B against the trial's base, the difference level, and what the user must decide; NOTE.md has every scenario.
Trials start from the same base and do not build on each other, so their gains do not add up.
Before this file passes 100 lines, move old dropped and adopted rows to ARCHIVE.md: a longer doc needs a Contents list, and this one is read whole.

## Queue
Methods not tried yet, most promising first, each with its evidence; a blocked one says what it needs (a dependency, a user decision).

## Trials
| Trial | Status | main-scenario A/B vs base | Difference | Needs the user | Reason |
|---|---|---|---|---|---|
