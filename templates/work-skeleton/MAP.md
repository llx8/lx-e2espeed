---
tier: on-demand
read_when: locating code on the end-to-end path, or choosing where to look next
---
# Code map

Pointers only (`path:function`). Never paste code here; pasted code goes stale.

## End-to-end path and time breakdown
Shares come from the stage timing on the tree named below; they decide where to look first.
The checker flags this map once a kept experiment changes a file it points to: re-run the stage timing on best, then rewrite the breakdown, the hot spots, and this line.
- Measured on: <baseline commit, plus " + exp up to NNN" once the timed tree includes experiments; NNN is the highest kept or delivered one in it>

<Measurement conditions, in this paragraph because the line above holds only the stamp: date, tool, environment, scenario, number of calls, SM clock, and where the raw output is.>

    <ASCII flow: each stage with its device ([CPU] or [GPU]), dtype, and shape, and every copy between them>

| Step | Code | Share of e2e time |
|---|---|---|
| Input | <path:function> | |
| Preprocessing | | |
| Host-to-device copy | | |
| Model forward | | |
| Device-to-host copy | | |
| Postprocessing | | |

## Hot spots
- <path:function>: <why it matters, with the profile evidence>

## Notes
- <Surprises such as global state, hidden syncs, lazy initialization.>
