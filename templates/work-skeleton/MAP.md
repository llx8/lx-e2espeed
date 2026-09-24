---
tier: on-demand
read_when: locating code on the end-to-end path, or choosing where to look next
---
# Code map

Pointers only (`path:function`). Never paste code here; pasted code goes stale.

## End-to-end path and time breakdown
Shares come from the stage timing at baseline; they decide where to look first.

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
