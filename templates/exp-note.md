---
tier: on-demand
read_when: reproducing, reverting, or building on this experiment
---
# NNN-name

- Date: <YYYY-MM-DD>
- Base: <baseline commit and the kept experiments this one builds on; filled by exp.sh>
- Status: running | kept | delivered | dropped | parked | adopted

## Evidence
<The profile result that points here, and this step's share of end-to-end time from MAP.md.>

## Hypothesis
<What should get faster and why.>

## Plan
<Files and functions to change as path:function, contracts touched and their downstream impact, how verify covers the change, expected gain.>

## Result
<Three sets per scenario from the A/B's summary lines, with every round in this folder's ab-*.json: baseline (PROJECT.md, Metric), before (best, the A/B's best side; one row per A/B if they differ), and after (each variant). The change against the baseline is descriptive.>

| | median | P90 | P99 | peak memory | vs baseline | vs best |
|---|---|---|---|---|---|---|
| baseline | | | | | | |
| before (best) | | | | | | |
| <variant> | | | | | | |

Difference: <level L0 to L3, magnitude and extent, scenario and input size>
Verify: <pass or fail, with details>

## Conclusion
<Keep or drop, and what was learned.>
