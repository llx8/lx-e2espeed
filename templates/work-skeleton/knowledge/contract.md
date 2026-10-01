---
tier: on-demand
read_when: changing anything between the decoded input and the returned output, building the harness or verify, or judging whether an output difference reaches downstream
---
# Contract

## Entry points
<Each function production calls, with its arguments and defaults, and the values production passes.>

## Input format
<Type, dtype, layout, color order, value range, and sizes, and each transform before the model.>

## Output format
<The exact structure returned: types, shapes, counts, order, and ranges.>

## What downstream can observe
<Which output properties reach the consumer, including any serialization, and which do not.>

## Modes and options
<Flags and defaults that change results, and which configuration production runs.>

## Open questions
- <Which output differences the consumer tolerates; the answers set the difference levels in PROJECT.md.>
