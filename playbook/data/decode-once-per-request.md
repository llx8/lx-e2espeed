---
tier: on-demand
read_when: profiling shows the symptom below
---
# Decode each input once per request

- Category: data
- Applies to: requests that reference the same encoded image or mask several times (index lists, repeated layers)

## Symptom
The decode stage takes much longer on some requests than their distinct inputs explain; the trace shows the same bytes decoded again.

## Technique
Within one request, key decoded inputs by their index or a hash of the bytes and decode each distinct one once; copy only where a consumer modifies it in place.

## Effect
-10.45 ms (-16.6%) on average, about 20 ms on the requests that repeat a large PNG; L0.

## Side effects
The decoded image is shared: an in-place change by one consumer leaks into the others unless copied.

## Does not apply
- Requests whose inputs are all distinct.

## Evidence
- ✓ 2026-10: two-model mask service, index lists that repeat one 2048-px PNG mask: -10.45 ms (-16.6%), L0.
