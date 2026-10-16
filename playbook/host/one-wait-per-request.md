---
tier: on-demand
read_when: profiling shows the symptom below
---
# One wait per request

- Category: host
- Applies to: PyTorch or TensorRT services whose request path reads small GPU results (flags, counts, sizes) back several times

## Symptom
The runtime API trace shows many stream synchronizes per request, each waiting for a small value the next CPU decision needs; with several workers the waits grow and hold host cores, since CUDA's default wait spins.

## Technique
Queue all the GPU work that does not depend on a CPU decision, copy every small result into one pinned buffer, and wait once; then make the decisions on the host. Keep data the next stage uses on the GPU instead of copying it out and back. Where a CPU copy is still needed later, queue it with an event and wait on the event only where it is read. Count the synchronizes per request before and after in the trace.

## Effect
One worker: about -1.2 ms on an 8.9 ms request (descriptive, across days); the main gain is under load, where each removed wait frees a spinning core.

## Side effects
A copy queued without a wait is valid only after its event: every reader must wait on it, offline tools included.

## Does not apply
- A short path with one engine call and one wait already: removing a stream sync there was inside noise.

## Evidence
- ✓ 2026-10: two-model mask service (segmentation transformer and classifier), RTX 5090: three waits on the request path down to one, masks kept on the GPU: bench 8.92 -> 7.70 ms (descriptive, not an A/B), outputs L0.
- ✗ 2026-10: image background-type CNN classifier service, RTX 5090, one engine call: one stream sync and fewer copies, -0.26 ms (-2.9%) inside noise; dropped.
