---
tier: on-demand
read_when: profiling shows the symptom below
---
# MPS for several workers per GPU

- Category: serving
- Applies to: services that run several worker processes on one GPU, each with its own CUDA context

## Symptom
Throughput grows far less than the worker count and P50 rises with it: without MPS the workers' contexts take turns on the GPU (3 workers gave 1.4x to 2.0x of one).

## Technique
Start the MPS control daemon before the workers (in the image's CMD, on by default with a switch to turn it off) and give every worker the same `CUDA_MPS_PIPE_DIRECTORY`; check in the MPS control log that each worker's PID connects as a client. Keep the pipe directory path short: it is a Unix socket, and a path over 108 bytes makes the daemon exit at once. Compare every response with a single-process run without MPS.

## Effect
3 workers: throughput 2.6x to 3.1x of one worker (+50% to +91% over 3 workers without MPS), P50 at the single-worker level. One worker: no change (within 5%, 0.05 to 0.15 ms slower).

## Side effects
Every kernel shares one server: a kernel with a grid barrier can hang all workers (card `mps-grid-barrier`), and a worker killed on a timeout can take the server down (error 807). Each worker keeps its own memory (about 1.3 GB RAM and 1 GB GPU memory in one service).

## Does not apply
- Requests whose bottleneck is outside the workers: light requests stopped at the same rate for 2 and 3 workers with or without MPS (the dispatcher or client), so they show nothing.
- A single worker.

## Evidence
- ✓ 2026-09: image-to-guidance-map CNN service, RTX 5090, 1 to 3 workers, closed-loop load through the real HTTP framework: requests with RoIs +15% to +91% throughput at 2 to 3 workers, 32400 responses byte-identical to the reference.
