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
Start the MPS control daemon before the workers (on by default with a switch to turn it off) and give every worker the same `CUDA_MPS_PIPE_DIRECTORY`; check in the MPS control log that each worker's PID connects as a client.
Every project's Dockerfile does this from its first change, whatever production's image does (the user, 2026-10-20: merged images without it were caught by operations' load tests several times). The checker fails a project whose tracked Dockerfile starts no daemon, itself or in a script its ENTRYPOINT or CMD runs; one the user exempts is named in PROJECT.md as `- Without MPS: <path> (<reason>)`.
Before delivery, run the image's CMD in a container and check that an MPS server starts: one delivered image had no daemon line and its load test stayed flat from 2 workers. Keep the pipe directory path short: it is a Unix socket, and a path over 108 bytes makes the daemon exit at once. Compare every response with a single-process run without MPS.

## Effect
3 workers: throughput 2.6x to 3.1x of one worker (+50% to +91% over 3 workers without MPS), P50 at the single-worker level. One worker: no change (within 5%, 0.05 to 0.15 ms slower). On an L20 with 12 workers: 5.4x to 7.2x of one worker for light image services, 3.5x for a host-heavy one; the peak comes at 12 workers on 16 host threads.

## Side effects
Every kernel shares one server: a kernel with a grid barrier can hang all workers (card `mps-grid-barrier`), and a worker killed on a timeout can take the server down (error 807). Each worker keeps its own memory (about 1.3 GB RAM and 1 GB GPU memory in one service).

## Does not apply
- Requests whose bottleneck is outside the workers: light requests stopped at the same rate for 2 and 3 workers with or without MPS (the dispatcher or client), so they show nothing.
- A single worker.
- Requests whose host work is mostly spinning waits: a mask service with 21 stream synchronizes per request stopped at 3.5x from 8 workers, the CPU full of spin-waits (cards `one-wait-per-request`, `blocking-sync-wait`).
- Workers past what one NVJPG engine decodes on a GPU with one: JPEG services on an RTX 5090 stopped at about 980 requests/s from 4 workers (card `gpu-jpeg-decode`).

## Evidence
- ✓ 2026-09: image-to-guidance-map CNN service, RTX 5090, 1 to 3 workers, closed-loop load through the real HTTP framework: requests with RoIs +15% to +91% throughput at 2 to 3 workers, 32400 responses byte-identical to the reference.
- ✓ 2026-10: image background-type CNN classifier service, L20, nvImageCodec 0.9, 1 to 16 workers: 359 -> 2589 req/s at 12 (7.2x), limited by the four NVJPG engines.
- ✓ 2026-10: face classification CNN service, the same setup: 303 -> 1714 req/s at 12 (5.7x), SM-bound.
- ✓ 2026-10: heatmap line-segmentation CNN service, the same setup: 227 -> 1597 req/s at 12 (7.0x), 1685 at 16, SM-bound.
- ✓ 2026-10: ViT image feature encoder service, the same setup: 312 -> 1696 req/s at 12 (5.4x), SM-bound.
- ✓ 2026-10: two-model mask service, the same setup: 95 -> 330 req/s at 12 (3.5x), host-bound by spinning synchronizes.
- ✓ 2026-10: camera-calibration CNN-transformer service with Triton solver kernels, 6 workers, without -> with MPS: RTX 5090 116 -> 245 req/s (2.1x), L20 109.5 -> 177.6 (1.6x), the image's CMD daemon line added after delivery.
