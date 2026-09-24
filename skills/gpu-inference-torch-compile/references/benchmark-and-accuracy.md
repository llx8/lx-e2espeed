# Benchmark and accuracy

## Required matrix

Run eager and every compiled candidate in separate fresh processes, with the same hardware, weights, inputs, pre/post tensor contract, seeds, and service configuration. Cover the common shape and every production bucket; the maximum image, ROI, and tile paths; empty-data and data-dependent branches; serial repeated requests and production concurrency; cold start with an empty cache and warm start with a compatible cache; and enough warm requests to see latency converge and graphs stabilize.

Never compare eager in the same process after a compiled run, where it inherits allocator or cache state.

## Record separately

Process and model initialization; compile and autotuning; first call or capture per bucket; warmup count before stability; steady-state end-to-end median and tail; candidate-module CUDA time; graph count, breaks, guard failures, and recompiles; peak CUDA allocated and reserved memory and host memory; cache size, hit evidence, and startup behavior; throughput and latency under production concurrency.

Use wall-clock time for service end to end and CUDA events or a profiler for GPU regions; synchronize only at measurement boundaries; keep verbose compile logs off while timing.

## Full accuracy gate

Run the project's complete production accuracy suite; a single `torch.testing.assert_close` on one output is not enough. Keep the pre/post tensor semantics and compare: final images under project-approved metrics; routing, shape, metadata, masks, pad/crop, blending, color order, and range; the first diverging intermediate tensor on failure; every bucket, tile boundary, empty or data-dependent path, and repeated-request state; non-finite counts and the project's determinism and repeatability rules.

A failed accuracy gate rejects the compile result. Never loosen tolerances to accept compiled output.

## Delivery decision

Report absolute and relative end-to-end results without a universal percentage threshold. Amortize cold start over the real worker lifecycle, and state the change in peak memory, cache lifecycle, compatibility risk, and maintenance cost. A warm microbenchmark gain alone does not justify deployment.
