---
tier: on-demand
read_when: profiling shows the symptom below
---
# Blocking sync for host waits

- Category: host
- Applies to: CUDA services with several worker processes per GPU whose libraries share the primary context (PyTorch, TensorRT, Triton, nvImageCodec), where host CPU under load matters

## Symptom
Under load each worker holds one core (about 100%, 98% user time) whatever its throughput, so CPU grows with the worker count while requests/s stay flat: CUDA's default host wait spins in stream synchronizes and device-to-host copies.

## Technique
Before any CUDA context exists, so before importing torch or any other CUDA user, set `CU_CTX_SCHED_BLOCKING_SYNC` (0x04) on the device's primary context through the driver API (ctypes on libcuda.so.1: `cuInit`, `cuDeviceGet`, `cuDevicePrimaryCtxSetFlags`). Every library on the primary context inherits it. Raise at import on any error: once the context exists the call returns 708 and changes nothing.
Measure CPU per worker (`os.times` over the load) and requests/s at 1, 2, and the production worker count, and single-request latency, before and after.

## Effect
6 workers with MPS: CPU 599% -> 114% (L20) and 598% -> 167% (RTX 5090), requests/s and latency within 1%; 1 worker 100% -> 55-65%. What remains is the real host work (about 7 ms per request in one service). Single request +0.08 ms (L20, above its 0.06 floor), +0.21 ms on the RTX 5090 inside noise; outputs L0.

## Side effects
Process-wide: every wait in the process sleeps and pays an OS wake-up. Set in the service's entry module, it does not reach offline tools that import the libraries another way.

## Does not apply
- None measured yet. It frees cores but removes no wait: where waits cost latency, see card `one-wait-per-request`.

## Evidence
- ✓ 2026-10: camera-calibration CNN-transformer service with Triton solver kernels and NVJPG decode, 1, 2, 6 workers with MPS, L20 and RTX 5090: CPU 599% -> 114% and 598% -> 167% at 6 workers, requests/s within 1%, single request +0.08 ms (L20), L0.
