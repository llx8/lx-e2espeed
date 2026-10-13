---
tier: on-demand
read_when: profiling shows the symptom below
---
# Reading a hang under MPS

- Category: serving
- Applies to: several worker processes on one GPU under NVIDIA MPS, hung in a kernel

## Symptom
Workers stop answering, GPU-Util 100% at low power; `cuda-gdb` cannot attach ("nvidia-cuda-mps-server is running"), and `nsys` or `ncu` record nothing for a kernel that never ends.

## Technique
Take a user-triggered GPU coredump of one worker and read it offline. Start the workers with `CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1`, `CUDA_COREDUMP_PIPE=/tmp/corepipe_%p`, `CUDA_COREDUMP_FILE=/tmp/core_%p.cuda`, `CUDA_COREDUMP_SHOW_PROGRESS=1`, and `CUDA_COREDUMP_GENERATION_FLAGS=skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory` (without the skips a dump is gigabytes). Once hung, trigger exactly one worker: `dd if=/dev/zero bs=1M count=1 > /tmp/corepipe_<pid>` (an `echo` can stay buffered); about 7 s and 200 MB. Read it with the same CUDA version's `cuda-gdb -batch -ex "target cudacore core_<pid>.cuda" -ex "info cuda kernels" -ex "info cuda blocks" -ex "info cuda warps"`: kernels names the stuck kernel and its grid, blocks the resident count (340 launched and 300 running means the grid never fit), warps one thread per block whose PC differs from the rest; `cuda block (<b>,0,0) thread (0,0,0)` and `x/14i $pc-0x60` show the spin (`YIELD`, `MEMBAR.SC.GPU`, `LDG.E.STRONG.GPU`, a backward branch): a software grid barrier. Python stacks come from `py-spy dump` on the worker PIDs.
Operations around it: a worker killed with GPU work pending tears the MPS server down, every later client fails with CUDA error 807, and the service dies; so the dispatcher's worker timeout exceeds the longest request, a client is ended with `terminate_client <server pid> <client pid>` through `nvidia-cuda-mps-control` before any SIGKILL, and a broken server is quit (`echo quit | nvidia-cuda-mps-control`) and started again, since it never heals. With `CUDA_MPS_LOG_DIRECTORY` on a writable directory, server.log shows "Server teardown in progress" and "status 807".

## Effect
Root cause from one dump, without reproducing the hang single-process (it does not reproduce there).

## Side effects
The MPS server is finished after a dump (teardown, new clients get 807), so the container is recreated afterwards. A worker killed during the dump truncates it: raise the worker timeout to an hour while reproducing.

## Does not apply
- Triggering several workers at once: each dump faults the others and every file comes out truncated (cuda-gdb fails an `m_num_devices > 0` assertion); the untriggered workers write empty "No CUDA kernels" dumps, which are ignored.
- Bisecting the pipeline single-process (aux streams, decoder backend, warmup): the kernels run when they own the GPU.

## Evidence
- ✓ 2026-09: 6 workers of an image service hung under MPS; one coredump showed a Myelin normalization kernel's grid of 340 blocks with 300 resident, spinning on a grid barrier; fixed with a GroupNorm plugin (card `mps-grid-barrier`).
