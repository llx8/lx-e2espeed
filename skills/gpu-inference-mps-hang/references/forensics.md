# MPS hang forensics and operations

## Contents
- GPU coredump under MPS
- Coredump pitfalls
- How the scanner works
- Dead ends
- MPS operations

## GPU coredump under MPS

Under MPS, `cuda-gdb` cannot attach ("nvidia-cuda-mps-server is running"), and `nsys`/`ncu` record nothing for a kernel that never finishes (`ncu --replay-mode range` also fails to capture events under MPS). The MPS documentation ("Interaction with Tools") allows GPU coredumps, and they work:

```bash
# Worker process environment (inherited from the container: docker run -e or Dockerfile ENV)
CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
CUDA_COREDUMP_PIPE=/tmp/corepipe_%p          # trigger pipe, %p = pid
CUDA_COREDUMP_FILE=/tmp/core_%p.cuda
CUDA_COREDUMP_SHOW_PROGRESS=1                # progress on stderr, visible in container logs
CUDA_COREDUMP_GENERATION_FLAGS=skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory

# After the hang, trigger one worker only (write 1 MB; a short echo may stay buffered)
dd if=/dev/zero bs=1M count=1 > /tmp/corepipe_<pid>
# "coredump: All done" after about 7 s, a file of about 200 MB. Read it offline with cuda-gdb
# from the same CUDA version (no GPU needed):
cuda-gdb -batch -ex "target cudacore core_<pid>.cuda" \
    -ex "info cuda kernels" -ex "info cuda blocks" -ex "info cuda warps"
# Find the spin: switch to one block/thread and disassemble around the PC
#   -ex "cuda block (32,0,0) thread (0,0,0)" -ex "x/14i \$pc-0x60"
```

`info cuda kernels` gives the stuck kernel and its grid/block size; `info cuda blocks` gives the resident block count (300 of 340 running means the grid did not fit); in `info cuda warps`, one warp per block has a different active PC, and its disassembly is the software grid barrier:

```
MEMBAR.SC.GPU ; ERRBAR ; CGAERRBAR ; CCTL.IVALL ;
LDG.E.STRONG.GPU R20, desc[UR10][R18.64] ;
ISETP.NE.U32.AND P2, PT, R20, 0xaa, PT ;
@P2 BRA <backward>                          # thread 0 spins on a global counter
```

## Coredump pitfalls

- Triggering several workers at once truncates all dumps (cuda-gdb asserts `m_num_devices > 0`; `file` reports a corrupted program header). One client's coredump faults the other clients of the same server; trigger one worker and ignore the others' empty "No CUDA kernels" dumps.
- Killing the worker during the dump also truncates it; cuda-gdb requires no signals during the dump. Raise the dispatcher's worker and schedule timeouts (for example to 3600 s) while reproducing.
- After a coredump, that MPS server is in teardown (new clients get 807); rebuild the container.
- `CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1` also works, but the generation flags are finer; without skipping global memory a dump is several GB.
- Coredumps need no `SYS_PTRACE`; `py-spy` for Python stacks does.

## How the scanner works

1. Name prefilter (milliseconds): cubins inside an engine are plain ELF, and Myelin fused-kernel symbol names are plain text. The failing kernels are all fused normalizations: `__myl_Inor*` (InstanceNorm), `__myl_Gnor*` / `__myl_ReluGnor*` (GroupNorm), `__myl_*MeanMean*` (a two-pass reduction written out by hand). No such names usually means done.
2. Confirmation (seconds): run `cuobjdump -sass` in parallel on the matching cubin slices only, looking for one function containing both `YIELD` and `LDG.E.STRONG.GPU`. A 660 MB plan takes about 5 s; the old serial cuobjdump approach took minutes. In practice the prefilter missed nothing: a full disassembly of 530 cubins also found zero.

Scanning several production engines found 4 to 14 barrier kernels in each engine whose normalization was exported as ONNX `GroupNormalization` or `InstanceNormalization` (including through a registered native group-norm symbolic), and none where GroupNorm already went through a plugin.

## Dead ends

- Bisecting TensorRT aux streams (`max_aux_streams=0`), image-decoder backends, warmup coverage, container start order, or single-process loops reproduces nothing: the trigger is kernels from several processes competing for SMs.
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` partitioning makes it worse: the engine sizes its grid to the whole GPU, so a partitioned client can never fit it and hangs on the first warmup inference. If partitioning is required, build the engine under the same partition.
- Internal TensorRT options such as `TRT_INTERNAL_OPTIONS` / `-kgen:fdl=0` do not affect PDL, and PDL is not the cause.
- Capture worker stacks with `py-spy dump`. When matching processes with `pgrep -f`, bracket one character of the pattern (`"[s]rv ..."`) so `pkill -f` does not kill your own shell.
- In stress scripts, tell a hang apart from request errors: a dispatcher timeout returning 503 still counts as completed; judge a hang by the time since the last successful request.

## MPS operations

- Each container runs its own `nvidia-cuda-mps-control -d`, so several pods on one GPU each have an MPS server. On the host, `nvidia-smi` shows `nvidia-cuda-mps-server` rather than the workers, and memory is charged to the server (type M+C).
- A client must be idle before it is killed. To force it, first `echo "terminate_client <server pid> <client pid>" | nvidia-cuda-mps-control`, then SIGKILL. A dispatcher that cannot do this needs a worker timeout longer than any possible request, or one slow request takes the whole service down.
- Point `CUDA_MPS_LOG_DIRECTORY` at a writable directory to get `control.log` and `server.log`; "Server teardown in progress" or "status 807" there is the cascade above.
- An MPS server does not heal after an error: `echo quit | nvidia-cuda-mps-control` and restart the daemon (in a container, recreate the container).
