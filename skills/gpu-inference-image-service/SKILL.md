---
name: gpu-inference-image-service
description: "Optimize a GPU image-inference service whose repo pins a separate algorithm repo: the change sequence that worked across such projects, the user's usual answers, the decisions that differed, and the delivery order."
---

# Optimize an image service with a pinned algorithm repo

The common part of several of the user's projects (2026-09 to 2026-10), so a new one starts from what worked and the user decides only what differs. Company names, images, repos, and reference code for each step: `lx-e2espeed/local/site.md`.

## Shape of these projects
- Service repo: an API module with Init, PreProcess, AlgProcess, and PostProcess, run by a compiled service framework (a dispatcher plus one worker library per Python version); a dependency file that pins the framework and the algorithm repo (repo, branch, commit); a script that clones both and runs `git lfs pull`; a Dockerfile on the company's runtime image; CI from a shared template.
- Algorithm repo: model, pre- and postprocessing, engine export and loading. Most of the time is spent there, so it changes first.

## Onboarding: the user's usual answers
Propose these in onboarding step 3 as the starting point, one or two at a time; record only what the user confirms.
- Layout: when the algorithm repo needs its own model work (precision, plugins, engine shipping), onboard it as its own project first, then the service, which picks it up through the pin. Otherwise link the algorithm checkout into the service project (a symlink at the project top to the sibling checkout on a dev branch, recorded in `- Linked repos:`), so one experiment changes both.
- Baseline: the speed of the released version in the previous production image, run there through a run-image tool (when it cannot run, the repo README's numbers); the outputs of the baseline commit, or of the model at full precision (PyTorch FP32, TF32 off) when production's outputs are known to be wrong.
- Metric: warm end-to-end latency, request bytes to response bytes, one worker; cold start reported beside it.
- Levels: L1 is the spread of N engine rebuilds from one ONNX (card `engine-build-variance`); above L0 the user judges each result, no thresholds.
- MPS: on in production, about four workers per GPU (ask the count), so no grid-barrier kernel may ship (`gpu-inference-mps-hang`). The service image starts the MPS daemon by default in its start command, before the server, even at one worker; `ENABLE_MPS=0` skips it (the user, 2026-10-12; reference Dockerfile in `local/site.md`).
- Input NVJPG does not take (other formats, progressive, truncated): fail fast with an error response, never route it to the original decoder (the user, 2026-10-12).
- Production GPUs: ask which, as compute capabilities, before building anything that ships.
- Environment: the runtime image's versions locally (a conda env from its pip freeze); never the old CUDA major.
- Interface: the algorithm may take a CUDA tensor beside its NumPy input when that removes copies (the user prefers the faster path); the NumPy path keeps working.

## The usual sequence
One experiment each, in this order unless the profile says otherwise; skip a step whose evidence is absent. Gains come from different projects and do not add up.

| Step | Evidence first | Change | Seen |
|---|---|---|---|
| Runtime port | the new image's TensorRT rejects the code | TensorRT 8 binding API to the tensor-name API (`set_input_shape`, `set_tensor_address`, `execute_async_v3`), raising when a call returns False | needed for the move |
| JPEG on NVJPG | decode is a large share of the service | nvImageCodec into a CUDA tensor in PreProcess, the algorithm takes it, the video capability in the Dockerfile, other inputs fail fast (card `gpu-jpeg-decode`) | -6 to -16 ms |
| Preprocessing on the GPU | CPU warp, letterbox, normalize | one Triton kernel bit-exact with the CPU reference (`gpu-inference-dataflow`) | 11.35 to 1.93 ms with the decode, L1 |
| FP16 engine | the engine dominates in FP32 | pure FP16, strongly typed; needs approval (card `fp16-trt-engine`) | -62% |
| CUDA Graph | enqueue gaps | capture `execute_async_v3` per shapes and addresses (card `cuda-graph-trt`) | -6.6%, -9.1% |
| Readback in the graph | a D2H and sync per call | capture the D2H into pinned memory, where pinned memory is allowed | -2.6% |
| Several items per request | one engine call per item | one profile per batch size, graphs captured at Init (card `dynamic-batch-profiles`) | P99 8.95 to 5.46 ms |
| Plugins | GroupNorm, deformable attention, V2 built-ins | `gpu-inference-trt-plugins` | -14.8%, -4.5%; MPS-safe |
| Ship the engine | built at start or in CI | prebuilt per compute capability, by LFS, loaded alone (card `ship-prebuilt-engine`) | cold start about halved, -450 MiB |
| Cleanup | files over 200 lines, dead code | below | L0 |

Small service changes beside them: `.dockerignore` keeps weights, ONNX, and offline tools out of the image, never the engine (1.22 to 0.30 GB); the framework pin moves to its newest commit when the pinned one lacks the runtime's Python.

## Decisions that differed
Ask each; never carry one project's answer into another.
- Pinned memory: allowed in one project, ruled out in another.
- Several items per request: pipelining items over slots was landed, then removed by the user (production's GPU also runs other work); a dynamic batch replaced it.
- Precision past FP16: FP8 and BF16 were measured, never landed.

## Cleanup before delivery
The user's wish: code a person can read, each item its own experiment, verify L0.
- Split our own files over 200 lines by purpose, moving code only; delete dead code, branches that never run, unused files, and IDE config. Leave vendored code as it is and requirements untouched.
- Init fails fast: a missing engine or library, an LFS pointer, another GPU, or another TensorRT raises naming the cause and the fix (`git lfs pull`, the build script).
- Each changed repo's readme says what ships, how to rebuild it, and what it binds to.

## Delivery order
`rules/wrap-up.md` step 1 has the rule; in these projects it ran as:
1. Algorithm repo: one commit per kept experiment on the dev branch in the user's identity, engines and libraries by LFS; then the dev branch merged into the branch it was created from.
2. Service repo: the dependency file's algorithm line pins that merge commit; the service's own changes as their own commits (api.py, Dockerfile, CI variables).
3. The user pushes the algorithm repo with its LFS objects first, then the service. Before handing over, `git lfs ls-files` lists every engine and library, and nothing tracked is an LFS pointer.
Then the checks of `gpu-inference-delivery`.

## Pitfalls seen
- TensorRT 10.13: with two CumSum layers in one network, one returned the other's result (fixed in 10.14). A cumsum positional encoding lost its vertical axis.
- TensorRT 10.16 on sm_120 accumulates FP16 matrix kernels in FP16 whatever the constraints; read kernel names with the engine inspector before promising a precision mix.
- The build image may lack the video capability and production's GPU: never run the service's Init in CI.
- A container killed mid-build left partial files that the next start took as a cache: build to a temporary name, then rename.
- NVJPG runs one job at a time: a large decode submitted first delays the small ones behind it.
- The framework's worker timeout kills a worker with GPU work outstanding; under MPS that can put the MPS server into teardown (error 807 for every new worker) until the container restarts.
- A production GPU learned late (sm_86 instead of sm_120) invalidated every shipped plan and library: ask in onboarding, again before delivery.
