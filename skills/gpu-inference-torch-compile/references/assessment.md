# torch.compile admission

## Good candidates

- A measured eager PyTorch module dominates a repeated hot path.
- Inputs have fixed shapes, a small bucket set, or a validated dynamic-shape policy.
- CPU/GPU transfers and Python orchestration outside the module no longer dominate.
- Deployment can warm up ahead of traffic and can persist or deliberately rebuild compatible artifacts.
- The compile behavior of custom operators and control flow is known.
- The candidate holds enough adjacent eager operators for Inductor to fuse, or repeated kernel launch overhead is material.
- Weights and mutable module state are loaded before compilation and stay stable during requests.

## Reject or defer

- The module runs once per process, or cold start dominates latency.
- Shapes or Python branches cause runaway recompilation.
- A mature TensorRT or upstream diffusion backend already solves the hotspot more robustly.
- The measured stage is too small to matter end to end.
- Deployment cannot align PyTorch, Triton, CUDA, driver, and cache identity.
- The request path changes modules, adapters, hooks, dtypes, devices, or configuration.
- Python or custom-operator graph breaks split the main work and cannot be removed cleanly.
- `reduce-overhead` exceeds the maximum-input memory budget, or CUDA Graph tensor lifetimes conflict with the service.

## Candidate boundaries in image inference

Consider CNN blocks and restoration networks on fixed tile shapes; pure tensor pre/post chains already resident as CUDA FP16; the inner `encode`/`decode` of a VAE with an explicit height/width policy; a repeated diffusion transformer or UNet denoising body only when it runs native eager (not Nunchaku or TensorRT); and LUT, blending, mask, morphology, or geometry chains that analysis shows are dominated by many eager launches with no mature fused operator.

Never compile HTTP handlers, request schemas, image codecs, file access, model or weight loaders, scheduler construction, output serialization, or error and fallback policy.

## Experiment report

Report compile latency, first call, warm distribution, graph and variant counts, graph breaks, recompiles, cache behavior, generated-kernel evidence, peak CUDA allocated and reserved memory, maximum-input stability, full accuracy, concurrency, and end-to-end change. A microbenchmark of the compiled module alone is not enough.

## Versions

Confirm behavior against the project's pinned PyTorch version. These notes were written against PyTorch 2.13: [torch.compile API](https://docs.pytorch.org/docs/2.13/generated/torch.compile.html), [dynamic shapes](https://docs.pytorch.org/docs/2.13/torch.compiler_dynamic_shapes.html), [troubleshooting](https://docs.pytorch.org/docs/2.13/torch.compiler_troubleshooting.html), and [compile caching](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).

Constraints that still hold from earlier practice: compile only repeated hotspots, define the shape policy, warm up ahead of traffic, manage the disk cache, and treat VAE encode/decode as a narrow boundary. Earlier fixed gains and memory conclusions do not carry over.
