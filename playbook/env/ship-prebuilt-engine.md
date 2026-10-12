---
tier: on-demand
read_when: profiling shows the symptom below
---
# Ship the prebuilt engine with the repo

- Category: env
- Applies to: services that build or export their TensorRT engine at start or in CI

## Symptom
Cold start builds the engine (minutes, gigabytes of RAM) or imports the model code and weights only to load an engine; or CI builds the engine on a runner whose GPU is not production's.

## Technique
Build the engine offline once per production compute capability with a build script that refuses to overwrite, name it by compute capability (`<name>_sm120.plan`), and commit it with Git LFS in the repo that loads it, with its plugin library. Production loads only that file: no model code, weights, or build at start. A missing file, an LFS pointer, or a plan from another GPU or TensorRT fails at Init and names the fix. CI builds no engine; `.dockerignore` keeps weights and ONNX out of the image, never the engine.

## Effect
Cold start roughly halved and process memory -450 MiB; the image 1.22 -> 0.30 GB once weights and ONNX stay out; containers stop building at start (84 to 89 s and 8.6 GiB each).

## Side effects
One LFS object per engine and compute capability (250 MB for a large model); a rebuild is a manual step after a model, TensorRT, or GPU change.

## Does not apply
- Production GPUs not known: a plan from another compute capability does not load. Confirm them first; one project learned late that production was sm_86.
- Repos whose policy keeps engines out of git: build on site at start, with a file lock for several workers.

## Evidence
- ✓ 2026-09: image-to-guidance-map CNN service: engine exported in the build image and committed by LFS, CI build removed (its runner's GPU differed), responses unchanged.
- ✓ 2026-10: DETR-style pose model, FP16 plan of 250 MB, RTX 5090, TensorRT 10.16: cold start 3.875 -> 1.954 s, -448 MiB, outputs L0.
- ✓ 2026-10: the service around that model: cold start 4.54 -> 3.10 s, -446 MiB; image 1.22 -> 0.30 GB.
- ✓ 2026-10: face classification CNN service: sm_120 and sm_89 engines by LFS, CI build removed, L0.
