---
tier: on-demand
read_when: profiling shows the symptom below
---
# Engine difference triage

- Category: compute
- Applies to: any built engine (TensorRT or similar) whose end-to-end output differs beyond the agreed level

## Symptom
Verify grades an engine candidate beyond the accepted level and the difference has no single obvious cause: a rebuilt, mixed-precision, plugin-rewritten, or tiled engine.

## Technique
Find the first failing boundary, on one saved input tensor and the same named output, in this order: eager FP32 against ONNX FP32 (export semantics); ONNX FP32 against the FP32 engine (lowering and tactics); the FP32 engine against FP16 or BF16 (precision); the original graph against the plugin-rewritten graph at one precision; untiled against tiled, with edge tiles and blend regions.
Then bisect inside it: expose the tensor at the topological midpoint of the suspect path as an extra ONNX output, compare it in both backends, move the bound, and repeat until one node or a small subgraph is left; a branch feeding a bad merge is bisected on its own.
The usual culprits by node: resize and sampling (coordinate transform, antialiasing, rounding); normalization (reduction axes, epsilon, accumulation dtype, layout); attention (scale folding, mask broadcast, softmax axis); convolution (layout, padding, tactic precision); plugin (serialization fields, format contract, workspace, stream); tiling (pad and crop extent, overlap weights, boundary shapes).
Pin only the isolated node or subgraph to higher precision: a global FP32 switch is a diagnostic, not the repair, and the repair's bench must show that the fix did not pick a slower tactic.

## Effect
Turns "the engine is wrong" into one node or subgraph, so the repair stays local and its bench shows whether it cost speed.

## Side effects
Marking an intermediate tensor as an output can block its fusion and change the tactics around it: compare at one precision, and bench an engine built without the extra outputs.

## Does not apply
- A difference inside the engine-noise spread (L1) is rebuild variance, not a defect: card `engine-build-variance`.

## Evidence
Method from the retired engine skill, absorbed into the rules on 2026-10-13 (lx-e2espeed 937dc71); the boundaries listed are the ones the FP16, plugin, and tiling projects of this playbook hit. No project has recorded a run of the whole procedure yet.
