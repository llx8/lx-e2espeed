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
Pinning, when the precision boundary fails: fix the random draws first, then pin layer groups to FP32 by name with obey-precision constraints, coarse to fine (module, stage, layer type such as normalization, attention, activation, then single layer), testing both directions (the group alone in FP32, the group alone in reduced precision). Measure why it fails (the input range of the suspect layers in the FP32 run) before naming a repair.
The repair pins the smallest group in the project's own build script, with a check that it matched the expected number of layers. Time it engine-only over several builds per side in interleaved rounds (tactics differ per build) and end to end; a solver iterating on the output (Levenberg-Marquardt, RANSAC) moves by whole iterations, so report its iteration counts. Rebuild a few times and rerun the random sample before calling it stable.

## Effect
Turns "the engine is wrong" into one node or subgraph, so the repair stays local and its bench shows whether it cost speed.

## Side effects
Marking an intermediate tensor as an output can block its fusion and change the tactics around it: compare at one precision, and bench an engine built without the extra outputs.

## Does not apply
- A difference inside the engine-noise spread (L1) is rebuild variance, not a defect: card `engine-build-variance`.

## Evidence
Method from the retired engine skill, absorbed into the rules on 2026-10-13 (lx-e2espeed 937dc71); the boundaries listed are the ones the FP16, plugin, and tiling projects of this playbook hit. The precision boundary has one full run, below; no project has run the export and tiling boundaries yet.
- ✓ 2026-10: segmentation transformer, FP16 engine on sm_120 only, NaN on 19 of 6000 inputs: traced to the attention normalization, its 16 layers pinned to FP32 on that GPU: NaN gone, L1 (as a rebuild), +4 MiB.
- ✓ 2026-10: camera-calibration CNN-transformer, production FP16 engine up to 370 px off the FP32 network on its own scenarios. Random NMF bases fixed at export: FP32 engine within 0.1 px of eager. Bisected by module, then each normalization: two backbone LayerNorms, exported as ReduceMean/Sub/Pow, square (x - mean) of up to 1475 past FP16's 65504, zeroing 8 to 83% of tokens.
  The four backbone LayerNorms (36 layers, matched by name, shape, constant, and cast layers skipped since a pinned cast fails the build) pinned to FP32 with obey-precision: 370 to 10 px, 100 random images median 25 to 3.9 px, inside the FP32 seed spread, engine +0.01 to +0.15 ms over three builds per side; end to end moved by whole solver iterations, one step fewer on L20 PNG (-5 ms).
