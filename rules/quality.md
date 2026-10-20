---
tier: on-demand
read_when: setting up the references at onboarding or a pass, grading a change's outputs, judging an engine's precision, or reporting quality
---
# Output quality

How good outputs are is judged against the network itself, not against what production returns today: production can carry a defect of its own (one production FP16 engine was off by up to 370 px on its own scenarios).

## Two references
- Quality baseline: the network run by its framework in FP32 (the framework's defaults) in the target environment, inside the unmodified request path with only the inference call replaced, so decoding, preprocessing, and postprocessing stay the project's. Production's own run gives speed only.
- Identity reference: the unmodified code's own outputs (`verify --freeze`). A change that claims identical outputs is graded L0 to L3 against it (`rules/benchmarking.md`, Correctness).
- Both are frozen in baseline/ at onboarding or at the start of a pass, each with the command that rebuilds it.

## Randomness and the noise floor
- Find every random draw in the network (`torch.rand` in a forward, RandomUniformLike in its export) and seed it before each forward, with the code's own seed when it has one.
- Run the baseline with a few other seeds: their spread per scenario is the noise floor of quality. A gap inside it is no quality change; give it beside every quality number.
- An engine built from an export with random ops fixes its draws at build. When the question is precision alone, replace those ops by the baseline's draws (constants in the ONNX) on both sides.
- Check the export once: a TensorRT FP32 engine of the same ONNX lands inside the seed spread, or export or lowering is wrong (card `engine-difference-triage`).

## The metric
- Grade in the output's own units, as the caller uses it, chosen with the user. Transforms: the displacement of every original pixel between candidate and baseline (maximum and mean) and the canvas size difference. Images: the pixel metrics of `rules/benchmarking.md`. Classifiers: label flips and score shifts. Thresholds are the user's (PROJECT.md).
- Inputs: the scenarios; inputs the user picks from production-like data on this machine (candidates copied to `lx-work/local/inputs/`, nothing added without the user); and, for a change that moves outputs, a fixed random sample of about 100, reported as median, P90, maximum, and counts per band.
- To show a gap, apply the outputs (warp the image with each homography, overlay the masks) and put the largest cases side by side.

## Precision of an engine
A production precision whose distance to the quality baseline exceeds the seed spread is a finding: report it as a risk, whatever else the work is about. Find the smallest group of layers that closes it, as card `engine-difference-triage` says (Pinning). The repair changes outputs, so it is the user's.

## Reporting
Say what the metric measures in plain words first, then production, best, and each option against the quality baseline in one table, beside the seed spread.
