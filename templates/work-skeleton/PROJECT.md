---
tier: on-demand
read_when: deciding scope, metric, accuracy, contracts, deployment, or how to run things
---
# Project

Facts agreed with the user during onboarding; what the code and the user showed behind them is in knowledge/. Change them only with the user, and record why in DECISIONS.md.

## Purpose
<One or two lines: what the model does in this project.>

## End-to-end scope
- Measurement boundary: <what the timed path includes and excludes, for example decoded input to final output without file I/O>

## Workload
<Input shapes, batch sizes, request pattern, warm or cold start, data-dependent cases.>
- Main scenario: <the scenario headline numbers and exp/INDEX.md use>
- Maximum input: <largest input the service accepts, or none>

## Metric
<Latency by default. The exact measurement, the target, and the baseline every result is compared with: where its speed comes from (the baseline commit's own bench by default, or the source the user chose, DECISIONS.md), its numbers, and its files in baseline/numbers/.>
- Bench time budget: <how long one bench run may take; WARMUP and REPEAT in adapter/pipeline.py fit it>

## Accuracy standard
<Source of the reference outputs, the baseline's outputs (the baseline commit's own by default), comparison method, and inputs covered.>
- Metrics and pass criterion: <for images, for example max error and PSNR thresholds>
- Difference levels: <L0 identical; L1 up to ...; L2 up to ...; L3 beyond; or none, when the user judges every result that is not L0>
- Accepted level: <the highest level the user accepts, or none>

## Contracts
- Interface: <what downstream code relies on: argument order, return structure, channel order, sizes>
- Tensor contract: <pre/post dtype, range, layout, color order, interpolation, rounding>
- Consumers: <who uses the output, and how they pin this code>

## Deployment
- Production environment: <image or environment, versions>
- Code and weight loading: <how production loads the repo and resolves weights>
- Workers per GPU: <count, and whether MPS is used>
- Delivery includes: <CI steps, dependency locks, runtime libraries, engines>

## Done when
<The target, a point of diminishing returns, or a deadline.>

## Constraints
<Code that must not change, allowed dependencies, techniques the user ruled out.>
- Private terms: <project, company, product, and model names, comma-separated; the checker keeps them out of lx-e2espeed>

## Environment
- Target hardware: <where the model runs in production; note if it differs from this machine>
- Python environment: <name and absolute interpreter path>
- Versions: <driver, CUDA, framework, TensorRT>
- Profilers: <working nsys and ncu commands>
- Rules version at onboarding: lx-e2espeed <rules-version> (written by setup.sh; later pulls leave it as is, git log of lx-e2espeed has the rest)
- Baseline commit: <the project commit every experiment starts from; exp.sh rebuilds the tree from it. Not the baseline that results compare with (Metric)>

## Commands
<How to run the project, the environment check, the bench, and verify.>
