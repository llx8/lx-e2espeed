---
tier: on-demand
read_when: profiling shows the symptom below
---
# TensorRT builder optimization level

- Category: env
- Applies to: TensorRT 10 engines built by project or experiment scripts, dynamic shapes and plugins included

## Symptom
An engine built by an experiment or prototype script runs slower than the production engine of the same network, and its plan lacks fusions or tactics the production plan has.

## Technique
Check `builder_optimization_level` in the build script before comparing anything: a prototype often lowers it (0 to 2) for fast rebuilds. Build the engine that is measured and shipped at level 3 (TensorRT's default) and say the level in the engine's record; keep the low level for iteration only.

## Effect
398 -> 344 ms (-14%) on a dynamic-shape FP16 engine, level 0 to 3, matching the production engine of the time (343 ms).

## Side effects
Build time grows with the level. Levels 4 and 5: not measured.

## Does not apply
- Scripts that never set the level: TensorRT already builds at 3.

## Evidence
- ✓ 2026-09: two-network image color-transfer model at 1536x1536, dynamic-shape FP16 engine with a D=16 attention plugin, RTX 5090, TensorRT 10: level 0 398 ms, level 3 344 ms, production engine 343 ms.
