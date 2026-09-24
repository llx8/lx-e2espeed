---
tier: on-demand
read_when: updating or questioning the skills in this directory
---
# Upstream

Copied unchanged from https://github.com/NVIDIA/TensorRT-LLM, path .claude/skills/, commit 261c5cc9c54814622392d176de534713d2711f53 (checked 2026-09-24). License: Apache-2.0, see LICENSE.

Kept: perf-nsight-systems, perf-nsight-compute-analysis, perf-torch-cuda-graphs, perf-torch-sync-free, perf-workload-profiling. They are generic PyTorch and CUDA guidance.
Left out: perf-analysis and perf-optimization (autopilot loops routed to specialist agents, against the collaborative workflow), perf-host-analysis, perf-host-optimization, and perf-optimization-casebook (specific to TensorRT-LLM).

setup.sh global installs them to ~/.claude/skills/. They cannot load from here: Claude Code does not look for skills inside git-ignored directories such as <project>/lx-e2espeed/ (tested 2026-09-24 with Claude Code 2.1.281).

The only local change: each SKILL.md description is shortened to one line, because every session loads the descriptions. Reapply that after an update.

Do not edit these files otherwise; the rules in CLAUDE.md win where they differ. Update only with the user's approval, by copying a newer upstream commit and changing the commit above.
