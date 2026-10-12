---
tier: on-demand
read_when: updating or questioning the skills in this directory
---
# Upstream

Copied unchanged from https://github.com/NVIDIA/TensorRT-LLM, path .claude/skills/, commit 261c5cc9c54814622392d176de534713d2711f53 (checked 2026-09-24). Compared again with commit 4f442e50a598ab62836c9fc5a2cc38e49cbc9b5e on 2026-10-12: the five kept skills were unchanged upstream. License: Apache-2.0, see LICENSE.

Kept: perf-nsight-systems, perf-nsight-compute-analysis, perf-torch-cuda-graphs, perf-torch-sync-free, perf-workload-profiling. They are generic PyTorch and CUDA guidance.
Left out: perf-analysis and perf-optimization (autopilot loops routed to specialist agents, against the collaborative workflow), perf-host-analysis, perf-host-optimization, and perf-optimization-casebook (specific to TensorRT-LLM). Added upstream by 2026-10-12 and not evaluated: exec-* (cluster and container execution), kernel-cute-writing, kernel-tileir-optimization, kernel-triton-writing, and trtllm-* (TensorRT-LLM development); kernel-triton-writing is the one worth a look when a project writes Triton kernels.

setup.sh global installs them to ~/.claude/skills/. They cannot load from here: this directory is outside every project, and Claude Code did not load skills even from a git-ignored copy inside one (tested 2026-09-24 with Claude Code 2.1.281).

The only local change: each SKILL.md description is shortened to one line, because every session loads the descriptions. Reapply that after an update.

Do not edit these files otherwise; the rules in CLAUDE.md win where they differ. Update only with the user's approval, by copying a newer upstream commit and changing the commit above.
