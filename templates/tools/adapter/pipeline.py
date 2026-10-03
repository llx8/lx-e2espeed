"""Project adapter: everything the generic tools in lx-work/tools/ need to know about this project.

Copy from lx-e2espeed/templates/tools/adapter/ at onboarding and fill in every TODO. Bench, verify, and
cpu-profile import only this module, so they measure the same path and run candidates the same way.
A candidate is a Python file in the candidate/ directory of an experiment or night trial that defines `Pipeline`.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

# Scenarios from PROJECT.md, main scenario first.
SCENARIOS = ("main",)  # TODO
WARMUP, REPEAT = 5, 30  # TODO: fit the bench time budget in PROJECT.md (Metric)
NOISE_MS = {}  # TODO: noise floor in ms per scenario, from the latest `bench --ab best`; STATUS.md shows it
BASELINE = ""  # TODO in onboarding step 5: glob, relative to lx-work, of the baseline's bench JSON files (PROJECT.md, Metric)

# Stage mode wraps these attributes of the pipeline object (dotted paths); the rest is "other".
STAGES = {}  # TODO, for example {"preprocess": ("prepare_inputs",), "model": ("model",)}

# verify: arrays that must be equal, arrays graded by their largest absolute difference, and PROJECT.md levels.
EXACT = ()  # TODO, for example ("boxes",)
GRADED = ()  # TODO, for example ("image",)
PEAK = 255.0  # value range of the graded arrays, for PSNR
LEVELS = ((0, "L0"), (1, "L1"), (4, "L2"))  # TODO: largest difference allowed per level; beyond is L3
ACCEPTED = "L0"  # TODO: from PROJECT.md


def criterion(metrics):
    """Extra pass criterion from PROJECT.md over {array: {max, mae, psnr, changed_pct, over1_pct}}."""
    return True  # TODO, for example all(m["psnr"] >= 50 for m in metrics.values())


def project_root(arg=None):
    """The project directory: the argument, or the directory that holds lx-work."""
    return Path(arg).resolve() if arg else Path(__file__).resolve().parents[2]


def load_pipeline(project, candidate=None):
    """A ready pipeline object: the project's own, or the candidate module's `Pipeline`."""
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    if candidate:
        spec = importlib.util.spec_from_file_location("candidate", candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Pipeline()
    raise NotImplementedError("TODO: import and build the project's pipeline")


def load_scenario(project, name):
    """Inputs of one scenario, read once; real production inputs, never downsampled."""
    raise NotImplementedError("TODO")


def prepare(inputs):
    """Fresh copies of the inputs for one call, made outside the timer, so in-place edits cannot leak."""
    raise NotImplementedError("TODO")


def call(pipeline, prepared):
    """One end-to-end call within the measurement boundary; return its output."""
    raise NotImplementedError("TODO")


def sync():
    """Wait for queued GPU work; a no-op when the pipeline synchronizes itself."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except ImportError:
        pass


def reset_peak_memory():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_memory():
    """Short peak-memory text for the summary; memory outside PyTorch comes from the bench's nvidia-smi."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return (f"peak alloc {torch.cuda.max_memory_allocated() >> 20} MiB "
                    f"reserved {torch.cuda.max_memory_reserved() >> 20} MiB")
    except ImportError:
        pass
    return ""


def to_arrays(output):
    """The output as named NumPy arrays for verify."""
    raise NotImplementedError("TODO")


def describe(output):
    """A few words about one output for the summary, for example the number of detected regions."""
    return ""


def versions():
    """Library versions for the result JSON."""
    found = {"numpy": np.__version__}
    for name in ("torch", "tensorrt", "cv2"):
        try:
            found[name] = __import__(name).__version__
        except ImportError:
            pass
    return found
