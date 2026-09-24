# Measurement Rules

- Do not count first-call CUDA initialization, engine build, or compile time as steady-state latency; report them separately.
- Do not place `torch.cuda.synchronize()` inside every production stage. Instrument only for a diagnostic run or use CUDA events.
- Profile the real shape and layout after preprocessing.
- Report data-dependent branches separately, including empty masks, no-face cases, and tile counts.
- Measure output encoding and host transfer when they are part of the user-visible request.
- Use Nsight Systems only when the PyTorch trace cannot explain CPU scheduling, stream gaps, or library launches. Use Nsight Compute only for a proven kernel hotspot.
