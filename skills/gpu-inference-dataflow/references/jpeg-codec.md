# CUDA JPEG Boundary

Use `scripts/ops/nvimgcodec_jpeg.py` when the service receives JPEG bytes, the downstream path remains on CUDA, and profiling shows CPU decode/encode or host-device transfer is material.

Keep one `NvJpegTorchCodec` instance resident. Decode JPEG bytes directly to a CUDA HWC uint8 tensor, retain the nvImageCodec owner, and use the active CUDA stream. Encode directly from a CUDA HWC tensor and synchronize only before consuming the returned host byte stream.

Validate:

- baseline versus progressive JPEG and supported chroma subsampling;
- grayscale and RGB inputs;
- EXIF orientation behavior;
- RGB/BGR channel order;
- full/limited color range and ICC/color metadata requirements;
- encode quality and chroma subsampling;
- odd dimensions, maximum dimensions, corrupt input, and unsupported channels;
- decode/encode latency including the required final synchronization;
- image quality against the service reference.

JPEG is lossy. Compare decoded pixels with tolerances and final image quality, not byte equality. Metadata or watermark insertion may require a CPU/container step after encoding; report that separately rather than claiming a fully GPU-only response path.
