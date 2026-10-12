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

## Hardware decode in a service (NVJPG through nvImageCodec, 2026-09 to 2026-10)
- The container needs the video driver capability (`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`); without it the hardware decoder cannot be created (`Could not create decoder. Code: 7`). Create it at Init with a small probe image and fail there.
- Send NVJPG only complete baseline JPEGs: SOI (`FF D8`) first and EOI (`FF D9`) last. It does not take progressive (`FF C2`) or truncated files, and OpenCV versions disagree on truncated ones (one returned None, another half an image). A file that ends in EOI but is corrupt inside cannot be told apart before decoding; its damaged area fills differently from the CPU decoder.
- Inputs NVJPG does not take fail fast with an error response; never route them to the original decoder (the user, 2026-10-12). A GPU error fails the same way.
- Match the CPU decoder the service replaces: chroma upsampling mode and EXIF orientation. NVJPG then differs by about one level from libjpeg; one project accepted that as L1.
- Tell progressive from baseline by walking the segments to the frame header (8-bit SOF0 or SOF1 go to NVJPG), not by searching the bytes for `FF C2`: EXIF thumbnails and tables can contain it, and 12-bit or lossless frames have other markers.
- NVJPG runs one job at a time: when several images are decoded per request, submit the small ones first and the large one last on its own stream.
- A declined input makes nvImageCodec print a warning to stderr; expect it in service logs.
- GPU JPEG encode (nvJPEG) does not reproduce libjpeg-turbo's bytes: identical pixels before encoding gave responses up to 6 to 15 levels apart. Keep CPU encode where responses must stay byte-equal. With nvImageCodec, pass `quality_type=QualityType.QUALITY` with `quality_value`, or the value is ignored (worse images, 40% fewer bytes).
