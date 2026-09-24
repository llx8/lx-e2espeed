# Maintained Memory Patterns

## Size-aware routing

Use the fastest tensor-native path below a measured pixel or shape limit and a bounded-memory tiled path above it. Test immediately below, at, and above the threshold.

## Metadata-first tiling

Store coordinates and overlap metadata, slice the source only when processing a tile, and write or blend the result before advancing.

Use this primarily for large-image preprocessing and postprocessing: masks, morphology, filtering, color operations, restoration, blending, and paste-back. Avoid constructing a list of cropped image tensors.

## Ordinary versus overlap tiles

Use ordinary non-overlapping tiles for pointwise or strictly local operations whose output does not depend on neighboring pixels. Use overlap tiles when a receptive field, resize/filter support, convolution, morphology, restoration, or blending crosses tile boundaries. Crop the valid center or combine overlaps with weights whose sum is one.

## Static tile with pad/crop

Pad edge tiles to the engine's supported shape, run the same engine, crop to the valid extent, and blend using behavior-matched weights.

## Diffusers VAE tiling plus CNN TensorRT

For large VAE encode/decode, preserve Diffusers tile size, traversal, latent scaling, overlap, blend, and true edge-tile shapes. Replace only the encoder/decoder execution with validated TensorRT engines when profiling shows worthwhile end-to-end value. Keep batch static at B1; use bounded dynamic H/W when static pad/crop changes valid pixels. Test encoder and decoder boundary tiles independently before the complete pipeline, then validate the downstream diffusion result because small VAE errors can be amplified.

## Shared non-concurrent workspace

Let TensorRT execution contexts that never overlap reuse one device-memory pool. Document the concurrency assumption and fail if concurrent execution is later introduced.

## Empty-region skip

Skip a tile before model execution when its effective mask is empty. Preserve any output initialization and blending semantics expected by the original path.
