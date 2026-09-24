# Maintained Dataflow Patterns

## GPU-resident stage chain

Decode or accept input once, convert to a CUDA tensor once, and keep masks, crops, model outputs, blends, and restoration inputs on device until the final required host output.

## Data-dependent skip

Compute cheap metadata such as mask occupancy before allocating full-size intermediates or invoking a restore model. Cover empty, tiny, and normal regions in tests.

## Transform fusion by reuse

Reuse an already resized or normalized tensor when consumers require identical semantics. Do not fuse operations whose interpolation or rounding differs.

## Narrow custom operator exception

Consider a custom operator only after profiling a stable, repeated image primitive. Compare against the mature library implementation, include edge sizes and layouts, and retain the custom path only with end-to-end value.
