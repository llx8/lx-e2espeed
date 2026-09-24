# Dynamic shapes

## Prefer a finite shape contract

For an online image service, choose in this order: fixed production shapes; a small explicit bucket set that includes the tile and maximum-input boundaries; bounded dynamic height/width only when buckets are impractical and recompilation is controlled.

Do not set `dynamic=True` just because image sizes vary. Dynamic kernels can lose optimizations, and Python values, strides, layouts, branches, or scalar extraction can still create guards and graph variants.

## Static and bucketed execution

With `dynamic=False`, warm up every supported bucket in a fixed order at startup, recording compile time, peak memory, and graph/cache identity for each. Reject unsupported shapes before the compiled function; never rescale silently or fall back to eager.

The bucket key must include every attribute that changes compiled behavior: device, dtype, rank, shape, stride/layout, optional inputs, Python booleans/enums, model or adapter identity, and relevant configuration.

## Bounded dynamic dimensions

When an approved experiment truly needs dynamic dimensions, mark only the dimensions proven to vary, before the first compiled call:

```python
torch._dynamo.mark_dynamic(sample, 2, min=minimum_h, max=maximum_h)
torch._dynamo.mark_dynamic(sample, 3, min=minimum_w, max=maximum_w)
output = compiled(sample)
```

`mark_dynamic` and `maybe_mark_dynamic` are version-sensitive. Check them against the project's pinned PyTorch version and keep them in one place, the project's compile integration layer. Without production evidence, do not make channel, batch, or latent dimensions dynamic.

Test VAE encode and decode separately for image and latent height/width, including divisibility constraints, pad/crop, overlap tiles, and maximum long-edge routing. Dynamic VAE compilation does not replace the memory skill's repeated maximum-input end-to-end validation.

## Guards and recompilation

Enable recompile logging and cover the full production shape distribution. Shape, stride, dtype, device, Python scalars, module/global state, or control flow can each trigger a new graph. Record the failing guard and fix the contract or move to buckets; never hide runaway graph variants by raising the cache limit.

Reset compiler state only between isolated experiments, never in the production request path. Clearing caches before every request measures compile cost, not steady-state service performance.
