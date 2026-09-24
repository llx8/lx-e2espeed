# Cache and deployment

## Cache identity

A compile cache is an executable deployment artifact, not a generic model file. Its identity binds at least: model weights and candidate source; PyTorch, Python, Triton, CUDA runtime, driver, GPU architecture, and related library or custom-operator versions; compile backend, mode, options, dynamic policy, dtype, layout, and shape buckets; image build and project version.

Resolve cache paths through the project's artifact or configuration mechanism; never hardcode absolute or repo-relative paths.

## Cold-start experiments

In separate fresh processes, measure:

1. Empty cache: compile and autotuning cost.
2. Compatible warm local cache: lookup and remaining startup cost.
3. Deliberately incompatible identity: the rejection or rebuild behavior.

Warm every production bucket before declaring the service ready. Unless the service contract accepts that latency, never declare ready and let the first user request compile an unknown graph.

## Local and portable caches

PyTorch keeps local Dynamo/Inductor/Triton caches, and newer versions may offer APIs to save and load compile artifacts. These are version-pinned features: confirm the target version supports them and reuse them only in compatible environments.

When using `TORCHINDUCTOR_CACHE_DIR`, set it through the project's deployment configuration before the compiled workload is imported or initialized. Give each immutable build a controlled writable or prefilled directory; incompatible versions must not share an unversioned mutable directory.

Portable or remote caches follow the same provenance, integrity, permission, and cleanup rules as engine artifacts. Never copy an arbitrary cache directory from a reference project or a development machine into production.

## Workers and concurrency

State whether each worker compiles independently, reuses a read-only prebuilt cache, or loads approved artifacts before start. Test lock contention during concurrent startup, and CUDA Graph and memory-pool behavior under steady concurrent requests. Multiply GPU and host memory budgets by the real worker count.

Production errors fail startup, readiness, or the request according to the service policy. Never recover a cache miss, compile failure, or compiled runtime error with eager in the same service process.
