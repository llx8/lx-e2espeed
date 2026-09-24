# Handoff checklist

Run after code, accuracy, performance, and maximum-input validation pass. Apply the sections the project's deployment contract calls for; any failed check blocks the handoff.

## 1. Weight path contract

For services that load the algorithm repo as a subdirectory, so the process working directory is not the repo:

- List every model, VAE, LoRA, TensorRT engine, and auxiliary weight actually read by code, configuration, and startup arguments.
- Each reference is a string fragment relative to the algorithm repo root, joined onto that root in code.
- Determine the repo root from the module's file location or the project's existing entry point, never from `cwd`.
- No development-machine absolute paths, paths into other repos, `../` escapes, or absolute paths into the remote weight store.
- Start the entry point from a directory outside the repo and confirm every path resolves; a missing file fails fast and prints its canonical relative path.

## 2. Remote weight store

The store path cannot be inferred from the repo. Ask the user and record the answer; without it, the handoff is blocked.

1. Inventory the weights the algorithm actually uses: canonical relative path, size, and SHA-256; for sharded weights, the index file and every shard.
2. Check each item under the same relative layout in the store.
3. Missing, different size, or different digest means inconsistent: sync the verified local file to the store.
4. After syncing, recompute the inventory and compare every item; matching names are not proof of matching content.
5. Scan the store for symlinks (`find <remote-weight-root> -type l -print`); the result must be empty. Replace any symlink with the real file and verify its digest again.
6. Never delete unrelated weights in the store without confirmation; sync only what this algorithm needs, and record what was added, replaced, or left alone.

## 3. Self-contained repo and runtime libraries

- Scan imports, `PYTHONPATH`, `sys.path`, configuration, shell scripts, `ctypes.CDLL`, plugin loading, and engine build code for paths into other repos.
- No submodules: no `.gitmodules`, and no mode `160000` entries in `git ls-files --stage`.
- No nested repos: no `.git` directory or file anywhere below the repo root. Copy external source without its repository metadata and track the real files.
- No `git clone`, `git submodule update`, or downloads of source or libraries during install, build, startup, or requests.
- Copied external source keeps its source URL, pinned revision, and license, with local imports.
- Shared libraries the service needs (TensorRT plugins, custom `.so`) are in the repo and committed, with GPU architecture, CUDA, TensorRT ABI, source revision, and SHA-256 recorded, and verified to load in a clean deployment environment.
- Whether `.plan`/`.engine` files are committed follows the project's artifact policy; they never replace required runtime libraries.
- Check the staged content: required source and libraries must be tracked, not only present on the development machine, in another repo, in a build directory, or as an unfetched Git LFS pointer.

## Record

The delivery summary lists: the weight path inventory, the store path, pre- and post-sync differences with digest checks, the symlink scan, the cross-repo reference scan, the submodule/gitlink/nested `.git`/runtime-download scans, the committed runtime libraries with compatibility facts, and anything not finished or not reachable.
