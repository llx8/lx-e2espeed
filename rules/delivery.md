---
tier: on-demand
read_when: wrap-up step 1, for the handoff checks the deployment calls for, and step 2, for the facts to state; or when the user asks whether something is ready to deliver
---
# Delivering an optimized inference service

Which checks apply depends on how this project is deployed; the answers belong in `lx-work/PROJECT.md` from onboarding. Ask the user for anything the repository cannot tell, such as a remote weight store path, and mark the delivery blocked until you have it. Never assume a default mount point.
Run the checks after code, accuracy, performance, and maximum-input validation pass; any failed check blocks the handoff.

## Always
1. Fail fast. The selected production backend either completes or stops with an actionable error. Show that an initialization or execution failure terminates instead of silently running a slower path (PyTorch, CPU, PIL/OpenCV, a smaller model). Optional-dependency detection belongs in setup or diagnostics, never per request.
2. Reference path kept, not used as fallback. The eager or reference implementation stays available as an explicit offline configuration for comparison and rollback.
3. Self-contained. Delivered code imports nothing from `lx-work/`, `lx-e2espeed/`, sibling repositories, `/tmp`, or build directories; section 3 has the scans.
4. The build path contains the change. An optimization that exists only in a local working tree, a symlinked copy, or an uncommitted export patch is not delivered. Check dependency locks, export scripts, and CI steps that rebuild engines, and name what must be committed or re-locked.
5. Engines are not built in the request path. Check that engine, plugin, dtype, shape, and path mismatches fail fast at startup.
6. Production defaults stay. What production's image sets up by default (the MPS daemon, driver capabilities) is in the delivered image; a gap there blocks the handoff, never a backlog note.

## When the deployment calls for it
- Weight path contract: when production loads the algorithm repo as a subdirectory (for example from a service wrapper), every weight is referenced relative to the repo root and joined in code; test by starting the entry point from another directory. Section 1.
- Remote weight store: when weights are deployed to a store, inventory them with SHA-256, compare, sync what differs, re-verify, and scan for symlinks. Section 2.
- Shared runtime libraries: when the service loads TensorRT plugins or other `.so` files, they ship with the project and are recorded with GPU architecture, CUDA, TensorRT ABI, source revision, and SHA-256. Section 3.
- Multi-worker GPU sharing: when several workers share a GPU through MPS, scan the engines first (`playbook/serving/mps-grid-barrier.md`).
- Maximum input: when the service accepts very large inputs, the repeated run at the maximum input PROJECT.md names has passed.

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
- Whether `.plan` or `.engine` files are committed follows the project's artifact policy; they never replace required runtime libraries.
- Check the staged content: required source and libraries must be tracked, not only present on the development machine, in another repo, in a build directory, or as an unfetched Git LFS pointer.

## Facts to state at delivery
The baseline (what results compare with) and where it came from; environment, commands, inputs, warmup, and repetitions; end-to-end latency (cold start and warm P50/P90/P99) and peak memory in three sets: the baseline, before (the code the delivery starts from), and after (the delivered code); each affected stage's latency and share of the original total;
the complete accuracy command, input matrix, metrics, tolerances, and result, with production and the optimized version each against the reference (the fixed engine only attributes a change to the optimization or the environment move);
the pre/post tensor contract and which semantics were verified unchanged; files and runtime artifacts changed; assumptions, unsupported shapes, cold-start cost, and the rollback path; what was not measured or validated; the remaining bottleneck and the next evidence-backed candidate.
Never infer a speedup from code shape alone.
When the checks above ran, the summary also lists their results: the weight path inventory, the store path, pre- and post-sync differences with digest checks, the symlink scan, the cross-repo reference scan, the submodule, gitlink, nested `.git`, and runtime-download scans, the committed runtime libraries with their compatibility facts, and anything not finished or not reachable.
