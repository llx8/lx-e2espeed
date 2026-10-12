#!/usr/bin/env bash
# Run a tool command inside a production image instead of the local environment, for numbers that must come from
# production's versions: the baseline speed in the previous image, a check on another TensorRT.
# Usage, from the project directory, under a small cap (it covers this script and the docker client only):
#   systemd-run --scope --quiet -p MemoryMax=2G -p MemoryLimit=2G lx-work/tools/run-image/main.sh <env> <command>
# <env> is a key of IMAGES below. The container has its own memory cap (LX_MEM, default 24g; size it from the
# job's peak) and is killed when MemAvailable falls under 20% of RAM; it is removed after the run, so the image
# never changes. The project is mounted read-only at its own path, lx-work/local writable, and the target of each
# symlink at the project top (a linked repo) read-only at its own path. With ENGINES set, lx-work/local/engines/<env>
# is mounted over that project directory, so each image keeps its own built engines (LX_ENGINES=<name> mounts
# lx-work/local/engines/<name> instead, for builds that must not replace them). Wheels in lx-work/local/wheels/<env>/
# are installed with --no-deps first: custom ops built for that image, or another TensorRT's wheels.
set -euo pipefail
# TODO at onboarding: the images of PROJECT.md (Environment, Deployment), for example
#   [current]=registry.example.com/runtime:cuda13-trt10.16 [previous]=registry.example.com/runtime:cuda12.8-trt10.13
declare -A IMAGES=()
ENGINES=""   # TODO or empty: the project directory that holds built engines, for example engines
PY_PATH=""   # TODO or empty: the image's python directory to put first on PATH, for example /opt/venv/bin

ENV=${1:-}
if [ -z "$ENV" ] || [ -z "${IMAGES[$ENV]:-}" ]; then
    echo "usage: $0 <${!IMAGES[*]}> <command>" >&2
    exit 2
fi
shift
PROJECT=$(cd "$(dirname "$0")/../../.." && pwd)
LOCAL=$PROJECT/lx-work/local
NAME=lx-$ENV-$$
WHEELS=$LOCAL/wheels/$ENV
mkdir -p "$WHEELS"
MOUNTS=(-v "$PROJECT:$PROJECT:ro" -v "$LOCAL:$LOCAL" -v "$WHEELS:/lx-wheels:ro")
for link in "$PROJECT"/*; do
    if [ -L "$link" ]; then
        target=$(readlink -f "$link")
        MOUNTS+=(-v "$target:$target:ro")
    fi
done
if [ -n "$ENGINES" ]; then
    built=$LOCAL/engines/${LX_ENGINES:-$ENV}
    mkdir -p "$built"
    MOUNTS+=(-v "$built:$PROJECT/$ENGINES")
fi

trap 'docker kill "$NAME" >/dev/null 2>&1 || true' INT TERM
docker run --rm --name "$NAME" --gpus all --memory="${LX_MEM:-24g}" --memory-swap="${LX_MEM:-24g}" \
    "${MOUNTS[@]}" -w "$PROJECT" -e PYTHONDONTWRITEBYTECODE=1 -e LX_PY_PATH="$PY_PATH" --entrypoint bash \
    "${IMAGES[$ENV]}" -c \
    'if [ -n "$LX_PY_PATH" ]; then export PATH=$LX_PY_PATH:$PATH; fi
     if ls /lx-wheels/*.whl >/dev/null 2>&1; then pip install -q --no-deps --no-index /lx-wheels/*.whl >/dev/null || exit 1; fi
     exec "$@"' run-image "$@" &
PID=$!
TOTAL=$(awk '/MemTotal/{print $2}' /proc/meminfo)
while kill -0 $PID 2>/dev/null; do
    AVAIL=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ $((AVAIL * 100 / TOTAL)) -lt 20 ]; then
        echo "run-image: MemAvailable ${AVAIL} kB is under 20% of RAM; killing the container" >&2
        docker kill "$NAME" >/dev/null 2>&1 || true
    fi
    sleep 2
done
wait $PID
