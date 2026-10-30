#!/usr/bin/env bash
# Run from a pinned checkout on the authorized Linux host. Does not auto-start jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m scripts.h_gpu_task "$@"
