#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

export CUDA_VISIBLE_DEVICES
exec mineru-router --host "$HOST" --port "$PORT" --local-gpus auto
