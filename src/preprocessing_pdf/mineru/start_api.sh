#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Safer defaults for macOS/Apple Silicon where MinerU PDF render subprocesses
# can be unstable under heavy parallel rendering.
export MINERU_PDF_RENDER_THREADS="${MINERU_PDF_RENDER_THREADS:-1}"
export MINERU_PDF_RENDER_TIMEOUT="${MINERU_PDF_RENDER_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY="${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}"

echo "[start_api] mineru-api: $(command -v mineru-api)"
echo "[start_api] python: $(command -v python || true)"
echo "[start_api] host=${HOST} port=${PORT} MINERU_PDF_RENDER_THREADS=${MINERU_PDF_RENDER_THREADS}"

exec mineru-api --host "$HOST" --port "$PORT"
