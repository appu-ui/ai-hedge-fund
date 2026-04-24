#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "Starting AI Hedge Fund (production) on http://0.0.0.0:5000 ..."
exec python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 5000
