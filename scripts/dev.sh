#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

cleanup() {
  echo "Shutting down..."
  if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting backend (FastAPI) on http://localhost:8000 ..."
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Starting frontend (Vite) on http://0.0.0.0:5000 ..."
cd app/frontend
npm run dev -- --host 0.0.0.0 --port 5000 &
FRONTEND_PID=$!

wait -n "$BACKEND_PID" "$FRONTEND_PID"
