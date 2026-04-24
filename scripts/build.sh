#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/app/frontend"

echo "Installing frontend dependencies..."
npm ci

echo "Building frontend..."
npm run build
