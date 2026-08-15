#!/usr/bin/env bash
# Assemble the Hugging Face Space (Docker SDK) from this repo and upload it.
# Usage: HF_TOKEN=... scripts/sync_space.sh [space_id]
set -euo pipefail
SPACE="${1:-Halleck45/OpenPronounce}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp -r "$ROOT/openpronounce" "$ROOT/templates" "$ROOT/static" "$TMP/"
cp "$ROOT/server.py" "$ROOT/pyproject.toml" "$ROOT/Dockerfile" "$ROOT/.dockerignore" "$TMP/"
cp "$ROOT/docs/space-README.md" "$TMP/README.md"
find "$TMP" -name __pycache__ -type d -exec rm -rf {} +

hf repo create "$SPACE" --repo-type space --space-sdk docker 2>/dev/null || true
hf upload "$SPACE" "$TMP" . --repo-type space --delete "*" --commit-message "Sync from GitHub"
echo "https://huggingface.co/spaces/$SPACE"
