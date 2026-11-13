#!/usr/bin/env bash
# Build the ALchemist desktop app (.dmg and .zip) on macOS (Apple Silicon). Unsigned (ad-hoc).
#
# Usage, from anywhere:
#   PYTHON=/path/to/python packaging/electron/build_macos.sh [--skip-frontend] [--skip-server]
#
# PYTHON must have ALchemist's dependencies plus pyinstaller. Needs Node.js.
# Output: packaging/electron/dist/
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SKIP_FRONTEND=0
SKIP_SERVER=0
for arg in "$@"; do
  case "$arg" in
    --skip-frontend) SKIP_FRONTEND=1 ;;
    --skip-server) SKIP_SERVER=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ $SKIP_FRONTEND == 0 ]]; then
  (cd "$ROOT/alchemist-web" && npm ci --no-audit --no-fund && npm run build)
fi
if [[ $SKIP_SERVER == 0 ]]; then
  "$ROOT/packaging/desktop/build_macos.sh" --skip-frontend
fi

cd "$HERE"
npm ci --no-audit --no-fund
node scripts/set-version.mjs
npx electron-builder --mac --publish never
ls -lh dist/*.dmg dist/*.zip
