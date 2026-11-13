#!/usr/bin/env bash
# Build alchemist-server, the headless server the Electron app runs, on macOS (Apple Silicon).
#
# Usage, from anywhere:
#   PYTHON=/path/to/python packaging/desktop/build_macos.sh [--skip-frontend]
#
# PYTHON must have ALchemist's dependencies plus pyinstaller.
# Output: packaging/desktop/dist/alchemist-server/ (packaging/electron bundles it).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

if [[ "${1:-}" != "--skip-frontend" ]]; then
  (cd "$ROOT/alchemist-web" && npm ci --no-audit --no-fund && npm run build)
fi

cd "$ROOT"
"$PYTHON" -m PyInstaller "$HERE/alchemist.spec" --noconfirm \
  --distpath "$HERE/dist" --workpath "$HERE/build"
# Keep Spotlight out of the build folders.
touch "$HERE/dist/.metadata_never_index" "$HERE/build/.metadata_never_index"

echo "Server: $HERE/dist/alchemist-server ($(du -sh "$HERE/dist/alchemist-server" | cut -f1))"
