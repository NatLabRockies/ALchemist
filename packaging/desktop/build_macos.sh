#!/usr/bin/env bash
# Build the unsigned ALchemist.app and a .dmg on macOS (Apple Silicon).
#
# Usage, from anywhere:
#   PYTHON=/path/to/python packaging/desktop/build_macos.sh [--skip-frontend]
#
# PYTHON must have ALchemist's dependencies plus pyinstaller and pystray.
# Output: packaging/desktop/dist/ALchemist.app and ALchemist-<version>-macos-arm64.dmg
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

VERSION="$("$PYTHON" -c 'import alchemist_core; print(alchemist_core.__version__)')"
ARCH="$(uname -m)"
DMG="$HERE/dist/ALchemist-$VERSION-macos-$ARCH.dmg"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$HERE/dist/ALchemist.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "ALchemist" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null

echo "App: $HERE/dist/ALchemist.app ($(du -sh "$HERE/dist/ALchemist.app" | cut -f1))"
echo "DMG: $DMG ($(du -h "$DMG" | cut -f1))"
