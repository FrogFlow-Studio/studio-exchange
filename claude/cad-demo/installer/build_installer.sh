#!/usr/bin/env bash
# Сборка установщика CAD Intelligence для Windows на Linux.
#   bash build_installer.sh <рабочая_папка> [<папка_для_setup.exe>]
# Нужны: git, python3 с pip и Pillow, makensis (пакет nsis), curl, unzip.
# Python для Windows берётся из NuGet (python.org из облака бывает закрыт), библиотеки —
# готовые колёса win_amd64 с PyPI. Сканы студии и ключи в установщик не попадают:
# туда идут только файлы из git.
set -euo pipefail
WORK=$(realpath -m "${1:?рабочая папка}")
OUT=$(realpath -m "${2:-$WORK}")
HERE=$(cd "$(dirname "$0")" && pwd)
DEMO=$(dirname "$HERE")
REPO=$(git -C "$DEMO" rev-parse --show-toplevel)
PYVER=3.12.10
mkdir -p "$WORK" "$OUT"
STAGE="$WORK/stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/app"

echo "[1/5] Python $PYVER для Windows"
if [ ! -x "$WORK/py/tools/python.exe" ] && [ ! -f "$WORK/py/tools/python.exe" ]; then
  mkdir -p "$WORK/py"
  curl -sSL -m 300 -o "$WORK/python.nupkg" "https://api.nuget.org/v3-flatcontainer/python/$PYVER/python.$PYVER.nupkg"
  (cd "$WORK/py" && unzip -q -o "$WORK/python.nupkg")
fi
cp -r "$WORK/py/tools" "$STAGE/python"

echo "[2/5] Библиотеки (колёса для Windows)"
pip install -q --no-cache-dir --target "$STAGE/python/Lib/site-packages" --platform win_amd64 \
  --python-version 3.12 --implementation cp --abi cp312 --only-binary=:all: -r "$DEMO/requirements.txt"
find "$STAGE/python" -name __pycache__ -type d -prune -exec rm -rf {} +

echo "[3/5] Программа из git (без сканов студии и ключей)"
REV=$(git -C "$REPO" rev-parse HEAD)
git -C "$REPO" archive HEAD claude/cad-demo chatgpt-1/cad-intelligence/cad-acceptance-2026-11-12 | tar -x -C "$STAGE/app"
# изменения, ещё не попавшие в коммит, в установщик не идут: собирать с чистого дерева
if [ -n "$(git -C "$REPO" status --porcelain -- claude/cad-demo)" ]; then
  echo "Внимание: в claude/cad-demo есть незакоммиченные изменения, они в установщик не попали" >&2
fi
echo "$REV" > "$STAGE/app/claude/cad-demo/VERSION"
rm -rf "$STAGE/app/claude/cad-demo/installer" "$STAGE/app/claude/cad-demo/bench_chips"
# makensis на Linux падает (Segmentation fault) на файлах с русскими именами: в установщик идут только
# файлы с латинскими именами (это документы, программе они не нужны)
find "$STAGE/app" -depth -name '*[! -~]*' -print -exec rm -rf {} +

echo "[4/5] Картинки"
python3 "$HERE/make_art.py" "$STAGE/art"

echo "[5/5] Сборка setup.exe"
VERSION="$(date +%Y.%m.%d)-${REV:0:7}"
makensis -V2 -DVERSION="$VERSION" -DSTAGE="$STAGE" -DOUTFILE="$OUT/CAD-Intelligence-Setup.exe" "$HERE/cad_intelligence.nsi"
ls -la "$OUT/CAD-Intelligence-Setup.exe"
sha256sum "$OUT/CAD-Intelligence-Setup.exe"
