#!/usr/bin/env bash
# Download competition data (SkillCorner opendata, A-League 2024/25) + full body-pose files.
# Nothing downloaded here may ever be committed. Target dir: $SIGHTLINE_DATA_DIR (default ./data).
# Tip: keep the data dir OUTSIDE cloud-synced folders (OneDrive/Dropbox) - it is ~3.5 GB.
set -euo pipefail

DATA_DIR="${SIGHTLINE_DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"
OPENDATA_REPO="https://github.com/SkillCorner/opendata.git"
HF_BASE="https://huggingface.co/datasets/SkillCorner/opendata-bodypose/resolve/main"
mkdir -p "$DATA_DIR/pose"
echo "Data dir: $DATA_DIR"

command -v git-lfs >/dev/null || { echo "git-lfs is required (https://git-lfs.com)"; exit 1; }

# 1) Tracking, events, phases, match sheets (tracking .jsonl are LFS, ~100 MB each)
if [ ! -d "$DATA_DIR/opendata/.git" ]; then
  git clone --depth 1 "$OPENDATA_REPO" "$DATA_DIR/opendata"
else
  git -C "$DATA_DIR/opendata" pull --ff-only
fi
git -C "$DATA_DIR/opendata" lfs pull

# 2) Full body pose (~620 MB zipped each), verified against SkillCorner's MANIFEST.json
MANIFEST="$DATA_DIR/opendata/data/bodypose/MANIFEST.json"
for MID in 1925299 1996435; do
  OUT="$DATA_DIR/pose/$MID.jsonl.zip"
  EXPECTED=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['files']['raw/$MID.jsonl.zip']['sha256'])" "$MANIFEST")
  if [ -f "$OUT" ] && [ "$(sha256sum "$OUT" | cut -d' ' -f1)" = "$EXPECTED" ]; then
    echo "pose $MID: already present and verified"; continue
  fi
  echo "pose $MID: downloading"
  curl -L --fail --retry 3 -C - -o "$OUT" "$HF_BASE/raw/$MID.jsonl.zip"
  GOT=$(sha256sum "$OUT" | cut -d' ' -f1)
  [ "$GOT" = "$EXPECTED" ] || { echo "pose $MID: checksum mismatch ($GOT)"; exit 1; }
  echo "pose $MID: verified"
done
echo "Done."
