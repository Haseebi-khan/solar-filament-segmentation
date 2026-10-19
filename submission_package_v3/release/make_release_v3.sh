#!/usr/bin/env bash
# Packs the M5 weights (DINOv3 ConvNeXt-S, 5 folds, ~1.2 GB) with the DINOv3 licence into one zip for a GitHub
# Release. GitHub refuses files over 100 MB in git, so these weights cannot be committed; the other two members
# of the final submission (r34_2048, effv2s_1536, < 100 MB per fold) are already in the repository.
#   submission_package_v3/release/make_release_v3.sh
# Output: release_v3/m5_cnxs_1536_v3_weights.zip (*.zip is git-ignored). Upload it as a release asset (TASKS.md).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
M=$ROOT/work/models/m5_cnxs_1536_v3
OUT=$ROOT/release_v3
for k in 0 1 2 3 4; do
  [ -f "$M/fold$k.pt" ] || { echo "missing $M/fold$k.pt"; exit 1; }
done
mkdir -p "$OUT"
cp "$ROOT/submission_package_v3/release/LICENSE_DINOv3.md" "$M/LICENSE_DINOv3.md"
( cd "$ROOT/work/models" && rm -f "$OUT/m5_cnxs_1536_v3_weights.zip" \
  && "$ROOT/.venv/bin/python" - "$OUT/m5_cnxs_1536_v3_weights.zip" <<'EOF'
import glob, sys, zipfile
files = sorted(glob.glob("m5_cnxs_1536_v3/fold?.pt") + glob.glob("m5_cnxs_1536_v3/fold?.log")) \
    + ["m5_cnxs_1536_v3/LICENSE_DINOv3.md"]
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_STORED, allowZip64=True) as z:   # weights do not compress
    for f in files:
        z.write(f)
print(f"zipped {len(files)} files")
EOF
)
rm -f "$M/LICENSE_DINOv3.md"
( cd "$OUT" && sha256sum m5_cnxs_1536_v3_weights.zip > m5_cnxs_1536_v3_weights.zip.sha256 )
ls -lh "$OUT"
cat "$OUT/m5_cnxs_1536_v3_weights.zip.sha256"
echo "restore: cd work/models && python -m zipfile -e <zip> .  (gives work/models/m5_cnxs_1536_v3/fold*.pt)"
