#!/bin/sh
# DEPRECATED diagnostic harness. Prefer the shared `crystalx solve` service.
#
# Earlier versions rewrote symmetry and inserted a private chemical prior. Those
# transformations have been removed. This retained opt-in harness copies explicitly
# supplied INS/HKL files byte-for-byte; it provides no composition or case defaults.
# Originals and toolbox files are never modified, and nothing is downloaded.
#
# Usage (all three arguments required):
#   CRYSTALX_ALLOW_DEPRECATED_HARNESS=1 sh crystalx_core/scripts/first_live_shelxt.sh \
#     /path/to/input.ins /path/to/input.hkl explicit_stem
#
set -eu

if [ "${CRYSTALX_ALLOW_DEPRECATED_HARNESS:-}" != "1" ]; then
    echo "first_live_shelxt.sh is a deprecated direct-tool diagnostic harness." >&2
    echo "Use 'crystalx solve' or 'crystalx-web' instead. To run it anyway, set" >&2
    echo "CRYSTALX_ALLOW_DEPRECATED_HARNESS=1." >&2
    exit 2
fi

REPO=$(cd "$(dirname "$0")/../.." && pwd)
CORE="$REPO/crystalx_core"
TOOLBOX="${CRYSTALX_TOOLBOX:-$REPO/toolbox}"
if [ "$#" -ne 3 ]; then
    echo "usage: $0 /path/to/input.ins /path/to/input.hkl explicit_stem" >&2
    exit 2
fi
INS=$1
HKL=$2
STEM=$3
case "$STEM" in
    ""|*[!A-Za-z0-9_-]*) echo "error: stem must contain only letters, digits, '_' or '-'" >&2; exit 2 ;;
esac
for input in "$INS" "$HKL"; do
    if [ ! -f "$input" ]; then echo "error: explicit input not found: $input" >&2; exit 2; fi
done

WORK="$REPO/runs/live_shelxt_$(date -u +%Y%m%dT%H%M%SZ)"
PY="${PYTHON:-python3}"

export PYTHONPATH="$CORE/src"
export PYTHONDONTWRITEBYTECODE=1

echo "=============================================================================="
echo " 0. environment"
echo "=============================================================================="
echo "  python  : $($PY -V 2>&1)"
echo "  core    : $($PY -m crystalx_core --version)"
echo "  toolbox : $TOOLBOX"
echo "  case    : $INS"
echo "  scratch : $WORK"
echo

echo "=============================================================================="
echo " 1. tool resolution -- is your shelxt usable on this machine?"
echo "=============================================================================="
$PY -m crystalx_core doctor --toolbox "$TOOLBOX" || true
echo

echo "=============================================================================="
echo " 2. diagnosis (read-only; no external program runs here)"
echo "=============================================================================="
$PY -m crystalx_core diagnose --ins "$INS" --hkl "$HKL"
echo

echo "=============================================================================="
echo " 3. plan -- using only the explicitly supplied declarations"
echo "=============================================================================="
set +e
$PY -m crystalx_core plan --ins "$INS" --hkl "$HKL"
PLAN_STATUS=$?
set -e
echo "  plan exit status: $PLAN_STATUS"
echo

echo "=============================================================================="
echo " 4. stage a scratch copy -- originals are never touched"
echo "=============================================================================="
mkdir -p "$WORK"
cp "$HKL" "$WORK/$STEM.hkl"
cp "$INS" "$WORK/$STEM.ins"
echo "  copied supplied instructions without chemistry or symmetry edits"
echo

echo "=============================================================================="
echo " 5. YOUR shelxt, on YOUR data. This is the first real invocation."
echo "=============================================================================="
echo "  \$ $TOOLBOX/shelxt $STEM"
echo "  (runtime depends on the explicitly supplied data)"
echo
cd "$WORK"
set +e
"$TOOLBOX/shelxt" "$STEM"
SHELXT_STATUS=$?
set -e
echo
echo "  shelxt exit status: $SHELXT_STATUS"
echo "  files it produced:"
ls -la "$WORK" | sed 's/^/    /'
echo

echo "=============================================================================="
echo " 6. parse the result back through crystalx_core"
echo "=============================================================================="
$PY - "$WORK" "$STEM" <<'PYEOF'
import sys
from pathlib import Path

from crystalx_core.io.shelx_res import read_res

work, stem = Path(sys.argv[1]), sys.argv[2]
solutions = sorted(work.glob(f"{stem}_[a-z].res"))
if not solutions:
    print("  no <stem>_?.res was produced -- SHELXT did not reach a solution.")
    print("  That is a reportable result, not a crash. Keep the .lxt listing.")
    raise SystemExit(0)

for path in solutions:
    res = read_res(path)
    print(f"  {path.name}")
    print(f"      atoms          : {res.n_atoms}")
    print(f"      SFAC           : {' '.join(res.header.sfac)}")
    print(f"      LATT / SYMM    : {res.header.latt} / {len(res.header.symm)} operators")
    symmetry = res.header.symmetry()
    print(f"      chirality      : {symmetry.compatibility.value}")
    print(f"                       {symmetry.explain()}")
    print(f"      strongest Q    : {[round(q.height, 1) for q in res.strongest_q_peaks(5)]}")
    disagreements = res.label_element_disagreements()
    if disagreements:
        print(f"      label/SFAC disagreements: {disagreements[:5]}")
PYEOF

echo
echo "=============================================================================="
echo " done. Everything above is in: $WORK"
echo " Nothing in realdata_examples/ or toolbox/ was modified."
echo "=============================================================================="
