#!/bin/sh
# Owner-run real-tool acceptance on an isolated copy of explicitly supplied private inputs.
#
# The plan, commands, working directories and limits are recorded in
#   ai_records/acceptance/20261013_claude_web_mvp_real_tool_acceptance.md (Part A)
# and were written before this script first ran. Revised in P3.1 (2026-10-14) after the Codex
# review found that the first version reported success when a step had failed.
#
# What it does, in order:
#   0. prints the host facts;
#   1. if SHELXT or SHELXL is an x86_64-only build, checks that this Mac can launch an x86_64
#      process at all, using a *system* binary (arch -x86_64 /usr/bin/true). No crystallographic
#      program is involved. If it cannot, it stops here: installing or repairing Rosetta 2 is the
#      owner's decision and is never done by this script;
#   2. copies the two raw input files into a fresh, ignored directory and checks their SHA-256
#      against the expected digests (stops on any difference);
#   3. prints the static tool status (crystalx doctor);
#   4. runs the preliminary solve through the same application service the Web interface uses:
#      SHELXT, a bounded isotropic SHELXL refinement, then PLATON -- each with a wall-clock timeout
#      and operating-system limits, in its own process group;
#   5. checks the run record with `crystalx verify-run`: every stage must have succeeded on real
#      programs whose digests still match, and every file the next step needs must exist. A run
#      that merely "completed" (for example with PLATON blocked) is NOT accepted.
#
# Exit status (also written to <acceptance dir>/status):
#   0  accepted (mechanically: every stage ran and left its files; not a judgement of the structure)
#   1  the run happened but is not accepted (a stage failed, was blocked or skipped, ...)
#   2  setup problem (CLI or Python missing, input files missing, cannot create the directory)
#   3  host translation unavailable: x86_64 SHELX cannot be launched on this Mac
#   4  input digest mismatch: the copies do not match the explicitly supplied or captured input digests
#   5  the solve produced no run record
#
# It never opens <expert answer RES>, never modifies the originals or the
# toolbox, never installs anything, and writes only under runs/ (ignored by git). Everything it
# prints is also saved to <acceptance dir>/transcript.txt.
#
# Usage, from anywhere:
#   CRYSTALX_ACCEPT_SOURCE_DIR=/path/to/originals CRYSTALX_ACCEPT_STEM=explicit_stem \
#     sh crystalx_web/scripts/acceptance_private_case.sh
#
# Source directory and stem are required. Optional expected SHA-256 variables let
# the caller independently pin input identity; otherwise originals are hashed just
# before copying. Other CRYSTALX_ACCEPT_* variables support isolated synthetic tests.

set -u

REPO=${CRYSTALX_ACCEPT_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
CLI=${CRYSTALX_ACCEPT_CLI:-$REPO/.venv/bin/crystalx}
PY=${CRYSTALX_ACCEPT_PYTHON:-$REPO/.venv/bin/python}
SRC=${CRYSTALX_ACCEPT_SOURCE_DIR:-}
STEM=${CRYSTALX_ACCEPT_STEM:-}
if [ -z "$SRC" ] || [ -z "$STEM" ]; then
    echo "error: CRYSTALX_ACCEPT_SOURCE_DIR and CRYSTALX_ACCEPT_STEM are required" >&2
    exit 2
fi
case "$STEM" in
    *[!A-Za-z0-9_-]*) echo "error: stem must contain only letters, digits, '_' or '-'" >&2; exit 2 ;;
esac
TOOLBOX=${CRYSTALX_ACCEPT_TOOLBOX:-$REPO/toolbox}
RUNS_ROOT=${CRYSTALX_ACCEPT_RUNS_ROOT:-$REPO/runs}
EXPECT_INS=${CRYSTALX_ACCEPT_EXPECT_INS_SHA256:-}
EXPECT_HKL=${CRYSTALX_ACCEPT_EXPECT_HKL_SHA256:-}

for tool in "$CLI" "$PY"; do
    if [ ! -x "$tool" ]; then
        echo "error: $tool not found. Run: sh crystalx_core/scripts/bootstrap_dev.sh" >&2
        exit 2
    fi
done
for file in "$SRC/$STEM.ins" "$SRC/$STEM.hkl"; do
    if [ ! -f "$file" ]; then
        echo "error: input file $file not found" >&2
        exit 2
    fi
done

# A fresh directory every time. mkdir without -p fails if the name exists, so two runs started in
# the same second never share a directory.
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUNS_ROOT" || exit 2
DIR="$RUNS_ROOT/acceptance_private_case_$STAMP"
n=1
while ! mkdir "$DIR" 2>/dev/null; do
    n=$((n + 1))
    if [ "$n" -gt 50 ]; then
        echo "error: cannot create a fresh acceptance directory under $RUNS_ROOT" >&2
        exit 2
    fi
    DIR="$RUNS_ROOT/acceptance_private_case_${STAMP}_$n"
done
mkdir "$DIR/input" || exit 2

sha256() {
    "$PY" -I -c 'import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
        h.update(block)
print(h.hexdigest())' "$1"
}

main() {
    echo "acceptance directory: $DIR"
    echo "started (UTC): $STAMP"
    echo
    echo "== 0. host"
    uname -a
    sw_vers 2>/dev/null || true
    echo "logical CPUs: $(sysctl -n hw.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
    echo
    echo "== 1. x86_64 launch check (system binary, not SHELX)"
    needs_x86=0
    for name in shelxt shelxl; do
        desc=$(file -b "$TOOLBOX/$name" 2>/dev/null || echo "unknown (file(1) failed)")
        echo "$name: $desc"
        case "$desc" in
            *arm64*) ;;
            *x86_64*) needs_x86=1 ;;
        esac
    done
    if [ "$needs_x86" -eq 1 ]; then
        arch -x86_64 /usr/bin/true
        launch=$?
        echo "x86_64 launch exit=$launch"
        ls -l /Library/Apple/usr/share/rosetta/rosetta /Library/Apple/usr/libexec/oah/libRosettaRuntime 2>&1
        pgrep -lx oahd 2>/dev/null || echo "oahd (Rosetta daemon) is not running"
        if [ "$launch" -ne 0 ]; then
            echo "RESULT: BLOCKED (host translation) -- this Mac cannot launch x86_64 programs, and"
            echo "SHELXT/SHELXL here are x86_64 builds. Nothing was started. Installing or repairing"
            echo "Rosetta 2 is the owner's decision; this script never does it."
            return 3
        fi
    else
        echo "no x86_64-only SHELX build detected: launch check not needed"
    fi
    echo
    echo "== 2. isolated copy of the raw input"
    if [ -z "$EXPECT_INS" ]; then EXPECT_INS=$(sha256 "$SRC/$STEM.ins") || return 2; fi
    if [ -z "$EXPECT_HKL" ]; then EXPECT_HKL=$(sha256 "$SRC/$STEM.hkl") || return 2; fi
    cp "$SRC/$STEM.ins" "$SRC/$STEM.hkl" "$DIR/input/" || return 2
    got_ins=$(sha256 "$DIR/input/$STEM.ins") || return 2
    got_hkl=$(sha256 "$DIR/input/$STEM.hkl") || return 2
    echo "ins $got_ins (expected $EXPECT_INS)"
    echo "hkl $got_hkl (expected $EXPECT_HKL)"
    if [ "$got_ins" != "$EXPECT_INS" ] || [ "$got_hkl" != "$EXPECT_HKL" ]; then
        echo "RESULT: FAIL (input digest mismatch) -- these are not the expected files; nothing was run."
        return 4
    fi
    echo
    echo "== 3. static tool status (header inspection; not a launch test)"
    if command -v xattr >/dev/null 2>&1; then
        xattr -l "$TOOLBOX/shelxt" "$TOOLBOX/shelxl" "$TOOLBOX/platon_special" 2>&1 | head -20
    fi
    "$CLI" --version
    "$CLI" doctor --toolbox "$TOOLBOX"
    echo
    echo "== 4. preliminary solve (same service as the Web interface)"
    "$CLI" solve \
        --ins "$DIR/input/$STEM.ins" \
        --hkl "$DIR/input/$STEM.hkl" \
        --toolbox "$TOOLBOX" \
        --runs-dir "$DIR/runs"
    solve_status=$?
    echo "solve exit=$solve_status"
    echo
    echo "== 5. acceptance check of the run record (reads the record only; never the answer)"
    run_dir=""
    count=0
    for candidate in "$DIR/runs"/r-*; do
        if [ -d "$candidate" ]; then
            run_dir=$candidate
            count=$((count + 1))
        fi
    done
    if [ "$count" -ne 1 ]; then
        echo "RESULT: FAIL -- expected exactly one run record under $DIR/runs, found $count"
        return 5
    fi
    echo "run record: $run_dir"
    "$CLI" verify-run "$run_dir" --expect-input "ins=$EXPECT_INS" --expect-input "hkl=$EXPECT_HKL"
    verify_status=$?
    "$CLI" verify-run "$run_dir" --json --expect-input "ins=$EXPECT_INS" --expect-input "hkl=$EXPECT_HKL" \
        > "$DIR/acceptance.json"
    echo "verify-run exit=$verify_status (verdict: $DIR/acceptance.json)"
    echo "finished (UTC): $(date -u +%Y%m%dT%H%M%SZ)"
    if [ "$solve_status" -eq 0 ] && [ "$verify_status" -eq 0 ]; then
        echo "RESULT: PASS -- every required stage ran on real programs and left its files."
        echo "This is a mechanical check. Scoring against the reference structure is a separate step."
        return 0
    fi
    echo "RESULT: FAIL -- solve exit=$solve_status, verify-run exit=$verify_status. Not accepted."
    return 1
}

# The pipe into tee would otherwise replace main's status with tee's: keep it in a file.
{ main; echo $? > "$DIR/status"; } 2>&1 | tee "$DIR/transcript.txt"
status=$(cat "$DIR/status" 2>/dev/null || echo 2)
echo
echo "Transcript: $DIR/transcript.txt"
echo "Exit status: $status"
exit "$status"
