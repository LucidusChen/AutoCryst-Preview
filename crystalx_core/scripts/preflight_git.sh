#!/bin/sh
# Inspect the actual index, including a caller-supplied GIT_INDEX_FILE.
# No staging, repository initialization, object writes or network access.
# Only an explicit private-only flag permits the owner-approved upstream source copy.
set -eu
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
exec "${PYTHON:-python3}" "$REPO/scripts/git_preflight.py" --repo "$REPO" "$@"
