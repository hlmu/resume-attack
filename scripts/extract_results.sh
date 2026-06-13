#!/bin/bash
# Extract the compressed paper results (results/paper_results.tar.gz) into
# results/. Run `git lfs pull` first to fetch the archive. Idempotent: skips
# extraction if the directories are already present.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ARCHIVE=results/paper_results.tar.gz

if [ -d results/revision_multi_seed_aligned ]; then
  echo "Results already extracted; nothing to do."
  exit 0
fi

if [ ! -f "$ARCHIVE" ] || [ "$(stat -c%s "$ARCHIVE" 2>/dev/null || echo 0)" -lt 1000000 ]; then
  echo "ERROR: $ARCHIVE missing or is an unfetched LFS pointer." >&2
  echo "Run: git lfs install && git lfs pull" >&2
  exit 1
fi

echo "Extracting $ARCHIVE ..."
tar xzf "$ARCHIVE"
echo "Done. Result directories restored under results/."
