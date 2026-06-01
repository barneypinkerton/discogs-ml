#!/usr/bin/env bash
# Create ~/DiscogsData layout (safe to re-run).
set -euo pipefail
ROOT="${DISCOGS_DATA_ROOT:-$HOME/DiscogsData}"
mkdir -p "$ROOT"/{dumps,xml,csv,db,catalog,embeddings,profile,exports,candidate_audio}
echo "Data directories ready at: $ROOT"
ls -la "$ROOT"
