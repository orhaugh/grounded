#!/usr/bin/env bash
# grounded: the one gate. Generate the tape, embed + index it through clink's
# ML_PREDICT, retrieve with VECTOR_SEARCH, and check the retrievals against the
# oracle. Runs entirely offline against the deterministic embed server.
#
#   scripts/get-clink.sh   # once, installs clink into .clink/prefix
#   scripts/run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLINK="${CLINK:-${ROOT}/.clink/prefix/bin/clink}"
PORT=8899   # must match the endpoint in sql/index.sql
cd "${ROOT}"

if [[ ! -x "${CLINK}" ]]; then
    echo "clink not found at ${CLINK}; run scripts/get-clink.sh first" >&2
    exit 2
fi

python3 tools/rgen.py
mkdir -p out
rm -f out/corpus.ndjson out/queries_embedded.ndjson out/retrievals.ndjson

echo "== embedding server on :${PORT} (deterministic)"
python3 tools/embed_server.py --port "${PORT}" &
SRV=$!
trap 'kill "${SRV}" 2>/dev/null || true' EXIT
for _ in $(seq 1 100); do
    curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 && break
    sleep 0.1
done

echo "== index: embed the corpus and the queries (ML_PREDICT)"
"${CLINK}" run sql/index.sql

echo "== retrieve: VECTOR_SEARCH the queries against the corpus"
"${CLINK}" run sql/answer.sql

echo "== gate the retrievals against the oracle"
python3 tools/check.py
