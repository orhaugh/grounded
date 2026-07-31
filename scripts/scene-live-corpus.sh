#!/usr/bin/env bash
# Live corpus: the vector index is not frozen at deploy. This scene builds the base
# corpus, probes it with a question about a runbook that does not exist yet, appends
# a hotfix runbook to the corpus, and probes again - the probe now retrieves the
# hotfix. The scene re-opens the index between probes; on a long-running job,
# `corpus_refresh_ms` rebuilds the index inline on the same operator without a
# restart (clink's operator test VectorSearchOperator.CorpusRefreshPicksUpChangedCorpus
# exercises that inline path).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLINK="${CLINK:-${ROOT}/.clink/prefix/bin/clink}"
PORT=8899   # must match the endpoint in the sql/*.sql
cd "${ROOT}"

[[ -x "${CLINK}" ]] || { echo "clink not found at ${CLINK}; run scripts/get-clink.sh first" >&2; exit 2; }

python3 tools/rgen.py >/dev/null
mkdir -p out
rm -f out/corpus.ndjson out/queries_embedded.ndjson out/probe_embedded.ndjson \
      out/hotfix_embedded.ndjson out/probe_result.ndjson

python3 tools/embed_server.py --port "${PORT}" &
SRV=$!
trap 'kill "${SRV}" 2>/dev/null || true' EXIT
for _ in $(seq 1 100); do
    curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 && break
    sleep 0.1
done

probe_top() {
    python3 -c "import json; r=[json.loads(l) for l in open('out/probe_result.ndjson') if l.strip()]; print(r[0]['chunk_id'] if r else 'NONE')"
}

echo "== build the base corpus, embed the probe and the hotfix runbook"
"${CLINK}" run sql/index.sql
"${CLINK}" run sql/lc_embed.sql

echo "== BEFORE: probe the base corpus (the hotfix runbook does not exist yet)"
rm -f out/probe_result.ndjson
"${CLINK}" run sql/lc_probe.sql
BEFORE="$(probe_top)"
echo "   probe -> ${BEFORE}"

echo "== append the hotfix runbook to the live corpus"
cat out/hotfix_embedded.ndjson >> out/corpus.ndjson

echo "== AFTER: probe again (corpus now includes the hotfix)"
rm -f out/probe_result.ndjson
"${CLINK}" run sql/lc_probe.sql
AFTER="$(probe_top)"
echo "   probe -> ${AFTER}"

HOTFIX_IDS="$(python3 -c "import json; print(','.join(json.loads(l)['chunk_id'] for l in open('data/hotfix.ndjson')))")"
echo
if [[ ",${HOTFIX_IDS}," != *",${AFTER},"* ]]; then
    echo "FAIL: after adding the hotfix the probe did not retrieve it (got ${AFTER})"
    exit 1
fi
if [[ ",${HOTFIX_IDS}," == *",${BEFORE},"* ]]; then
    echo "FAIL: the probe retrieved the hotfix from the base corpus (impossible)"
    exit 1
fi
echo "OK: the probe retrieved '${BEFORE}' before the runbook existed and '${AFTER}'"
echo "    after it was added to the live corpus - the index picked it up with no restart."
