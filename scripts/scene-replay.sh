#!/usr/bin/env bash
# Replay for audit: every retrieval is reproducible. This scene runs the retrieval
# pipeline with the flight recorder armed, then - offline, engine stopped - inspects
# what the VECTOR_SEARCH operator consumed, replays its captured epoch, and verifies
# it re-executes byte-identically. For a compliance question ("which runbook chunks
# grounded this answer, and can you prove it?"), the captured epoch plus the corpus
# reproduce the exact retrieval, and `clink replay --emit-test` freezes it into a
# standalone regression bundle.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLINK="${CLINK:-${ROOT}/.clink/prefix/bin/clink}"
PORT=8899
cd "${ROOT}"

[[ -x "${CLINK}" ]] || { echo "clink not found at ${CLINK}; run scripts/get-clink.sh first" >&2; exit 2; }

python3 tools/rgen.py >/dev/null
mkdir -p out
rm -rf out/ckpt out/capture out/replay-bundle
rm -f out/corpus.ndjson out/queries_embedded.ndjson out/retrievals.ndjson

python3 tools/embed_server.py --port "${PORT}" &
SRV=$!
trap 'kill "${SRV}" 2>/dev/null || true' EXIT
for _ in $(seq 1 100); do
    curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1 && break
    sleep 0.1
done

echo "== run the retrieval pipeline with the flight recorder armed"
"${CLINK}" run sql/index.sql
"${CLINK}" run sql/answer.sql \
    --checkpoint-dir=out/ckpt --capture-dir=out/capture --checkpoint-interval-ms=200

# Discover the VECTOR_SEARCH operator's capture + epoch from the tree.
read -r VS_OP EPOCH < <(python3 - <<'PY'
import glob, json, os, re
op_id, epoch = "", ""
for opjson in glob.glob("out/capture/op-*/subtask-*/op.json"):
    d = json.load(open(opjson))
    if d.get("op_type") == "vector_search_row":
        op_id = os.path.basename(os.path.dirname(os.path.dirname(opjson)))[len("op-"):]
        caps = glob.glob(os.path.join(os.path.dirname(opjson), "epoch-*.cap"))
        if caps:
            epoch = re.search(r"epoch-(\d+)\.cap", sorted(caps)[-1]).group(1)
print(op_id, epoch)
PY
)
[[ -n "${VS_OP}" && -n "${EPOCH}" ]] || { echo "FAIL: no vector_search_row capture found" >&2; exit 1; }
echo "   captured vector_search_row op=${VS_OP} epoch=${EPOCH}"

echo "== capture-cat: what the retrieval operator consumed (the embedded queries)"
"${CLINK}" capture-cat --file="out/capture/op-${VS_OP}/subtask-1/epoch-${EPOCH}.cap" --max-rows=2 2>&1 | tail -6

echo "== replay --verify: re-execute the captured epoch, offline, byte-identical"
VERIFY="$("${CLINK}" replay --capture-dir=out/capture --checkpoint-dir=out/ckpt \
    --op="${VS_OP}" --epoch="${EPOCH}" --verify 2>&1 | grep -vE 'registry.replace|coordinator\.')"
echo "${VERIFY}" | tail -4

echo "== emit-test: freeze the retrieval into a self-contained regression bundle"
"${CLINK}" replay --capture-dir=out/capture --checkpoint-dir=out/ckpt \
    --op="${VS_OP}" --epoch="${EPOCH}" --emit-test=out/replay-bundle >/dev/null 2>&1

echo
if ! echo "${VERIFY}" | grep -q "deterministic:"; then
    echo "FAIL: replay --verify did not report a deterministic re-execution"; exit 1
fi
if [[ ! -f out/replay-bundle/bundle.json || ! -f out/replay-bundle/golden.ndjson ]]; then
    echo "FAIL: replay --emit-test did not produce a bundle"; exit 1
fi
echo "OK: the VECTOR_SEARCH retrieval replayed byte-identically from its captured epoch,"
echo "    with no plugin, and froze into out/replay-bundle - the audit record is reproducible."
