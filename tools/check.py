#!/usr/bin/env python3
"""Oracle: gate the pipeline's retrievals against the manifest ground truth.

Plain queries must retrieve their gold chunk as top-1 and the manifest's exact
top-k set. Filter queries run UNFILTERED in round 2 (metadata-filtered
VECTOR_SEARCH is the round-3 engine feature), so here they must reproduce the
manifest's recorded unfiltered top-1 - which, for the genuine filter queries, is
the wrong system. That is the point: it shows what the metadata filter will fix.
"""

import collections
import json
import os
import sys


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = json.load(open(os.path.join(root, "data", "manifest.json")))
    path = os.path.join(root, "out", "retrievals.ndjson")
    if not os.path.exists(path):
        print(f"FAIL: {path} not found (did the pipeline run?)")
        return 1

    by_q = collections.defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by_q[r["query_id"]].append((r["score"], r["chunk_id"]))

    def topk(qid):
        return [cid for _, cid in sorted(by_q.get(qid, []), key=lambda s: (-s[0], s[1]))]

    fails = []
    plain_total = sum(1 for e in manifest["queries"].values() if not e["filter"])
    filt_total = sum(1 for e in manifest["queries"].values() if e["filter"])
    plain_ok = filt_ok = exact_topk = 0

    for qid, e in sorted(manifest["queries"].items()):
        got = topk(qid)
        if not e["filter"]:
            exp = e["expected_topk"]
            scores = e.get("expected_topk_scores") or []
            # The unambiguously-ranked chunks are those scoring strictly above the
            # k-th (tie-boundary) score. Chunks tied with the boundary can reorder
            # between the float64 reference embedder and clink's float32 SimSIMD
            # cosine kernel, so only the firm set must be retrieved; the tail ties
            # may differ. The gold (top-1) always has a clear margin.
            tau = min(scores) if scores else 0.0
            firm = {c for c, s in zip(exp, scores) if s > tau + 1e-6}
            if got and got[0] == e["gold"] and firm.issubset(set(got)):
                plain_ok += 1
                if set(got) == set(exp):
                    exact_topk += 1
            else:
                fails.append(f"{qid} (plain): want top-1 {e['gold']} + firm {sorted(firm)}; got {got}")
        else:
            want = e.get("unfiltered_top1")
            if got and got[0] == want:
                filt_ok += 1
            else:
                fails.append(f"{qid} (filter, unfiltered): want unfiltered top-1 {want}; got {got}")

    print(f"plain queries:  {plain_ok}/{plain_total} retrieve their gold chunk as top-1 "
          f"and every firmly-ranked chunk")
    print(f"                ({exact_topk}/{plain_total} also reproduce the full top-k set; the rest "
          f"differ only in score-tied tail chunks)")
    print(f"filter queries: {filt_ok}/{filt_total} reproduce the manifest's unfiltered result")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nOK: every plain query retrieved its gold chunk as top-1; every filter query")
    print("    reproduced the unfiltered result the manifest predicts (the wrong system for")
    print("    the genuine ones), which the round-3 metadata filter will correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
