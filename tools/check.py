#!/usr/bin/env python3
"""Oracle: gate the pipeline's retrievals and grounded answers against the manifest.

Retrieval gate: every query must retrieve its gold chunk as top-1 and every
firmly-ranked chunk. Score-tied tail chunks may reorder between the float64
reference embedder and clink's float32 SimSIMD kernel, so only the firm set (score
strictly above the k-th) is required. For a filter query, expected_topk is the
metadata-scoped ranking, so the same gate proves the pre-filter selected the right
system.

Grounding gate: assemble a grounded answer per query - it cites exactly the chunks
the query retrieved - and verify no answer cites a chunk outside its retrieval. A
hallucinated citation (the negative control) must be caught; that is the guard
that matters once a live LLM writes the answer instead of the deterministic
assembler.
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
    text_by = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by_q[r["query_id"]].append((r["score"], r["chunk_id"]))
            text_by[(r["query_id"], r["chunk_id"])] = r.get("chunk_text", "")

    def ranked(qid):
        return [cid for _, cid in sorted(by_q.get(qid, []), key=lambda s: (-s[0], s[1]))]

    # --- Retrieval gate ---
    fails = []
    total = len(manifest["queries"])
    ok = exact_topk = 0
    filt_total = filt_genuine = 0

    for qid, e in sorted(manifest["queries"].items()):
        got = ranked(qid)
        exp = e["expected_topk"]
        scores = e.get("expected_topk_scores") or []
        tau = min(scores) if scores else 0.0
        firm = {c for c, s in zip(exp, scores) if s > tau + 1e-6}
        if got and got[0] == e["gold"] and firm.issubset(set(got)):
            ok += 1
            if set(got) == set(exp):
                exact_topk += 1
        else:
            fails.append(f"{qid}: want top-1 {e['gold']} + firm {sorted(firm)}; got {got}")
        if e["filter"]:
            filt_total += 1
            # "genuine": unscoped, the same query would surface a different chunk (the
            # wrong system), so the filter is doing real work rather than decorating.
            if e.get("unfiltered_top1") and e["unfiltered_top1"] != e["gold"]:
                filt_genuine += 1

    # --- Grounding gate: an answer may cite only chunks its query retrieved. ---
    def is_grounded(citations, retrieved):
        return set(citations).issubset(set(retrieved))

    answers = []
    grounded_ok = 0
    for qid in sorted(by_q):
        got = ranked(qid)
        # The grounded answer stitches the top retrieved runbook step and cites exactly
        # the chunks it drew from (a live LLM would synthesise across the same chunks).
        answer = {"query_id": qid,
                  "answer": text_by.get((qid, got[0]), "") if got else "",
                  "citations": got}
        answers.append(answer)
        if is_grounded(answer["citations"], got):
            grounded_ok += 1
        else:
            outside = sorted(set(answer["citations"]) - set(got))
            fails.append(f"{qid}: answer cites chunks outside its retrieval: {outside}")

    # Negative control: an answer that cites a chunk it did NOT retrieve must be
    # caught, or the grounding gate would be vacuous.
    any_qid = next(iter(sorted(by_q)), None)
    control_caught = any_qid is not None and not is_grounded(
        ["__ungrounded_chunk__"], ranked(any_qid))
    if not control_caught:
        fails.append("grounding negative control not caught: a hallucinated citation passed")

    with open(os.path.join(root, "out", "answers.ndjson"), "w") as fh:
        for a in answers:
            fh.write(json.dumps(a) + "\n")

    # --- Report ---
    print(f"queries: {ok}/{total} retrieve their gold chunk as top-1 and every firmly-ranked chunk")
    print(f"         ({exact_topk}/{total} reproduce the full top-k set; the rest differ only in "
          f"score-tied tail chunks)")
    print(f"metadata filter: {filt_total} queries scoped by system, {filt_genuine} of which would")
    print(f"                 retrieve the WRONG system unscoped - the pre-filter corrects them")
    print(f"grounding: {grounded_ok}/{len(answers)} answers cite only retrieved chunks; the gate "
          f"catches a hallucinated citation (control)")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nOK: every query retrieves its gold chunk as top-1; the metadata-scoped queries")
    print("    retrieve the right system; every answer is grounded in its retrieved chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
