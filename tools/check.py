#!/usr/bin/env python3
"""Oracle: hold the pipeline's retrievals to the manifest, and check the answers' citations.

    python3 tools/check.py

The retrieval contract is clink's VECTOR_SEARCH with filter_eq, as sql/answer.sql
calls it: for each query, the min(top_k, eligible) chunks nearest to it among the
chunks its metadata filter admits, each once, each with its cosine similarity,
which clink computes in float32. tools/rgen.py computes the same retrieval in
float64 from the tape alone and records for each query (data/manifest.json) how
many results it must return, the firm chunks, scoring more than 2 * SCORE_TOL
above the last of them, and the chunks tied at that boundary, within 2 * SCORE_TOL
of it, with their reference scores. SCORE_TOL (tools/embedder.py) bounds how far a
float32 score can sit from its float64 reference, so two chunks further apart than
twice that cannot change places, and two closer can. Nothing the oracle expects is
taken from the pipeline's output.

A query's retrieval passes when:
  - the manifest names it, and every query the manifest names was retrieved;
  - it returned the expected number of results, each chunk once;
  - every chunk is in the corpus (data/chunks.ndjson) and its filter admits it;
  - every score is a finite number within SCORE_TOL of the reference score;
  - the top result is the gold chunk, and every firm chunk is present;
  - every other result is one of the chunks tied at the boundary.

Then the citations. The answers are assembled deterministically, one per query,
from the top retrieved chunk, citing the chunks retrieved (out/answers.ndjson). No
answer may cite a chunk its query did not retrieve, and a negative control, an
answer citing a chunk that was not retrieved, must be caught. That shows citation
containment for this assembler. No language model writes these answers, so it
says nothing of the quality or factual accuracy of an answer one would write.
"""

import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedder import SCORE_TOL  # noqa: E402

SEVERITY = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def admitted(chunk, filt):
    """Whether a query's metadata filter admits a chunk, from the corpus's own
    metadata (the same rule tools/rgen.py applies)."""
    if filt.get("system") is not None and chunk["system"] != filt["system"]:
        return False
    if "min_severity" in filt and SEVERITY[chunk["severity"]] < SEVERITY[filt["min_severity"]]:
        return False
    return True


def valid_score(s):
    return isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s)


def check_retrievals(manifest, corpus, rows):
    """The ways the rows break the retrieval contract (none when they hold),
    and how many queries passed, and returned exactly the reference top-k."""
    fails = []
    by_q = collections.defaultdict(list)
    for i, r in enumerate(rows, 1):
        if not isinstance(r, dict) or not isinstance(r.get("query_id"), str) \
                or not isinstance(r.get("chunk_id"), str):
            fails.append(f"row {i}: no query_id or chunk_id")
            continue
        by_q[r["query_id"]].append(r)

    want = manifest["queries"]
    for qid in sorted(set(by_q) - set(want)):
        fails.append(f"{qid}: not a query the manifest names ({len(by_q[qid])} results)")
    for qid in sorted(set(want) - set(by_q)):
        fails.append(f"{qid}: no results")

    passed = exact = 0
    for qid in sorted(set(want) & set(by_q)):
        e, got, before = want[qid], by_q[qid], len(fails)
        ids = [r["chunk_id"] for r in got]
        if len(got) != e["expected_count"]:
            fails.append(f"{qid}: {len(got)} results, expected {e['expected_count']}")
        repeated = sorted(c for c, n in collections.Counter(ids).items() if n > 1)
        if repeated:
            fails.append(f"{qid}: {repeated} returned more than once")

        scores = {}
        for r in got:
            cid, s = r["chunk_id"], r.get("score")
            chunk = corpus.get(cid)
            if chunk is None:
                fails.append(f"{qid}: {cid} is not in the corpus")
            elif not admitted(chunk, e["filter"]):
                fails.append(f"{qid}: {cid} ({chunk['system']}) is outside the filter {e['filter']}")
            if not valid_score(s):
                fails.append(f"{qid}: {cid} has score {s!r}, not a finite number")
                continue
            ref = e["reference_scores"].get(cid)
            if ref is not None and abs(s - ref) > SCORE_TOL:
                fails.append(f"{qid}: {cid} scored {s:.9f} against the reference {ref:.9f}, "
                             f"more than {SCORE_TOL} apart")
            scores[cid] = max(s, scores.get(cid, s))

        ranked = sorted(scores, key=lambda c: (-scores[c], c))
        if not ranked or ranked[0] != e["gold"]:
            fails.append(f"{qid}: top result {ranked[0] if ranked else None}, not the gold {e['gold']}")
        missing = [c for c in e["firm"] if c not in ids]
        if missing:
            fails.append(f"{qid}: firm chunks {missing} missing")
        stray = [c for c in dict.fromkeys(ids) if c in corpus and c not in e["firm"] and c not in e["tied"]]
        if stray:
            fails.append(f"{qid}: {stray} neither firm nor tied at the boundary "
                         f"({e['boundary_score']:.9f} +/- {2 * SCORE_TOL})")
        if len(fails) == before:
            passed += 1
            exact += set(ids) == set(e["expected_topk"])
    return fails, {"queries": len(want), "passed": passed, "exact": exact}


def assemble(rows):
    """One answer per query: the text of its top retrieved chunk, citing every
    chunk it retrieved."""
    by_q = collections.defaultdict(list)
    for r in rows:
        if isinstance(r, dict) and isinstance(r.get("query_id"), str):
            by_q[r["query_id"]].append(r)
    answers = []
    for qid in sorted(by_q):
        ranked = sorted(by_q[qid], key=lambda r: (-r["score"] if valid_score(r.get("score")) else math.inf,
                                                  str(r.get("chunk_id"))))
        answers.append({"query_id": qid, "answer": ranked[0].get("chunk_text", ""),
                        "citations": [r.get("chunk_id") for r in ranked]})
    return answers


def cites_only_retrieved(citations, retrieved):
    return set(citations) <= set(retrieved)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = json.load(open(os.path.join(root, "data", "manifest.json")))
    if manifest.get("score_tolerance") != SCORE_TOL:
        print(f"FAIL: the manifest was written with score_tolerance {manifest.get('score_tolerance')}, "
              f"the oracle checks against {SCORE_TOL}: regenerate it with tools/rgen.py")
        return 1
    corpus = {}
    for line in open(os.path.join(root, "data", "chunks.ndjson")):
        if line.strip():
            c = json.loads(line)
            corpus[c["chunk_id"]] = c
    path = os.path.join(root, "out", "retrievals.ndjson")
    if not os.path.exists(path):
        print(f"FAIL: {path} not found (did the pipeline run?)")
        return 1
    rows, fails = [], []
    for i, line in enumerate(open(path), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                fails.append(f"{path} line {i} is not JSON")

    retrieval_fails, stats = check_retrievals(manifest, corpus, rows)
    fails += retrieval_fails

    answers = assemble(rows)
    retrieved = collections.defaultdict(set)
    for r in rows:
        if isinstance(r, dict):
            retrieved[r.get("query_id")].add(r.get("chunk_id"))
    contained = 0
    for a in answers:
        if cites_only_retrieved(a["citations"], retrieved[a["query_id"]]):
            contained += 1
        else:
            fails.append(f"{a['query_id']}: answer cites chunks outside its retrieval: "
                         f"{sorted(set(a['citations']) - retrieved[a['query_id']])}")
    # The negative control: a citation of a chunk the query did not retrieve
    # must be caught, or the citation check would be vacuous.
    some = answers[0]["query_id"] if answers else None
    if some is None or cites_only_retrieved(["__not_retrieved__"], retrieved[some]):
        fails.append("citation control not caught: a citation of a chunk that was not retrieved passed")
    with open(os.path.join(root, "out", "answers.ndjson"), "w") as fh:
        for a in answers:
            fh.write(json.dumps(a) + "\n")

    filt = [e for e in manifest["queries"].values() if e["filter"]]
    genuine = sum(1 for e in filt if e.get("unfiltered_top1") and e["unfiltered_top1"] != e["gold"])
    k = manifest["top_k"]
    print(f"retrieval: {stats['passed']}/{stats['queries']} queries return the expected results: the gold "
          f"chunk first, every firm chunk, the rest tied at the boundary, each score within "
          f"{SCORE_TOL} of the reference")
    print(f"           ({stats['exact']}/{stats['queries']} return exactly the reference top-{k}; the rest "
          f"differ only among chunks tied at the boundary)")
    print(f"metadata filter: {len(filt)} queries scoped by system, {genuine} of which would retrieve the "
          f"wrong system unscoped")
    print(f"citations: {contained}/{len(answers)} answers cite only chunks their query retrieved, and a "
          f"citation of one it did not is caught (control); a deterministic assembler writes them, no "
          f"language model")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  -", f)
        return 1
    print("\nOK: every query returns its gold chunk first and only the chunks the reference allows;")
    print("    the metadata-scoped queries retrieve the right system; every answer cites only")
    print("    retrieved chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
