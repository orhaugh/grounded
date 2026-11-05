"""tools/check.py against retrievals made here: the reference passes, a
substitution among chunks tied at the boundary passes, and each way of
breaking the retrieval contract fails with its own reason.

    python3 -m unittest discover -s tools -p 'test_*.py' -v

Most cases use a small corpus and manifest written out below, so each mutation
is plain to read; the last ones use the real tape (tools/rgen.py), so the
oracle is also shown passing the reference retrieval of every real query and a
real boundary substitution.
"""
import copy
import json
import math
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check  # noqa: E402
from embedder import SCORE_TOL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chunk(cid, system):
    return {"chunk_id": cid, "doc_id": cid.split("#")[0], "system": system, "severity": "high",
            "title": cid, "text": f"text of {cid}"}


CORPUS = {c["chunk_id"]: c for c in [
    chunk("auth-a#0", "auth"), chunk("auth-a#1", "auth"), chunk("auth-a#2", "auth"),
    chunk("auth-a#3", "auth"), chunk("net-b#0", "network"), chunk("net-b#1", "network"),
    chunk("net-b#2", "network"), chunk("net-b#3", "network"),
]}

# q1 is unscoped: two firm chunks, and auth-a#2 and net-b#1 tied at the
# boundary (0.2); auth-a#3 sits below it, more than the window away. f1 is
# scoped to network: one firm chunk above two tied at 0.0, for two slots.
MANIFEST = {"top_k": 3, "score_tolerance": SCORE_TOL, "queries": {
    "q1": {"filter": {}, "gold": "auth-a#0", "expected_count": 3,
           "expected_topk": ["auth-a#0", "auth-a#1", "auth-a#2"],
           "boundary_score": 0.2, "firm": ["auth-a#0", "auth-a#1"], "tied": ["auth-a#2", "net-b#1"],
           "reference_scores": {"auth-a#0": 0.9, "auth-a#1": 0.5, "auth-a#2": 0.2, "net-b#1": 0.2}},
    "f1": {"filter": {"system": "network"}, "gold": "net-b#0", "expected_count": 3,
           "expected_topk": ["net-b#0", "net-b#1", "net-b#2"],
           "boundary_score": 0.0, "firm": ["net-b#0", "net-b#1"], "tied": ["net-b#2", "net-b#3"],
           "reference_scores": {"net-b#0": 0.7, "net-b#1": 0.3, "net-b#2": 0.0, "net-b#3": 0.0}},
}}


def rows_for(manifest, picks=None):
    """The rows a correct pipeline writes: each query's reference top-k (or the
    picks given), each score its reference moved by a float32-sized amount."""
    rows = []
    for qid, e in manifest["queries"].items():
        for cid in (picks or {}).get(qid, e["expected_topk"]):
            rows.append({"query_id": qid, "chunk_id": cid, "system": CORPUS.get(cid, {}).get("system"),
                         "chunk_text": "", "score": e["reference_scores"].get(cid, 0.1) + 3e-8})
    return rows


class Oracle(unittest.TestCase):
    def judge(self, rows, manifest=MANIFEST, corpus=CORPUS):
        return check.check_retrievals(manifest, corpus, rows)[0]

    def assertFails(self, rows, *words):
        fails = self.judge(rows)
        self.assertTrue(fails, "the retrieval passed")
        self.assertTrue(any(all(w in f for w in words) for f in fails), f"no failure names {words}: {fails}")

    def without(self, rows, qid, cid):
        return [r for r in rows if not (r["query_id"] == qid and r["chunk_id"] == cid)]

    def test_the_reference_retrieval_passes(self):
        self.assertEqual(self.judge(rows_for(MANIFEST)), [])

    def test_a_substitution_among_chunks_tied_at_the_boundary_passes(self):
        rows = rows_for(MANIFEST, {"q1": ["auth-a#0", "auth-a#1", "net-b#1"],
                                   "f1": ["net-b#0", "net-b#1", "net-b#3"]})
        self.assertEqual(self.judge(rows), [])

    def test_a_missing_query_fails(self):
        rows = [r for r in rows_for(MANIFEST) if r["query_id"] != "f1"]
        self.assertFails(rows, "f1", "no results")

    def test_an_unexpected_query_fails(self):
        rows = rows_for(MANIFEST) + [{"query_id": "q9", "chunk_id": "auth-a#0", "score": 0.5}]
        self.assertFails(rows, "q9", "not a query the manifest names")

    def test_too_few_results_fail(self):
        self.assertFails(self.without(rows_for(MANIFEST), "q1", "auth-a#2"), "q1", "2 results, expected 3")

    def test_too_many_results_fail(self):
        rows = rows_for(MANIFEST) + [{"query_id": "q1", "chunk_id": "net-b#1", "score": 0.2}]
        self.assertFails(rows, "q1", "4 results, expected 3")

    def test_a_duplicate_chunk_fails(self):
        rows = self.without(rows_for(MANIFEST), "q1", "auth-a#2")
        rows.append({"query_id": "q1", "chunk_id": "auth-a#1", "score": 0.5})
        self.assertFails(rows, "q1", "auth-a#1", "more than once")

    def test_an_unknown_chunk_fails(self):
        rows = self.without(rows_for(MANIFEST), "q1", "auth-a#2")
        rows.append({"query_id": "q1", "chunk_id": "made-up#7", "score": 0.2})
        self.assertFails(rows, "q1", "made-up#7", "not in the corpus")

    def test_a_chunk_outside_the_filter_fails(self):
        rows = self.without(rows_for(MANIFEST), "f1", "net-b#2")
        rows.append({"query_id": "f1", "chunk_id": "auth-a#3", "score": 0.0})
        self.assertFails(rows, "f1", "auth-a#3", "outside the filter")

    def test_a_missing_firm_chunk_fails(self):
        rows = self.without(rows_for(MANIFEST), "q1", "auth-a#1")
        rows.append({"query_id": "q1", "chunk_id": "net-b#1", "score": 0.2})
        self.assertFails(rows, "q1", "firm chunks ['auth-a#1'] missing")

    def test_an_incorrect_first_result_fails(self):
        rows = rows_for(MANIFEST)
        for r in rows:
            if r["query_id"] == "q1" and r["chunk_id"] == "auth-a#1":
                r["score"] = 0.95
        self.assertFails(rows, "q1", "top result auth-a#1, not the gold auth-a#0")

    def test_a_lower_ranked_untied_replacement_in_the_tail_fails(self):
        rows = self.without(rows_for(MANIFEST), "q1", "auth-a#2")
        rows.append({"query_id": "q1", "chunk_id": "auth-a#3", "score": 0.1})
        self.assertFails(rows, "q1", "auth-a#3", "neither firm nor tied")

    def test_a_chunk_just_outside_the_tie_window_fails(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["queries"]["q1"]["reference_scores"]["auth-a#3"] = 0.2 - 3 * SCORE_TOL
        rows = self.without(rows_for(manifest), "q1", "auth-a#2")
        rows.append({"query_id": "q1", "chunk_id": "auth-a#3", "score": 0.2 - 3 * SCORE_TOL})
        fails = self.judge(rows, manifest)
        self.assertTrue(any("neither firm nor tied" in f for f in fails), fails)

    def test_invalid_scores_fail(self):
        for bad in (math.nan, math.inf, -math.inf, None, "0.5", True):
            rows = rows_for(MANIFEST)
            rows[0]["score"] = bad
            with self.subTest(score=bad):
                self.assertFails(rows, rows[0]["query_id"], "not a finite number")
        rows = [dict(r) for r in rows_for(MANIFEST)]
        del rows[0]["score"]
        self.assertFails(rows, "not a finite number")

    def test_a_score_off_its_reference_fails(self):
        rows = rows_for(MANIFEST)
        for r in rows:
            if r["query_id"] == "q1" and r["chunk_id"] == "auth-a#1":
                r["score"] = 0.5 + 2 * SCORE_TOL
        self.assertFails(rows, "q1", "auth-a#1", "more than")

    def test_a_row_without_ids_fails(self):
        self.assertFails(rows_for(MANIFEST) + [{"score": 0.4}], "no query_id or chunk_id")


class RealTape(unittest.TestCase):
    """The same oracle over the tape tools/rgen.py writes, every query."""

    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "rgen.py")], check=True,
                       capture_output=True)
        with open(os.path.join(ROOT, "data", "manifest.json")) as f:
            cls.manifest = json.load(f)
        with open(os.path.join(ROOT, "data", "chunks.ndjson")) as f:
            cls.corpus = {c["chunk_id"]: c for c in map(json.loads, f)}

    def rows(self, picks=None):
        out = []
        for qid, e in self.manifest["queries"].items():
            for cid in (picks or {}).get(qid, e["expected_topk"]):
                out.append({"query_id": qid, "chunk_id": cid, "score": e["reference_scores"][cid]})
        return out

    def test_the_reference_retrieval_of_every_query_passes(self):
        fails, stats = check.check_retrievals(self.manifest, self.corpus, self.rows())
        self.assertEqual(fails, [])
        self.assertEqual(stats["passed"], len(self.manifest["queries"]))

    def test_a_real_boundary_substitution_passes(self):
        picks = {}
        for qid, e in self.manifest["queries"].items():
            spare = [c for c in e["tied"] if c not in e["expected_topk"]]
            if spare:
                tail = [c for c in e["expected_topk"] if c not in e["firm"]]
                picks[qid] = [c for c in e["expected_topk"] if c != tail[-1]] + [spare[0]]
        self.assertTrue(picks, "the tape has no query with more tied chunks than slots")
        fails, _ = check.check_retrievals(self.manifest, self.corpus, self.rows(picks))
        self.assertEqual(fails, [])

    def test_the_gold_chunk_clears_the_tie_window_in_every_query(self):
        for qid, e in self.manifest["queries"].items():
            with self.subTest(query=qid):
                self.assertIn(e["gold"], e["firm"])


if __name__ == "__main__":
    unittest.main()
