-- Live-corpus scene: VECTOR_SEARCH the probe against the corpus currently on disk.
-- Run once before and once after the hotfix runbook is appended to out/corpus.ndjson;
-- each run re-opens (re-drains) the corpus, so the second run sees the added runbook.

CREATE TABLE corpus (
  chunk_id TEXT, doc_id TEXT, system TEXT, severity TEXT, title TEXT,
  chunk_text TEXT, vec DOUBLE PRECISION ARRAY
) WITH (connector = 'file', format = 'json', path = 'out/corpus.ndjson');

CREATE TABLE probe_embedded (
  query_id TEXT, qtext TEXT, f_system TEXT, emb DOUBLE PRECISION ARRAY
) WITH (connector = 'file', format = 'json', path = 'out/probe_embedded.ndjson');

CREATE TABLE probe_result (
  query_id TEXT, chunk_id TEXT, system TEXT, score DOUBLE PRECISION
) WITH (connector = 'file', format = 'json', path = 'out/probe_result.ndjson');

INSERT INTO probe_result
  SELECT query_id, chunk_id, system, score
  FROM VECTOR_SEARCH(TABLE probe_embedded, emb, corpus, DESCRIPTOR(vec), 1, metric = 'cosine');
