-- grounded: index build.
--
-- Embed every runbook chunk and every query through the HTTP embedding model,
-- writing the vector corpus and the embedded queries. The model is a plain
-- ML_PREDICT http provider pointed at the local embed server (deterministic in
-- CI, a real embedding API in the live mode). Run from the repository root.

CREATE MODEL embedder
  INPUT (text TEXT)
  OUTPUT (vec DOUBLE PRECISION ARRAY)
  WITH ('provider' = 'http',
        'endpoint' = 'http://127.0.0.1:8899/embed',
        'max_batch_size' = '32');

CREATE TABLE chunks (
  chunk_id TEXT, doc_id TEXT, system TEXT, severity TEXT, title TEXT, text TEXT
) WITH (connector = 'file', format = 'json', path = 'data/chunks.ndjson');

CREATE TABLE corpus (
  chunk_id TEXT, doc_id TEXT, system TEXT, severity TEXT, title TEXT,
  chunk_text TEXT, vec DOUBLE PRECISION ARRAY
) WITH (connector = 'file', format = 'json', path = 'out/corpus.ndjson');

INSERT INTO corpus
  SELECT chunk_id, doc_id, system, severity, title, text AS chunk_text, vec
  FROM ML_PREDICT(TABLE chunks, MODEL embedder, DESCRIPTOR(text));

CREATE TABLE queries (
  query_id TEXT, text TEXT
) WITH (connector = 'file', format = 'json', path = 'data/queries.ndjson');

CREATE TABLE queries_embedded (
  query_id TEXT, qtext TEXT, emb DOUBLE PRECISION ARRAY
) WITH (connector = 'file', format = 'json', path = 'out/queries_embedded.ndjson');

INSERT INTO queries_embedded
  SELECT query_id, text AS qtext, vec AS emb
  FROM ML_PREDICT(TABLE queries, MODEL embedder, DESCRIPTOR(text));
