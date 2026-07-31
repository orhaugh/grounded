-- Live-corpus scene: embed the probe query and the hotfix runbook's chunks through
-- the same model, so the probe can be searched and the hotfix appended to the corpus.
-- Assumes the base corpus (out/corpus.ndjson) was already built by sql/index.sql.

CREATE MODEL embedder
  INPUT (text TEXT)
  OUTPUT (vec DOUBLE PRECISION ARRAY)
  WITH ('provider' = 'http',
        'endpoint' = 'http://127.0.0.1:8899/embed',
        'max_batch_size' = '32');

CREATE TABLE probe (
  query_id TEXT, text TEXT, f_system TEXT
) WITH (connector = 'file', format = 'json', path = 'data/probe.ndjson');

CREATE TABLE probe_embedded (
  query_id TEXT, qtext TEXT, f_system TEXT, emb DOUBLE PRECISION ARRAY
) WITH (connector = 'file', format = 'json', path = 'out/probe_embedded.ndjson');

INSERT INTO probe_embedded
  SELECT query_id, text AS qtext, f_system, vec AS emb
  FROM ML_PREDICT(TABLE probe, MODEL embedder, DESCRIPTOR(text));

CREATE TABLE hotfix (
  chunk_id TEXT, doc_id TEXT, system TEXT, severity TEXT, title TEXT, text TEXT
) WITH (connector = 'file', format = 'json', path = 'data/hotfix.ndjson');

-- Same column layout as out/corpus.ndjson, so the rows can be appended to it.
CREATE TABLE hotfix_embedded (
  chunk_id TEXT, doc_id TEXT, system TEXT, severity TEXT, title TEXT,
  chunk_text TEXT, vec DOUBLE PRECISION ARRAY
) WITH (connector = 'file', format = 'json', path = 'out/hotfix_embedded.ndjson');

INSERT INTO hotfix_embedded
  SELECT chunk_id, doc_id, system, severity, title, text AS chunk_text, vec
  FROM ML_PREDICT(TABLE hotfix, MODEL embedder, DESCRIPTOR(text));
