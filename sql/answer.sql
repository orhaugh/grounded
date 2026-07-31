-- grounded: retrieve.
--
-- VECTOR_SEARCH each embedded query against the vector corpus and write the
-- top-k retrieved chunks with their cosine scores. Round 2 runs UNFILTERED: the
-- query's metadata filter is not yet applied (the metadata-filtered VECTOR_SEARCH
-- is the round-3 engine feature), so the filter queries deliberately retrieve the
-- wrong system here - which the oracle checks against the manifest to show why the
-- filter is needed. Run from the repository root, after index.sql.

CREATE TABLE corpus (
  chunk_id TEXT, doc_id TEXT, system TEXT, severity TEXT, title TEXT,
  chunk_text TEXT, vec DOUBLE PRECISION ARRAY
) WITH (connector = 'file', format = 'json', path = 'out/corpus.ndjson');

CREATE TABLE queries_embedded (
  query_id TEXT, qtext TEXT, f_system TEXT, emb DOUBLE PRECISION ARRAY
) WITH (connector = 'file', format = 'json', path = 'out/queries_embedded.ndjson');

CREATE TABLE retrievals (
  query_id TEXT, chunk_id TEXT, system TEXT, chunk_text TEXT, score DOUBLE PRECISION
) WITH (connector = 'file', format = 'json', path = 'out/retrievals.ndjson');

-- filter_eq scopes each query's search to corpus rows whose `system` equals the
-- query's `f_system` (null = no constraint), applied as a metadata pre-filter so
-- the filter queries retrieve the right system instead of the nearest overall.
INSERT INTO retrievals
  SELECT query_id, chunk_id, system, chunk_text, score
  FROM VECTOR_SEARCH(TABLE queries_embedded, emb, corpus, DESCRIPTOR(vec), 3,
                     metric = 'cosine', filter_eq = 'f_system:system');
