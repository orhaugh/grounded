# grounded

Streaming retrieval-augmented generation on [clink](https://github.com/orhaugh/clink)'s
AI surface. A deterministic synthetic corpus of ops and security runbooks is
chunked, embedded and held as a live vector index; a query stream is embedded,
retrieved against that index with `VECTOR_SEARCH`, and answered from the chunks
it retrieves. Retrieval is deterministic and oracle-gated; the answer is a
grounded assembly of the retrieved chunks by default, with an optional live LLM.

It is a downstream consumer, not part of clink: it installs clink into a local
prefix and builds against the installed CMake package the way any project would.
`scripts/get-clink.sh` installs the pinned release.

Where [card-sentry](https://github.com/orhaugh/card-sentry) is detection on
clink's CEP engine, grounded exercises the SQL-native AI surface: `CREATE MODEL`,
`ML_PREDICT` and `VECTOR_SEARCH`, over a streaming ingest-embed-retrieve pipeline.

## The tape

`tools/rgen.py` writes a deterministic corpus under `data/`: fifteen runbooks
across six systems (auth, database, network, payments, kubernetes, backup), each
tagged with a `system` and a `severity` and split into paragraph chunks. It also
writes a labelled query set and a `manifest.json` of ground-truth retrievals.

The embedder is a fully specified hashed bag-of-words cosine (FNV-1a, 512
dimensions, a fixed stopword list), so retrieval is reproducible with zero
dependencies and the same embedding runs identically in the oracle and in the
embed server the pipeline calls (`tools/embed_server.py`). The optional live mode
points the same `ML_PREDICT` endpoint at a real embedding API; that path is not
oracle-gated.

The generator self-checks every invariant the oracle relies on before it writes:

- Each query's intended chunk is the top-1 result under its filter.
- Each metadata-filter query would retrieve the **wrong system's** runbook
  without the filter. For example, "how do we fail over during an outage"
  unscoped surfaces the DNS-resolver runbook, but scoped to `payments` it
  returns the payment-gateway failover step. This is what the metadata-filtered
  retrieval earns.

## The pipeline

`scripts/run.sh` is the one command: it generates the tape, starts the embedding
server, and runs two `clink run` stages against the installed CLI.

- `sql/index.sql` embeds every chunk and every query through `ML_PREDICT`
  (`CREATE MODEL embedder ... WITH ('provider' = 'http', ...)`), writing the
  vector corpus and the embedded queries.
- `sql/answer.sql` retrieves each query's nearest chunks with `VECTOR_SEARCH` over
  that corpus and writes the results.

`tools/embed_server.py` is the embedding model the pipeline calls over HTTP - the
realistic RAG shape, and the seam where a real embedding API drops in for the live
mode. `tools/check.py` then gates the retrievals against the manifest: every plain
query must retrieve its gold chunk as top-1 and every firmly-ranked chunk (tied
tail chunks, which the reference and clink's cosine kernel may order differently,
may vary). The metadata-filter queries currently run unfiltered and reproduce the
wrong-system result the manifest predicts; scoping retrieval by `system` and
`severity` is the piece that follows.

```bash
scripts/get-clink.sh     # once: install clink into .clink/prefix
scripts/run.sh           # generate -> embed -> index -> retrieve -> gate
```

A good run ends on the oracle's verdict:

```
plain queries:  15/15 retrieve their gold chunk as top-1 and every firmly-ranked chunk
filter queries: 6/6 reproduce the manifest's unfiltered result
OK: ...
```
