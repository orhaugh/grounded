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
mode. `tools/check.py` then gates every query against the manifest: it must
retrieve its gold chunk as top-1 and every firmly-ranked chunk (score-tied tail
chunks, which the reference and clink's cosine kernel may order differently, may
vary). The metadata-scoped queries pass their `system` scope to `VECTOR_SEARCH` as
a pre-filter, so they retrieve the right system rather than the nearest overall.
The oracle also assembles a grounded answer per query and gates that no answer
cites a chunk outside its retrieval - a hallucinated citation is caught.

```bash
scripts/get-clink.sh     # once: install clink into .clink/prefix
scripts/run.sh           # generate -> embed -> index -> retrieve -> gate
```

A good run ends on the oracle's verdict:

```
queries: 21/21 retrieve their gold chunk as top-1 and every firmly-ranked chunk
metadata filter: 6 queries scoped by system, 3 of which would retrieve the wrong system unscoped
grounding: 21/21 answers cite only retrieved chunks; the gate catches a hallucinated citation
OK: ...
```

## More scenes

Beyond the one-command gate, two scripts show what a real deployment needs:

- `scripts/scene-live-corpus.sh` - the index is not frozen at deploy. It probes
  the corpus for a runbook that does not exist yet, appends a hotfix runbook, and
  probes again: the query now retrieves the hotfix. On a running job
  `corpus_refresh_ms` rebuilds the index inline, without a restart.
- `scripts/scene-replay.sh` - every retrieval is reproducible. It runs the
  pipeline with the flight recorder armed, then offline replays the `VECTOR_SEARCH`
  operator's captured epoch, verifies it re-executes byte-identically, and freezes
  it into a self-contained regression bundle. For "which runbook chunks grounded
  this answer, and can you prove it?", the captured epoch and the corpus reproduce
  the exact retrieval.
