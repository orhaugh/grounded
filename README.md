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
dependencies and the same embedding runs identically in Python (the oracle) and
in the clink pipeline (a C++ closure provider). The optional live mode swaps in a
real semantic embedding model via `ML_PREDICT`; that path is not oracle-gated.

The generator self-checks every invariant the oracle relies on before it writes:

- Each query's intended chunk is the top-1 result under its filter.
- Each metadata-filter query would retrieve the **wrong system's** runbook
  without the filter. For example, "fail over to the backup during the outage"
  scoped to `payments` returns the payment-gateway failover step and scoped to
  `network` returns the DNS failover step, while unscoped it would cite the
  backup-corruption runbook instead. This is what the metadata-filtered
  retrieval earns.

## Running

```bash
python3 tools/rgen.py    # writes the corpus, queries and manifest to data/
```

## Status

The tape and its retrieval oracle are in place and deterministic. The streaming
pipeline that consumes them is being built against a local clink checkout.
