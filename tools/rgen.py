#!/usr/bin/env python3
"""Generate the grounded runbook tape: corpus, queries, and the retrieval oracle.

Deterministic by construction (no RNG needed - the corpus and queries are
authored, not sampled), so the same inputs produce byte-identical outputs every
run. Writes under ``data/``:

  docs.ndjson      one row per runbook document: {doc_id, system, severity,
                   title, body}. ``body`` holds paragraphs separated by a blank
                   line; the pipeline chunks on that boundary.
  chunks.ndjson    the derived chunks (reference copy): {chunk_id, doc_id,
                   system, severity, title, text}. chunk_id = "<doc_id>#<i>".
  queries.ndjson   one row per query: {query_id, text, filter}. ``filter`` is
                   {} or {system} / {min_severity} / both.
  manifest.json    ground truth: for each query, the ordered top-k chunk ids the
                   deterministic embedder retrieves (under the filter), the gold
                   citation (top-1), and - for filter queries - the top-1 the
                   SAME query would retrieve WITHOUT the filter, which is a
                   different, wrong chunk. That difference is what proves the
                   metadata pre-filter earns its keep.

The embedder is defined once in embedder.py and shared with the embed server
(tools/embed_server.py), so the manifest's ground-truth retrievals and the
vectors the pipeline actually indexes are computed by the same code. It is a
deterministic hashed bag-of-words cosine (FNV-1a, 512 dims, a fixed stopword
list); the optional live mode swaps in a real semantic embedding model behind the
same HTTP contract and is not oracle-gated.

manifest.json also records each query's expected_topk_scores, so the oracle can
require the firmly-ranked chunks (score clearly above the k-th) while tolerating
score-tied tail chunks, which the float64 reference and clink's float32 SimSIMD
cosine kernel may order differently.
"""

import json
import os

from embedder import D, STOPWORDS, cosine, embed

TOP_K = 3


SEVERITY = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# --- The corpus: hand-authored runbooks. Distinctive vocabulary per document
# keeps lexical retrieval unambiguous; the self-check at the end enforces it. ---
DOCS = [
    # ---- system: auth ----
    {
        "doc_id": "auth-leaked-credential", "system": "auth", "severity": "high",
        "title": "Responding to a leaked credential",
        "body": [
            "A credential is considered leaked once it appears in a public repository, a paste site, or a third-party breach corpus. Confirm the exposure by matching the secret fingerprint against the alert before acting.",
            "Rotate the affected credential immediately. Issue a replacement secret, deploy it to every consumer, and only then revoke the old value so that live traffic is never left without a working key.",
            "Revoke all sessions and derived tokens minted from the leaked credential. Force re-authentication for the owning principal and invalidate refresh tokens so a copied credential cannot be exchanged for a fresh one.",
            "Audit access logs for the exposure window. Record every principal, source address, and resource touched while the leaked credential was valid, and file the findings for the post-incident review.",
        ],
    },
    {
        "doc_id": "auth-oauth-token-abuse", "system": "auth", "severity": "high",
        "title": "Stolen OAuth token abuse",
        "body": [
            "An OAuth access token is being replayed from an unfamiliar client. The signal is a valid token presenting from a new device fingerprint and a geography the principal has never used.",
            "Revoke the stolen token and its refresh token at the authorization server. Terminate the linked session and require step-up multi-factor authentication before a new token is granted.",
            "Narrow the blast radius by tightening the scope and audience of newly issued tokens for the affected client, and shorten their lifetime until the investigation closes.",
        ],
    },
    {
        "doc_id": "auth-mfa-lockout", "system": "auth", "severity": "medium",
        "title": "User locked out of multi-factor authentication",
        "body": [
            "A user has lost their second factor and cannot complete the multi-factor challenge. Verify the person out of band before touching their enrolment, because the reset flow itself is a takeover vector.",
            "Once identity is proven, reset the enrolled authenticator and issue single-use recovery codes. Require the user to enrol a new factor at the next sign in and expire the recovery codes afterwards.",
        ],
    },
    # ---- system: database ----
    {
        "doc_id": "db-primary-failover", "system": "database", "severity": "critical",
        "title": "Primary database failover",
        "body": [
            "The primary database is unresponsive to writes and health checks. Confirm it is genuinely down rather than briefly overloaded before promoting, because a split brain is worse than a short outage.",
            "Promote the most caught-up replica to primary. Choose the replica with the smallest replication lag, fence the old primary so it cannot accept writes, and redirect the write endpoint to the newly promoted node.",
            "After promotion, reattach the remaining replicas to the new primary and rebuild any that diverged. Verify the write path end to end before declaring the incident resolved.",
        ],
    },
    {
        "doc_id": "db-replication-lag", "system": "database", "severity": "high",
        "title": "Diagnosing replication lag",
        "body": [
            "Replicas are serving stale reads because replication lag has grown beyond the freshness budget. Measure the lag as the delay between the primary's latest committed transaction and the replica's applied position.",
            "A single long-running transaction on the primary is the most common cause: it holds the write-ahead log and the replica cannot advance past it. Identify and, if safe, cancel the offending transaction.",
            "If the replica is simply underpowered, throttle read traffic away from it until it catches up, then scale its resources so the apply rate keeps pace with the primary.",
        ],
    },
    {
        "doc_id": "db-deadlock-storm", "system": "database", "severity": "medium",
        "title": "Deadlock storm",
        "body": [
            "Transactions are aborting in bursts as the engine detects deadlocks and rolls back victims. A deadlock is a cycle of lock waits where two transactions each hold what the other needs.",
            "Capture the deadlock graph from the engine log to see the competing statements. The usual fix is to make all callers acquire locks in the same order so the cycle can never form.",
        ],
    },
    # ---- system: network ----
    {
        "doc_id": "net-dns-outage", "system": "network", "severity": "critical",
        "title": "DNS resolver outage",
        "body": [
            "Name resolution is failing across the fleet and services cannot reach each other by hostname. Confirm the resolver is the fault by querying it directly and comparing against a known-good external resolver.",
            "Fail over to the secondary resolver by updating the resolver configuration pushed to hosts, and raise the cache time to live so a flapping upstream does not translate into repeated lookup failures.",
            "Once the primary resolver is healthy, drain it back gradually and watch the query error rate rather than switching all traffic back at once.",
        ],
    },
    {
        "doc_id": "net-tls-cert-expiry", "system": "network", "severity": "high",
        "title": "Expired TLS certificate",
        "body": [
            "Clients are rejecting connections because the server's TLS certificate has expired. The handshake fails with a certificate validity error and no application traffic gets through.",
            "Rotate the expired certificate: issue a renewed certificate from the authority, install the full chain, and reload the terminating proxy so the new certificate is served without dropping established connections.",
            "Prevent a repeat by wiring the certificate expiry date into monitoring so an alert fires weeks before the validity window closes.",
        ],
    },
    {
        "doc_id": "net-ddos-mitigation", "system": "network", "severity": "high",
        "title": "Volumetric DDoS mitigation",
        "body": [
            "A volumetric denial-of-service flood is saturating the edge and legitimate requests are being crowded out. Identify the attack signature from the traffic sample: source distribution, protocol, and target port.",
            "Engage the scrubbing provider and apply rate limits at the edge keyed on the attack signature, taking care not to rate-limit legitimate clients sharing the same network.",
        ],
    },
    # ---- system: payments ----
    {
        "doc_id": "pay-gateway-timeout", "system": "payments", "severity": "critical",
        "title": "Payment gateway timeouts",
        "body": [
            "The upstream payment processor is timing out and authorizations are hanging. Distinguish a processor outage from a network problem on our side by testing a second processor endpoint.",
            "Fail over to the backup processor for new authorizations while the primary is degraded. Queue any in-flight requests idempotently so a retry after failover cannot capture a payment twice.",
            "When the primary recovers, reconcile the two processors' settlement reports against our ledger to catch any authorization that succeeded downstream but was lost on our side.",
        ],
    },
    {
        "doc_id": "pay-double-charge", "system": "payments", "severity": "high",
        "title": "Duplicate charge on a customer",
        "body": [
            "A customer has been charged twice for one order because a capture was retried without an idempotency key. Confirm the duplicate by matching two captures to a single order reference.",
            "Refund the duplicate capture, not the original, and notify the customer. Then backfill the missing idempotency key on the capture path so the same retry cannot double charge again.",
        ],
    },
    # ---- system: kubernetes ----
    {
        "doc_id": "k8s-node-pressure", "system": "kubernetes", "severity": "high",
        "title": "Node under resource pressure",
        "body": [
            "A node is evicting pods because it is under memory pressure and the kubelet has crossed its eviction threshold. Identify the node and the pods being evicted from the eviction events.",
            "Cordon the node so the scheduler stops placing new pods on it, then drain the evictable workloads to healthy nodes. Scale the node pool out if the whole cluster is short of capacity.",
        ],
    },
    {
        "doc_id": "k8s-image-pull-failure", "system": "kubernetes", "severity": "medium",
        "title": "Image pull failure",
        "body": [
            "Pods are stuck because the container image cannot be pulled from the registry. The pod events show an authentication failure or a missing tag against the registry.",
            "Repair the pull secret so the kubelet can authenticate to the registry, and confirm the referenced image tag actually exists before rolling the deployment again.",
        ],
    },
    # ---- system: backup ----
    {
        "doc_id": "backup-corruption", "system": "backup", "severity": "high",
        "title": "Corrupt backup detected",
        "body": [
            "The latest backup fails its integrity check and its checksum does not match the manifest recorded when the backup was taken. Treat the backup as unusable until proven otherwise.",
            "Fall back to the most recent backup that passes verification, and quarantine the corrupt artifact for analysis rather than deleting it. A corrupt backup often points at a failing storage device upstream.",
        ],
    },
    {
        "doc_id": "backup-restore-drill", "system": "backup", "severity": "medium",
        "title": "Restore drill",
        "body": [
            "A restore drill proves the backups are usable, not just present. Restore the chosen snapshot into an isolated environment so a botched drill can never touch production data.",
            "Verify the restored data against known row counts and a checksum sample, and record the wall-clock restore time so the recovery objective is grounded in a measured number rather than a hope.",
        ],
    },
]


def build_chunks(docs):
    chunks = []
    for d in docs:
        for i, para in enumerate(d["body"]):
            chunks.append({
                "chunk_id": f"{d['doc_id']}#{i}",
                "doc_id": d["doc_id"],
                "system": d["system"],
                "severity": d["severity"],
                "title": d["title"],
                "text": para,
            })
    return chunks


def matches_filter(chunk, filt):
    if not filt:
        return True
    if "system" in filt and chunk["system"] != filt["system"]:
        return False
    if "min_severity" in filt and SEVERITY[chunk["severity"]] < SEVERITY[filt["min_severity"]]:
        return False
    return True


def retrieve(query_text, chunks, chunk_vecs, filt=None, k=TOP_K):
    q = embed(query_text)
    scored = [(cosine(q, chunk_vecs[c["chunk_id"]]), c["chunk_id"])
              for c in chunks if matches_filter(c, filt)]
    # Sort by score desc, chunk_id asc as a deterministic tie-break.
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [cid for _, cid in scored[:k]]


# --- The queries. `gold` is the chunk the query is meant to surface. Plain
# queries need no filter; filter queries collide lexically with a chunk in
# another system, and the filter is what disambiguates. ---
QUERIES = [
    # Plain queries, no filter needed. Each is phrased to overlap one target
    # paragraph strongly and the rest of the corpus weakly.
    {"query_id": "q01", "text": "how do we respond to a leaked credential that showed up in a public repo",
     "filter": {}, "gold": "auth-leaked-credential#0"},
    {"query_id": "q02", "text": "revoke sessions and refresh tokens after a credential is exposed",
     "filter": {}, "gold": "auth-leaked-credential#2"},
    {"query_id": "q03", "text": "a valid oauth access token is being replayed from an unfamiliar device",
     "filter": {}, "gold": "auth-oauth-token-abuse#0"},
    {"query_id": "q04", "text": "user lost their second factor and cannot complete the mfa challenge",
     "filter": {}, "gold": "auth-mfa-lockout#0"},
    {"query_id": "q05", "text": "promote the most caught up replica and fence the old primary",
     "filter": {}, "gold": "db-primary-failover#1"},
    {"query_id": "q06", "text": "replicas serving stale reads because replication lag grew too large",
     "filter": {}, "gold": "db-replication-lag#0"},
    {"query_id": "q07", "text": "transactions aborting in bursts from a cycle of lock waits",
     "filter": {}, "gold": "db-deadlock-storm#0"},
    {"query_id": "q08", "text": "fail over dns to the secondary resolver and raise the cache time to live",
     "filter": {}, "gold": "net-dns-outage#1"},
    {"query_id": "q09", "text": "engage the scrubbing provider and apply rate limits keyed on the attack signature",
     "filter": {}, "gold": "net-ddos-mitigation#1"},
    {"query_id": "q10", "text": "fail over to the backup processor and queue in flight authorizations idempotently",
     "filter": {}, "gold": "pay-gateway-timeout#1"},
    {"query_id": "q11", "text": "customer charged twice because a capture retried without an idempotency key",
     "filter": {}, "gold": "pay-double-charge#0"},
    {"query_id": "q12", "text": "cordon the node then drain the evictable workloads to healthy nodes",
     "filter": {}, "gold": "k8s-node-pressure#1"},
    {"query_id": "q13", "text": "repair the pull secret so the kubelet can authenticate to the registry",
     "filter": {}, "gold": "k8s-image-pull-failure#1"},
    {"query_id": "q14", "text": "latest backup fails its integrity check and the checksum does not match",
     "filter": {}, "gold": "backup-corruption#0"},
    {"query_id": "q15", "text": "prove backups are usable by restoring a snapshot into an isolated environment",
     "filter": {}, "gold": "backup-restore-drill#0"},

    # Filter queries. f01-f03 genuinely need the filter: unfiltered each lands
    # in a DIFFERENT system (annotated in collides_with_system), and only the
    # metadata filter returns the right system's runbook. "how do we fail over
    # during an outage" unscoped surfaces the DNS runbook, but scoped to
    # payments it returns the gateway failover step; "fail over to the backup
    # during the outage" unscoped surfaces the gateway runbook, scoped to
    # network the DNS one; the generic "rotate" query wins in auth unscoped but
    # scoped to network returns the certificate rotation. f04-f06 are sanity
    # filters (already in-system) that confirm scoping and the severity floor do
    # not break the intended result.
    {"query_id": "f01", "text": "how do we fail over during an outage",
     "filter": {"system": "payments"}, "gold": "pay-gateway-timeout#1",
     "collides_with_system": "network"},
    {"query_id": "f02", "text": "fail over to the backup during the outage",
     "filter": {"system": "network"}, "gold": "net-dns-outage#1",
     "collides_with_system": "payments"},
    {"query_id": "f03", "text": "rotate the expired secret and reload the service",
     "filter": {"system": "network"}, "gold": "net-tls-cert-expiry#1",
     "collides_with_system": "auth"},
    {"query_id": "f04", "text": "revoke the stolen token and terminate the session",
     "filter": {"system": "auth"}, "gold": "auth-oauth-token-abuse#1",
     "collides_with_system": "auth"},
    {"query_id": "f05", "text": "an idempotency key is missing on the capture path",
     "filter": {"system": "payments"}, "gold": "pay-double-charge#1",
     "collides_with_system": "payments"},
    {"query_id": "f06", "text": "the primary database is down and unresponsive to writes",
     "filter": {"system": "database", "min_severity": "critical"}, "gold": "db-primary-failover#0",
     "collides_with_system": "database"},
]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(root, "data")
    os.makedirs(data, exist_ok=True)

    chunks = build_chunks(DOCS)
    chunk_vecs = {c["chunk_id"]: embed(c["text"]) for c in chunks}
    by_id = {c["chunk_id"]: c for c in chunks}

    # --- Self-check: every design invariant the pipeline and oracle rely on. ---
    failures = []
    manifest = {"embedder": {"dim": D, "hash": "fnv1a64", "stopwords": sorted(STOPWORDS)},
                "top_k": TOP_K, "queries": {}}

    for q in QUERIES:
        gold = q["gold"]
        if gold not in by_id:
            failures.append(f"{q['query_id']}: gold {gold} is not a real chunk")
            continue
        topk = retrieve(q["text"], chunks, chunk_vecs, q.get("filter") or None)
        if not topk or topk[0] != gold:
            failures.append(f"{q['query_id']}: gold {gold} is not top-1 under filter; got {topk}")
        qv = embed(q["text"])
        entry = {"filter": q.get("filter") or {}, "gold": gold, "expected_topk": topk,
                 "expected_topk_scores": [round(cosine(qv, chunk_vecs[c]), 9) for c in topk]}

        # For a real filter query, the SAME query without the filter must
        # surface a DIFFERENT, wrong-system chunk - otherwise the filter is
        # decorative and proves nothing.
        filt = q.get("filter") or {}
        gold_system = by_id[gold]["system"]
        if filt:
            unfiltered_top1 = retrieve(q["text"], chunks, chunk_vecs, None)[0]
            entry["unfiltered_top1"] = unfiltered_top1
            entry["unfiltered_top1_system"] = by_id[unfiltered_top1]["system"]
            # q.get("collides_with_system") == the gold system means we only
            # assert the filter keeps us in-system (a sanity query, e.g. f03/f05).
            demands_filter = q.get("collides_with_system") not in (None, gold_system)
            if demands_filter and by_id[unfiltered_top1]["system"] == gold_system:
                failures.append(
                    f"{q['query_id']}: filter is decorative - unfiltered top-1 {unfiltered_top1} "
                    f"is already in the gold system {gold_system}")
            if demands_filter and unfiltered_top1 == gold:
                failures.append(
                    f"{q['query_id']}: filter changes nothing - unfiltered top-1 is already the gold chunk")
        manifest["queries"][q["query_id"]] = entry

    if failures:
        print("SELF-CHECK FAILED - the tape is not internally consistent:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)

    # --- Emit. ---
    with open(os.path.join(data, "docs.ndjson"), "w") as fh:
        for d in DOCS:
            fh.write(json.dumps({
                "doc_id": d["doc_id"], "system": d["system"], "severity": d["severity"],
                "title": d["title"], "body": "\n\n".join(d["body"]),
            }) + "\n")

    with open(os.path.join(data, "chunks.ndjson"), "w") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")

    with open(os.path.join(data, "queries.ndjson"), "w") as fh:
        for q in QUERIES:
            fh.write(json.dumps({
                "query_id": q["query_id"], "text": q["text"], "filter": q.get("filter") or {},
            }) + "\n")

    with open(os.path.join(data, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    n_filter = sum(1 for q in QUERIES if q.get("filter"))
    print(f"tape written to {data}/")
    print(f"  docs:    {len(DOCS)}")
    print(f"  chunks:  {len(chunks)}")
    print(f"  queries: {len(QUERIES)} ({n_filter} with a metadata filter)")
    print("  self-check: every query's gold chunk is top-1 under its filter,")
    print("              and every real filter query would retrieve the wrong")
    print("              system without it.")


if __name__ == "__main__":
    main()
