#!/usr/bin/env python3
"""A tiny HTTP embedding model server, the target of clink's ML_PREDICT http provider.

clink's `CREATE MODEL ... WITH ('provider'='http', 'endpoint'='http://.../embed')`
POSTs each feature row as a JSON object and reads the model OUTPUT columns back by
name. For an embedder that is:

    POST /embed   {"text": "..."}                 ->  {"vec": [f0, f1, ...]}
    POST /embed   [{"text": "..."}, {"text": ...}] ->  [{"vec": [...]}, {"vec": [...]}]

(the array form is the batched path, one prediction per input, same order).

Default mode is DETERMINISTIC: it computes the shared hashed bag-of-words embedding
from embedder.py, so the vectors it serves are byte-identical to the ones rgen used
to build the manifest, and retrieval is reproducible with no model download and no
network. This is the CI-green path.

A real semantic embedding model could serve the same contract in its place; the
deterministic gate does not cover that.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from embedder import D, embed


def embed_one(obj):
    return {"vec": embed(str(obj.get("text", "")))}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the server quiet under a run
        pass

    def do_GET(self):
        # Readiness probe for run.sh.
        self._send(200, {"status": "ok", "dim": D})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"null")
        except (ValueError, json.JSONDecodeError) as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        if isinstance(payload, list):
            self._send(200, [embed_one(o) for o in payload])
        elif isinstance(payload, dict):
            self._send(200, embed_one(payload))
        else:
            self._send(400, {"error": "expected a JSON object or array"})

    def _send(self, code, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"embed_server: deterministic FNV-1a hashed-BoW, dim={D}, "
          f"listening on http://{args.host}:{args.port}/embed", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
