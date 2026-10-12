"""Throw a bunch of random jobs at the API so the Grafana dashboard has something to show.

    python scripts/load_test.py --jobs 30 --url http://localhost:8000
"""

import argparse
import json
import random
import time
import urllib.request

INITS = ["hotspot", "edge", "rings", "random"]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--jobs", type=int, default=20)
    p.add_argument("--delay", type=float, default=1.0, help="mean seconds between submissions")
    p.add_argument("--backend", default=None)
    p.add_argument("--wait", action="store_true", help="wait for everything to finish at the end")
    args = p.parse_args()

    ids = []
    for i in range(args.jobs):
        body = {
            "n": random.choice([256, 512, 1024, 2048]),
            "steps": random.choice([500, 1000, 2000, 5000]),
            "init": random.choice(INITS),
        }
        if args.backend:
            body["backend"] = args.backend
        job = post(f"{args.url}/jobs", body)
        ids.append(job["id"])
        print(f"[{i + 1}/{args.jobs}] {job['id']}  n={body['n']} steps={body['steps']} backend={job['params']['backend']}")
        time.sleep(random.expovariate(1 / args.delay))

    if args.wait:
        pending = set(ids)
        while pending:
            for jid in list(pending):
                j = get(f"{args.url}/jobs/{jid}")
                if j["status"] in ("done", "failed"):
                    pending.discard(jid)
                    extra = f"{j['result']['mlups']:.0f} MLUPS" if j["status"] == "done" else j["error"]
                    print(f"{jid} {j['status']} {extra}")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
