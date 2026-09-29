#!/usr/bin/env python3
"""
Publishes synthetic seller state into SELLABLE_RESOURCES_HASH, in the
exact shape common/world.py expects (see its module docstring): a plain
Redis HASH where each field is a seller's node id and each value is that
seller's full JSON status document, e.g.

    HSET sellable_resources clab-nebula-extended-serf1 \
      '{"node": "clab-nebula-extended-serf1", "collected_at": "...", "sellable": {"cpu": 238.0, "ram": 1902.0, "GPU": 0.0, "storage": 687.0}}'

Use this to test the pipeline before real sellers are wired up, or to
simulate a seller's capacity changing / a seller going offline.

Usage:
  # one-shot: publish 48 sellers with random capacity once
  python scripts/publish_fake_seller.py --count 48

  # keep republishing every N seconds (simulates periodic seller heartbeats,
  # and keeps sellers from being expired by SELLER_STALE_AFTER_SECONDS --
  # note: a *new* random draw each round counts as a change and resets
  # the staleness clock, same as a real seller reporting a fresh reading)
  python scripts/publish_fake_seller.py --count 48 --loop --interval 30

  # publish/refresh just one seller with specific numbers
  python scripts/publish_fake_seller.py --node clab-node-3 --cpu 64 --ram 256 --gpu 4 --storage 500 --carbon 3.2

  # simulate a seller going offline (removes its field from the hash --
  # exercises the "vanished from the hash" removal path, not just the
  # SELLER_STALE_AFTER_SECONDS timeout path)
  python scripts/publish_fake_seller.py --remove clab-node-3
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import redis

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
SELLABLE_RESOURCES_HASH = os.environ.get("SELLABLE_RESOURCES_HASH", "sellable_resources")


def publish_one(r, node, cpu, ram, gpu, storage, carbon, region=None):
    doc = {
        "node": node,
        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sellable": {
            "cpu": cpu,
            "ram": ram,
            "GPU": gpu,
            "storage": storage,
        },
    }
    # carbon/region aren't part of the real seller schema (they're inputs
    # to the objective, not something a seller reports) but the parser
    # still accepts them at the top level of the document if a bridge
    # ever adds them.
    if carbon is not None:
        doc["carbon"] = carbon
    if region:
        doc["region"] = region
    r.hset(SELLABLE_RESOURCES_HASH, node, json.dumps(doc))
    print(
        f"HSET {SELLABLE_RESOURCES_HASH}[{node}] cpu={cpu} ram={ram} GPU={gpu} "
        f"storage={storage} carbon={carbon}"
    )


def remove_one(r, node):
    removed = r.hdel(SELLABLE_RESOURCES_HASH, node)
    print(f"HDEL {SELLABLE_RESOURCES_HASH}[{node}] -> {'removed' if removed else 'was not present'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", help="publish just this one seller instead of --count random ones")
    ap.add_argument("--remove", metavar="NODE", help="remove this node's field from the hash (simulate going offline) and exit")
    ap.add_argument("--cpu", type=float, default=None)
    ap.add_argument("--ram", type=float, default=None)
    ap.add_argument("--gpu", type=float, default=None)
    ap.add_argument("--storage", type=float, default=None)
    ap.add_argument("--carbon", type=float, default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--count", type=int, default=48, help="number of synthetic sellers (seed-node-0..seed-node-{count-1})")
    ap.add_argument("--loop", action="store_true", help="keep republishing forever (simulates periodic heartbeats)")
    ap.add_argument("--interval", type=float, default=30.0, help="seconds between republishes when --loop")
    args = ap.parse_args()

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()

    if args.remove:
        remove_one(r, args.remove)
        return

    def publish_round():
        if args.node:
            publish_one(
                r, args.node,
                cpu=args.cpu if args.cpu is not None else random.randint(32, 256),
                ram=args.ram if args.ram is not None else random.randint(128, 2048),
                gpu=args.gpu if args.gpu is not None else random.randint(0, 8),
                storage=args.storage if args.storage is not None else random.randint(100, 2000),
                carbon=args.carbon if args.carbon is not None else round(random.uniform(1, 10), 2),
                region=args.region,
            )
        else:
            for i in range(args.count):
                publish_one(
                    r, f"seed-node-{i}",
                    cpu=random.randint(32, 256),
                    ram=random.randint(128, 2048),
                    gpu=random.randint(0, 8),
                    storage=random.randint(100, 2000),
                    carbon=round(random.uniform(1, 10), 2),
                )

    publish_round()
    if args.loop:
        print(f"\nLooping every {args.interval}s -- Ctrl+C to stop.")
        try:
            while True:
                time.sleep(args.interval)
                publish_round()
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()
