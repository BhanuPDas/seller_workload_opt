#!/usr/bin/env python3
"""
Simple load generator / smoke-test client. Posts randomized demand
requests to the API (mirroring generate_demands() from the original
simulation) and then polls /decision/<demand_id> for each one, printing
a summary table once all 4 algorithms have reported (or timing out on
whichever haven't -- expected for the batch methods if you send fewer
items than their S/BATCH_SIZE).

Usage:
  python scripts/send_test_demand.py --num-buyers 5 --num-items 60 --api http://localhost:8080
"""
import argparse
import random
import time

import requests


def make_demand(buyer_id):
    return {
        "buyer_id": buyer_id,
        "app_type": "test-workload",
        "ip": f"10.0.{random.randint(0,255)}.{random.randint(1,254)}",
        "lease_duration": 3600,
        "resources": {
            "cpu": {"demand_per_unit": random.randint(2, 30)},
            "mem": {"demand_per_unit": random.randint(4, 32)},
            "gpu": {"demand_per_unit": random.randint(0, 2)},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--num-buyers", type=int, default=5)
    ap.add_argument("--num-items", type=int, default=60)
    ap.add_argument("--interval", type=float, default=0.05, help="seconds between submits")
    ap.add_argument("--poll-timeout", type=float, default=30.0)
    args = ap.parse_args()

    buyers = [f"B{i}" for i in range(args.num_buyers)]
    demand_ids = []

    print(f"Submitting {args.num_items} demand items from {args.num_buyers} buyers to {args.api} ...")
    for i in range(args.num_items):
        buyer_id = random.choice(buyers)
        payload = make_demand(buyer_id)
        resp = requests.post(f"{args.api}/submit_demand", json=payload, timeout=5)
        resp.raise_for_status()
        body = resp.json()
        demand_ids.append(body["demand_id"])
        print(f"  [{i+1}/{args.num_items}] queued demand_id={body['demand_id']} buyer={buyer_id}")
        time.sleep(args.interval)

    print("\nPolling for results (batch methods may take longer to close their window/batch)...\n")
    deadline = time.time() + args.poll_timeout
    results = {}
    while time.time() < deadline and len(results) < len(demand_ids):
        for demand_id in demand_ids:
            if demand_id in results:
                continue
            resp = requests.get(f"{args.api}/decision/{demand_id}", timeout=5)
            body = resp.json()
            if body["comparison"]:
                results[demand_id] = body
        time.sleep(1)

    print(f"\n{len(results)}/{len(demand_ids)} demand items have a full 4-way comparison so far.\n")
    for demand_id, body in results.items():
        comp = body["comparison"]
        print(
            f"demand_id={demand_id[:8]}...  "
            f"recommended={comp.get('recommended_algorithm')} -> {comp.get('recommended_seller')}  "
            f"agreement={comp.get('agreement')}"
        )

    still_pending = [d for d in demand_ids if d not in results]
    if still_pending:
        print(f"\n{len(still_pending)} item(s) still pending (poll them yourself via /decision/<id>):")
        for d in still_pending[:10]:
            print(f"  {d}")


if __name__ == "__main__":
    main()
