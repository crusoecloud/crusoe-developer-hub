#!/usr/bin/env python3
"""Self-Serve Deployment for the fine-tuned adapter (crusoe-self-serve-deployments skill).

GPU-hours bill from creation until deletion. Nothing billable happens without --yes.

    export CRUSOE_API_KEY='...' CRUSOE_PROJECT_ID='...'     # project id is in the Console URL
    python 03_deploy.py flavors --base-model "Qwen/Qwen3.5-2B"
    python 03_deploy.py create  --flavor <flavor_id> --yes     # fused LoRA, 1 replica
    python 03_deploy.py status --watch                         # up to ~40 min
    python 03_deploy.py smoke
    python 03_deploy.py delete --yes                           # stops billing
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from decision_model import TRUST_ENV, token

STATE = Path(__file__).parent / "run_state.json"
ALIAS = "agnews-decider"  # becomes the `model` string for inference


def api() -> httpx.Client:
    project = os.environ.get("CRUSOE_PROJECT_ID") or raise_("export CRUSOE_PROJECT_ID='<project id from the Console URL>'")
    return httpx.Client(
        base_url=f"https://api.crusoecloud.com/v1/projects/{project}/foundry/selfserve",
        headers={"Authorization": f"Bearer {token()}"}, trust_env=TRUST_ENV, timeout=60)


def raise_(msg):
    raise SystemExit(msg)


def state() -> dict:
    return json.loads(STATE.read_text())


def save(**kv):
    STATE.write_text(json.dumps(state() | kv, indent=2))


def cmd_flavors(a, c):
    flavors = c.get("/flavors").raise_for_status().json()["flavors"]
    print(f"{'flavor_id':<40} {'model':<24} {'type':<14} lora  $/hr")
    for f in flavors:
        if a.base_model.lower() in f["model_name"].lower():
            print(f"{f['id']:<40} {f['model_name']:<24} {f['flavor_type']:<14} {f.get('lora_suitable')!s:<5} {f['price_per_hour']}")


def cmd_create(a, c):
    flavor = next((f for f in c.get("/flavors").raise_for_status().json()["flavors"] if f["id"] == a.flavor), None)
    if flavor is None:
        raise SystemExit("unknown flavor id; run `flavors` and copy one exactly")
    if not flavor.get("lora_suitable"):
        raise SystemExit("this flavor is not lora_suitable; pick one that is")
    # The API spec spells the field `fine_tuned_model`. The docs' `fine_tuned_model_id` is silently ignored
    # (you get a plain base-model deployment), so we use the spec's name and verify it below.
    body = {"flavor_id": a.flavor, "deployment_name": ALIAS, "replicas": a.replicas,
            "fine_tuned_model": state()["checkpoint_id"]}
    cost = float(flavor["price_per_hour"]) * a.replicas
    print("Deployment payload:\n" + json.dumps(body, indent=2))
    print(f"\nCost: {flavor['price_per_hour']}/hr x {a.replicas} replica(s) = {cost:.2f}/hr, billed until you run `delete --yes`")
    if not a.yes:
        raise SystemExit("billable: re-run with --yes to create")
    r = c.post("/deployments", json=body)
    r.raise_for_status()
    print(r.json())
    save(deployment_id=r.json().get("id") or r.json().get("deployment", {}).get("id"), alias=ALIAS)


def cmd_status(a, c):
    dep_id = state()["deployment_id"]
    while True:
        dep = c.get(f"/deployments/{dep_id}").raise_for_status().json()
        status = str(dep.get("status") or dep.get("deployment", {}).get("status")).lower()  # API returns lowercase
        print(time.strftime("%H:%M:%S"), status, "| adapter:", dep.get("fine_tuned_model") or "NONE (base model only!)")
        if status in ("ready", "failed") or not a.watch:
            break
        time.sleep(60)


def cmd_smoke(a, c):
    from decision_model import decide, inference_client

    async def go():
        async with inference_client() as client:
            print(await decide(client, state()["alias"], "Manchester United won 3-1 at Old Trafford on Saturday."))
    asyncio.run(go())


def cmd_delete(a, c):
    if not a.yes:
        raise SystemExit("re-run with --yes to delete (billing stops only on deletion)")
    r = c.delete(f"/deployments/{state()['deployment_id']}")
    print(r.status_code, "deleting; GET returns 404 once gone")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("flavors"); s.add_argument("--base-model", default="Qwen3.5-2B", help="display name, e.g. Qwen/Qwen3.5-2B")
    s = sub.add_parser("create"); s.add_argument("--flavor", required=True); s.add_argument("--replicas", type=int, default=1); s.add_argument("--yes", action="store_true")
    s = sub.add_parser("status"); s.add_argument("--watch", action="store_true")
    sub.add_parser("smoke")
    s = sub.add_parser("delete"); s.add_argument("--yes", action="store_true")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a, api())
