#!/usr/bin/env python3
"""Serverless Fine-Tuning on Crusoe: upload -> price estimate -> job -> checkpoints.

Follows the crusoe-sft skill. Nothing billable happens without --yes.

    export CRUSOE_API_KEY='...'                      # console.crusoecloud.com/security/inference-api-keys
    python 02_finetune.py models                # list base models; pick the exact id
    python 02_finetune.py estimate --model <id> # uploads data (free), prints price estimate + payload
    python 02_finetune.py create   --model <id> --yes
    python 02_finetune.py status                # poll until succeeded
    python 02_finetune.py checkpoints           # pick best by validation loss, saved to run_state.json

    pip install openai httpx
"""
import argparse
import json
import time
from pathlib import Path

import httpx
from openai import OpenAI

from decision_model import TRUST_ENV, token

HERE = Path(__file__).parent
STATE = HERE / "run_state.json"
HPARAMS = {  # skill defaults for a <=9B model; checkpoint_steps lowered so we get several to compare
    "n_epochs": 2, "batch_size": 64, "learning_rate": 1e-4,
    "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.0,
    "checkpoint_steps": 50, "eval_steps_per_epoch": 4,
}
SUFFIX = "agnews-decider"


def client() -> OpenAI:
    return OpenAI(api_key=token(), base_url="https://api.intelligence.crusoecloud.com/v1",
                  http_client=httpx.Client(proxy=None, trust_env=TRUST_ENV))


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kv) -> dict:
    state = load_state() | kv
    STATE.write_text(json.dumps(state, indent=2))
    return state


def upload(c: OpenAI, path: Path) -> str:
    """Upload once; reuse an existing file with the same name (never re-upload)."""
    for f in c.files.list().data:
        if f.filename == path.name and f.bytes == path.stat().st_size and f.purpose == "fine-tune":
            print(f"reusing {path.name} -> {f.id}")
            return f.id
    with path.open("rb") as fh:
        f = c.files.create(file=fh, purpose="fine-tune")
    print(f"uploaded {path.name} -> {f.id}")
    return f.id


def payload(model: str, train_id: str, val_id: str) -> dict:
    return {"model": model, "training_file": train_id, "validation_file": val_id, "suffix": SUFFIX,
            "method": {"type": "supervised", "supervised": {"hyperparameters": HPARAMS}}}


def cmd_models(args, c):
    models = httpx.get("https://api.intelligence.crusoecloud.com/v1/public/models",
                       headers={"Authorization": f"Bearer {token()}"}, trust_env=TRUST_ENV).json()["data"]
    for m in models:
        print(m["id"])


def cmd_estimate(args, c):
    train_id, val_id = upload(c, HERE / "data/train.jsonl"), upload(c, HERE / "data/val.jsonl")
    body = payload(args.model, train_id, val_id)
    est = c.post("/fine_tuning/jobs/estimate_price", cast_to=object,
                 body={k: body[k] for k in ("model", "method")} | {"training_file": train_id})
    save_state(model=args.model, train_file=train_id, val_file=val_id)
    print("\nJob payload:\n" + json.dumps(body, indent=2))
    print("\nPrice estimate (file-size based, not a quote):", est)
    print("\nRe-run with `create --yes` to launch.")


def cmd_create(args, c):
    state = load_state()
    if not (state.get("train_file") and state.get("model") == args.model):
        raise SystemExit("run `estimate` first (same --model) so you've seen the payload and price")
    if not args.yes:
        raise SystemExit("billable: re-run with --yes after reviewing the estimate")
    job = c.fine_tuning.jobs.create(**payload(args.model, state["train_file"], state["val_file"]))
    save_state(job_id=job.id)
    print("created", job.id, job.status)


def cmd_status(args, c):
    job_id = load_state()["job_id"]
    while True:
        job = c.fine_tuning.jobs.retrieve(job_id)
        print(time.strftime("%H:%M:%S"), job.status)
        if job.status in {"succeeded", "failed", "cancelled"} or not args.watch:
            break
        time.sleep(60)
    if job.status == "failed":
        for ev in c.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id).data:
            if ev.level in ("warn", "error"):
                print(ev.level, ev.message)
    if job.status == "succeeded":
        print("trained_tokens (billed):", job.trained_tokens)


def cmd_checkpoints(args, c):
    job_id = load_state()["job_id"]
    ckpts = c.fine_tuning.jobs.checkpoints.list(fine_tuning_job_id=job_id).data
    print(f"{'step':>5} {'train_loss':>10} {'valid_loss':>10}  fine_tuned_model_id")
    for k in sorted(ckpts, key=lambda k: k.step_number):
        print(f"{k.step_number:>5} {k.metrics.train_loss:>10.4f} {k.metrics.valid_loss or float('nan'):>10.4f}  {k.fine_tuned_model_id}")
    scored = [k for k in ckpts if k.metrics.valid_loss is not None]
    best = min(scored, key=lambda k: k.metrics.valid_loss)  # validation loss, not training loss
    save_state(checkpoint_id=best.fine_tuned_model_id, checkpoint_name=best.fine_tuned_model_checkpoint,
               checkpoint_step=best.step_number,
               checkpoint_metrics=[{"step": k.step_number, "train_loss": k.metrics.train_loss,
                                    "valid_loss": k.metrics.valid_loss} for k in sorted(ckpts, key=lambda k: k.step_number)])
    print(f"\nbest by validation loss: step {best.step_number} -> {best.fine_tuned_model_id} (saved to run_state.json)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("models", "estimate", "create", "status", "checkpoints"):
        sp = sub.add_parser(name)
        if name in ("estimate", "create"):
            sp.add_argument("--model", required=True, help="exact base model id from `models` (no silent substitution)")
        if name == "create":
            sp.add_argument("--yes", action="store_true")
        if name == "status":
            sp.add_argument("--watch", action="store_true", help="poll every 60s until terminal")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a, client())
