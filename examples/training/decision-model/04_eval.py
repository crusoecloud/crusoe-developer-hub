#!/usr/bin/env python3
"""Evaluate the deployment *as a decision model*, not just a classifier.

Beyond accuracy it reports what makes decision models useful in production:
calibration (does 0.9 confidence mean 90% right?), abstention (accuracy when you
only act on confident answers), and per-call latency. Writes results.json, which
is where every number in BLOG.md comes from.

    export CRUSOE_API_KEY='...'
    python 04_eval.py --model agnews-decider
    python 04_eval.py --model <other-model> --out base_results.json    # zero-shot baseline
"""
import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

from decision_model import LABELS, decide, inference_client

DATA = Path(__file__).parent / "data"
THRESHOLDS = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]


def load(name: str, limit: int | None):
    rows = [json.loads(l) for l in (DATA / f"{name}.jsonl").read_text().splitlines()]
    return [(r["messages"][1]["content"], r["messages"][2]["content"]) for r in rows][:limit]


async def run(model: str, rows, concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    async with inference_client() as client:
        async def one(text, gold):
            async with sem:
                t = time.perf_counter()
                out = await decide(client, model, text)
                return out | {"gold": gold, "ms": (time.perf_counter() - t) * 1000}
        return await asyncio.gather(*(one(t, g) for t, g in rows))


def nll(results, temperature):
    total = 0.0
    for r in results:
        logits = {l: r["debug"]["logprobs"].get(l, -30.0) / temperature for l in LABELS}
        peak = max(logits.values())
        log_z = peak + math.log(sum(math.exp(v - peak) for v in logits.values()))
        total -= logits[r["gold"]] - log_z
    return total / len(results)


def fit_temperature(results) -> float:
    """Single scalar fit on held-out data: the cheapest, most robust calibration."""
    grid = [x / 20 for x in range(10, 121)]  # 0.5 .. 6.0
    return min(grid, key=lambda t: nll(results, t))


def reprobs(r, temperature):
    logits = {l: r["debug"]["logprobs"].get(l, -30.0) / temperature for l in LABELS}
    peak = max(logits.values())
    w = {l: math.exp(v - peak) for l, v in logits.items()}
    z = sum(w.values())
    probs = {l: w[l] / z for l in LABELS}
    choice = max(probs, key=probs.get)
    return choice, probs[choice]


def ece(preds, bins=10):
    """Expected calibration error: gap between confidence and accuracy, averaged over confidence bins."""
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [(c, ok) for c, ok in preds if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if members:
            total += len(members) / len(preds) * abs(statistics.mean(c for c, _ in members) - statistics.mean(ok for _, ok in members))
    return total


def reliability(preds, bins=10):
    """Per confidence bin: how sure the model said it was vs how often it was right (for the chart)."""
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [(c, ok) for c, ok in preds if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if members:
            out.append({"confidence": statistics.mean(c for c, _ in members),
                        "accuracy": statistics.mean(ok for _, ok in members), "n": len(members)})
    return out


def summarize(results, temperature):
    preds = []
    for r in results:
        choice, conf = reprobs(r, temperature)
        preds.append((conf, choice == r["gold"], choice, r["gold"]))
    n = len(preds)
    lat = sorted(r["ms"] for r in results)
    abstain = []
    for t in THRESHOLDS:
        kept = [p for p in preds if p[0] >= t]
        abstain.append({"threshold": t, "coverage": len(kept) / n,
                        "accuracy": statistics.mean(p[1] for p in kept) if kept else None})
    return {
        "temperature": temperature, "n": n,
        "accuracy": statistics.mean(p[1] for p in preds),
        "mean_confidence": statistics.mean(p[0] for p in preds),
        "ece": ece([(p[0], p[1]) for p in preds]),
        "reliability": reliability([(p[0], p[1]) for p in preds]),
        "abstention": abstain,
        "per_class_accuracy": {l: statistics.mean(p[1] for p in preds if p[3] == l) for l in LABELS},
        "latency_ms": {"p50": lat[n // 2], "p95": lat[int(n * 0.95)], "mean": statistics.mean(lat)},
        "mean_unassigned_mass": statistics.mean(r["debug"]["unassigned"] for r in results),
    }


def plain_summary(out, threshold=0.9):
    cal = out["calibrated"]
    row = next((r for r in cal["abstention"] if r["threshold"] == threshold and r["accuracy"] is not None), None)
    lines = [f"In plain terms ({cal['n']:,} stories the model never saw):",
             f"  - It sorted {cal['accuracy']:.1%} of them correctly."]
    if row:
        lines.append(f"  - Acting only on answers it is at least {threshold:.0%} sure of covers {row['coverage']:.1%} of "
                     f"stories, and {row['accuracy']:.1%} of those are right. Send the rest to a person.")
    lines.append(f"  - Its confidence can be trusted: on average it is off by {cal['ece'] * 100:.1f} points "
                 f"(ECE {cal['ece']:.3f}, lower is better).")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="deployment alias")
    p.add_argument("--limit", type=int, help="use only the first N rows of val/test (cheap smoke run)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", default="results.json")
    a = p.parse_args()

    val = asyncio.run(run(a.model, load("val", a.limit), a.concurrency))
    test = asyncio.run(run(a.model, load("test", a.limit), a.concurrency))
    temperature = fit_temperature(val)  # fit on validation, report on test: no leakage
    out = {"model": a.model, "raw": summarize(test, 1.0), "calibrated": summarize(test, temperature)}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(plain_summary(out))
    print("\nDetails:")
    for name in ("raw", "calibrated"):
        s = out[name]
        print(f"{name:>10}: acc {s['accuracy']:.3f}  ECE {s['ece']:.3f}  mean conf {s['mean_confidence']:.3f}  (T={s['temperature']})")
    print("latency ms:", {k: round(v) for k, v in out["raw"]["latency_ms"].items()})
    print("abstention (calibrated):")
    for r in out["calibrated"]["abstention"]:
        acc = "  n/a" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
        print(f"   confidence >= {r['threshold']:<4} covers {r['coverage']:.1%} of traffic at {acc} accuracy")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
