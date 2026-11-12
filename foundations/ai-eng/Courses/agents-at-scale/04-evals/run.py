import argparse, asyncio, json, sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "03-agent"))
from config import settings
from shared import SLOTS
from agent import run_brief_traced
from cost import cost_usd
from trace import Usage
from graders import GRADERS

RUNS = Path("runs") / "evals"
RESULTS = Path(__file__).with_name("results")


@dataclass
class Outcome:
    ok: bool
    failed: list[str]
    usage: Usage
    seconds: float
    tool_calls: int


@dataclass
class TaskResult:
    task_id: str
    model: str
    pass_k: bool
    outcomes: list[Outcome] = field(default_factory=list)


class BudgetExhausted(Exception):
    pass


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


async def eval_task(task, model, k=3, spent: dict | None = None, max_usd: float | None = None) -> TaskResult:
    outcomes = []
    for i in range(k):
        async with SLOTS:
            run = await run_brief_traced(task["topic"], model=model,
                                         workdir=RUNS / task["id"] / model.replace("/", "_") / str(i))
        failed = [g.__name__ for g in GRADERS if not g(task, run)]
        outcomes.append(Outcome(ok=not failed, failed=failed, usage=run.usage,
                                seconds=run.seconds, tool_calls=len(run.tool_calls)))
        if spent is not None:
            spent["usd"] += run.cost_usd
            if max_usd is not None and spent["usd"] > max_usd:
                raise BudgetExhausted(spent["usd"])
    return TaskResult(task_id=task["id"], model=model, pass_k=all(o.ok for o in outcomes), outcomes=outcomes)


def report_row(results: list[TaskResult], model: str) -> dict:
    rs = [r for r in results if r.model == model]
    runs = [o for r in rs for o in r.outcomes]
    passing = [o for o in runs if o.ok]
    secs = sorted(o.seconds for o in runs)
    return {"model": model,
            "pass^k": sum(r.pass_k for r in rs) / max(len(rs), 1),
            "pass_rate": len(passing) / max(len(runs), 1),
            "calls": mean(o.tool_calls for o in runs) if runs else 0,
            "p50_s": secs[len(secs) // 2] if secs else 0,
            "p95_s": secs[min(int(len(secs) * 0.95), len(secs) - 1)] if secs else 0,
            "usd_per_pass": sum(cost_usd(o.usage, model) for o in runs) / max(len(passing), 1)}


def failures_by_grader(results: list[TaskResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        for o in r.outcomes:
            for g in o.failed:
                counts[g] = counts.get(g, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def write_report(results: list[TaskResult], path: Path, k: int) -> list[dict]:
    rows = [report_row(results, m) for m in dict.fromkeys(r.model for r in results)]
    lines = [f"# eval {date.today()} (k={k})", "",
             f"| model | pass^{k} | pass rate | calls | p50 s | p95 s | $ per pass |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['pass^k']:.2f} | {r['pass_rate']:.2f} | {r['calls']:.1f} | "
                     f"{r['p50_s']:.1f} | {r['p95_s']:.1f} | {r['usd_per_pass']:.4f} |")
    lines += ["", "## failures by grader", ""] + [f"- {g}: {n}" for g, n in failures_by_grader(results).items()]
    lines += ["", "## per task", ""]
    for r in results:
        lines.append(f"- {r.task_id} {r.model}: {'pass' if r.pass_k else 'FAIL'} "
                     + ", ".join("ok" if o.ok else "+".join(o.failed) for o in r.outcomes))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(json.dumps({"date": str(date.today()), "k": k, "rows": rows}, indent=2))
    print("\n".join(lines))
    return rows


async def main(args) -> int:
    tasks = load_jsonl(args.tasks)
    models = args.models.split(",") if args.models else [settings.model]
    spent = {"usd": 0.0}
    results: list[TaskResult] = []
    try:
        for m in models:
            for t in tasks:
                results.append(await eval_task(t, m, k=args.k, spent=spent, max_usd=args.max_usd))
    except BudgetExhausted as e:
        print(f"stopped: budget exhausted at ${e.args[0]:.2f}", file=sys.stderr)
    rows = write_report(results, Path(args.out) if args.out else RESULTS / f"{date.today()}.md", args.k)
    if args.baseline and rows:
        baseline = json.loads(Path(args.baseline).read_text())
        base_rate = baseline["rows"][0]["pass_rate"] if "rows" in baseline else baseline["pass_rate"]
        drop = (base_rate - rows[0]["pass_rate"]) * 100
        print(f"pass rate {rows[0]['pass_rate']:.2f} vs baseline {base_rate:.2f} (drop {drop:.1f} points)")
        if drop > 5:
            return 1
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=str(Path(__file__).with_name("tasks.jsonl")))
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--models", default="")
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--baseline", default=None)
    p.add_argument("--out", default=None)
    sys.exit(asyncio.run(main(p.parse_args())))
