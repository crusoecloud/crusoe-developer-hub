"""Run SGLang's serving benchmark inside the serving pod and collect the result.

``python3 -P -m sglang.bench_serving`` is an async HTTP load generator that ships in
the SGLang image. Running it *inside* the pod against ``127.0.0.1:8000`` keeps
your laptop's port-forward out of the measurement, and running it *detached* there
keeps your laptop's connection out of it too: a short ``kubectl exec`` starts it,
short reads check on it, and the result stays in the pod until it is read.

Reference: https://docs.sglang.io/developer_guide/benchmark_and_profiling.html
"""

from __future__ import annotations

import hashlib
import json
import shlex
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import k8s
from .manifest import MODEL_DIR, PORT

#: Fields we keep from the benchmark's JSON, in display order.
METRICS = (
    "completed",
    "duration",
    "request_throughput",
    "output_throughput",
    "total_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
)

LABELS = {
    "output_throughput": "Output throughput (tok/s)",
    "request_throughput": "Request throughput (req/s)",
    "total_throughput": "Total throughput (tok/s)",
    "median_ttft_ms": "Median TTFT (ms)",
    "p99_ttft_ms": "P99 TTFT (ms)",
    "median_tpot_ms": "Median TPOT (ms)",
    "p99_tpot_ms": "P99 TPOT (ms)",
    "median_itl_ms": "Median ITL (ms)",
    "median_e2e_latency_ms": "Median E2E latency (ms)",
}


@dataclass
class BenchResult:
    """One benchmark run, with the knobs that produced it."""

    label: str
    concurrency: int
    input_len: int
    output_len: int
    num_prompts: int
    dataset: str = "random-ids"
    seed: int = 42
    range_ratio: float | None = None  # bench_serving --random-range-ratio; None = its default
    cache_flushed: bool = False  # the server's prefix cache was emptied right before the run
    metrics: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item: str) -> float:
        # bench.output_throughput -> metrics["output_throughput"]
        metrics = self.__dict__.get("metrics", {})
        if item in metrics:
            return metrics[item]
        raise AttributeError(item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "concurrency": self.concurrency,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "num_prompts": self.num_prompts,
            "dataset": self.dataset,
            "seed": self.seed,
            "range_ratio": self.range_ratio,
            "cache_flushed": self.cache_flushed,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchResult:
        return cls(
            label=d["label"],
            concurrency=int(d["concurrency"]),
            input_len=int(d["input_len"]),
            output_len=int(d["output_len"]),
            num_prompts=int(d["num_prompts"]),
            dataset=d.get("dataset", "random-ids"),
            seed=int(d.get("seed", 42)),
            range_ratio=_float_or_none(d.get("range_ratio")),
            cache_flushed=bool(d.get("cache_flushed", False)),
            metrics={k: float(v) for k, v in d.get("metrics", {}).items()},
        )


def _float_or_none(v: Any) -> float | None:
    return None if v is None else float(v)


def bench_command(
    served_model_name: str,
    input_len: int = 1024,
    output_len: int = 256,
    num_prompts: int = 160,
    max_concurrency: int = 32,
    request_rate: str = "inf",
    dataset: str = "random-ids",
    output_file: str = "/tmp/bench.jsonl",
    seed: int = 42,
    range_ratio: float | None = None,
    extra: Iterable[str] = (),
) -> list[str]:
    """The ``bench_serving`` command line.

    ``random-ids`` builds prompts from random token ids on the fly, so it works
    inside a pod that has no Hub access (the ``random`` dataset downloads
    ShareGPT text to sample from; ``generated-shared-prefix`` is also local).

    ``request_rate="inf"`` with ``max_concurrency=N`` is a closed loop: exactly
    N requests in flight at all times, like N users who send the next prompt
    the moment the previous answer finishes. That is the shape that answers
    "how does latency change as more users arrive".

    ``range_ratio=1.0`` makes every prompt and answer exactly ``input_len`` and ``output_len``
    tokens; bench_serving's default draws each length at random below them.
    """
    ratio = [] if range_ratio is None else ["--random-range-ratio", str(range_ratio)]
    return [
        "python3",
        "-P",  # the ROCm image's working directory shadows the installed package
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--model",
        served_model_name,
        "--tokenizer",
        MODEL_DIR,
        "--dataset-name",
        dataset,
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        str(request_rate),
        "--max-concurrency",
        str(max_concurrency),
        "--seed",
        str(seed),
        "--output-file",
        output_file,
        "--disable-tqdm",
        *ratio,
        *extra,
    ]


def parse_result(
    json_line: str,
    label: str,
    concurrency: int,
    input_len: int,
    output_len: int,
    dataset: str = "random-ids",
    seed: int = 42,
    range_ratio: float | None = None,
) -> BenchResult:
    """Turn the benchmark's JSON line into a :class:`BenchResult`."""
    raw = json.loads(json_line)
    metrics = {k: float(raw[k]) for k in METRICS if k in raw and raw[k] is not None}
    return BenchResult(
        label=label,
        concurrency=concurrency,
        input_len=input_len,
        output_len=output_len,
        num_prompts=int(raw.get("completed", 0)),
        dataset=dataset,
        seed=seed,
        range_ratio=range_ratio,
        metrics=metrics,
        raw=raw,
    )


#: Reports the state of a detached run in the pod as one JSON line: "none" (never started here, or
#: the container restarted and /tmp is empty), "running", "done" (with the summary fields of the
#: result, a few hundred bytes instead of the ~40 KB line with every request's timings),
#: "failed" (with the end of the log) or "died" (the process is gone without an exit code). A zombie
#: counts as gone: the pod's PID 1 is the SGLang server, which never reaps an orphaned process, so a
#: killed run leaves its /proc entry behind.
_STATE_PY = r"""
import json, os, sys
b, keys = sys.argv[1], sys.argv[2:]
def out(**kw):
    print(json.dumps(kw)); sys.exit()
if not os.path.isdir(b + ".d"):
    out(state="none")
if os.path.exists(b + ".exit"):
    code = int((open(b + ".exit").read().strip() or "1"))
    if code == 0 and os.path.exists(b + ".jsonl"):
        r = json.loads(open(b + ".jsonl").read().strip().splitlines()[-1])
        out(state="done", summary={k: r.get(k) for k in keys})
    log = open(b + ".log").read()[-1500:] if os.path.exists(b + ".log") else ""
    out(state="failed", code=code, log=log)
def alive(pid):
    try:
        return open("/proc/" + pid + "/stat").read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except (OSError, IndexError):
        return False
pid = open(b + ".d/pid").read().strip() if os.path.exists(b + ".d/pid") else ""
out(state="running" if not pid or alive(pid) else "died")
"""

#: Empties the server's prefix (radix) cache. Without it, a second run on the same server finds
#: the same seed-42 prompts cached and measures cache hits, not serving: on one H200 with FP8,
#: 3,446 tok/s instead of 2,586. Every run starts with it, so every run starts from the same state.
#: SGLang only flushes when no request is running or waiting, so it is retried for up to a minute
#: (after a warm-up the server is idle within a second). If it never succeeds the run fails with a
#: clear message: a benchmark never runs on a warm cache without saying so.
_FLUSH_PY = f"""
import sys, time, urllib.error, urllib.request
msg = ""
for _ in range(20):
    try:
        req = urllib.request.Request("http://127.0.0.1:{PORT}/flush_cache", method="POST")
        msg = urllib.request.urlopen(req, timeout=60).read().decode()
        if msg.startswith("Cache flushed"):
            print(msg.splitlines()[0])
            sys.exit(0)
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode()
    except OSError as exc:
        msg = str(exc)
    time.sleep(3)
print("could not empty the server's prefix cache (" + " ".join(msg.split())[:200] + "); "
      "not benchmarking on a warm cache. Wait until nothing else is sending requests, then re-run the cell.")
sys.exit(1)
"""
FLUSH_CACHE = ["python3", "-c", _FLUSH_PY]

LOST = "the connection to the cluster was lost during the benchmark (did the laptop sleep?); re-run this cell"


class BenchmarkLost(RuntimeError):
    """The run disappeared or never finished: re-running the cell picks it up or starts it again."""


def _dot() -> None:
    print(".", end="", flush=True)


def run_id(pod: str, label: str, params: dict[str, Any]) -> str:
    """Same pod, same label, same parameters: the same run. A redeploy makes a new pod, so a new run."""
    # "v": runs started by an older version of this code (no cache flush) never match.
    key = json.dumps({"v": 2, "pod": pod, "label": label, **params}, sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _start_script(base: str, commands: list[list[str]]) -> str:
    """Start ``commands`` one after another in the background; return at once. The ``mkdir`` makes it
    safe to send twice (a retried start, or a re-run cell): the second one does nothing."""
    chain = " && ".join(shlex.join(c) for c in commands)
    inner = f"{chain}; echo $? > {base}.exit"
    return (
        f"mkdir {base}.d 2>/dev/null || exit 0; "
        f"nohup sh -c {shlex.quote(inner)} > {base}.log 2>&1 < /dev/null & "
        f"echo $! > {base}.d/pid"
    )


def _discard_script(base: str) -> str:
    """Move a failed or dead run's files aside (``*.prev-<time>``, the log kept for reference) so the
    same run id can start again: ``_start_script``'s ``mkdir`` would otherwise refuse forever."""
    files = " ".join(f"{base}{s}" for s in (".d", ".exit", ".log", ".jsonl", ".warmup.jsonl"))
    return f'ts=$(date +%s); for f in {files}; do [ -e "$f" ] && mv "$f" "$f.prev-$ts"; done; true'


def _why_failed(state: dict[str, Any]) -> str:
    if state["state"] == "died":
        return "its process stopped without a result"
    last = [ln for ln in str(state.get("log", "")).splitlines() if ln.strip()][-1:]
    return f"exit {state.get('code')}" + (f": {last[0].strip()[:160]}" if last else "")


def _state(name: str, namespace: str, base: str) -> dict[str, Any]:
    cmd = ["python3", "-c", _STATE_PY, base, "backend", "dataset_name", *METRICS]
    return json.loads(k8s.exec_in_pod(name, namespace, cmd, retry=True).strip())


def run_bench(
    name: str,
    namespace: str,
    served_model_name: str,
    label: str,
    input_len: int = 1024,
    output_len: int = 256,
    num_prompts: int = 160,
    max_concurrency: int = 32,
    request_rate: str = "inf",
    dataset: str = "random-ids",
    seed: int = 42,
    range_ratio: float | None = None,
    extra: Iterable[str] = (),
    warmup: dict[str, Any] | None = None,
    timeout_s: int = 600,
    poll_s: float = 5,
    fresh: bool = False,
    log=print,
    progress: Callable[[], None] | None = _dot,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> BenchResult:
    """Run the benchmark inside the serving pod, detached from your laptop, and return the result.

    One short ``kubectl exec`` starts ``bench_serving`` in the background in the pod, after emptying
    the server's prefix cache (:data:`FLUSH_CACHE`) and the ``warmup`` run, if given; short reads then check on it every ``poll_s`` seconds, printing a dot.
    Your laptop's connection only carries those reads, so Wi-Fi trouble or a sleeping laptop does not
    spoil the measurement: the reads are retried, and re-running the cell (same server, same label,
    same parameters) uses the finished result, or waits for the run still going, instead of starting
    another one. A previous run that failed or died is set aside and a new one starts.
    ``fresh=True`` measures again anyway.
    """
    params = dict(input_len=input_len, output_len=output_len, num_prompts=num_prompts,
                  max_concurrency=max_concurrency, request_rate=request_rate, dataset=dataset, seed=seed,
                  range_ratio=range_ratio, extra=list(extra), warmup=warmup)  # fmt: skip
    pods = k8s.workload_pods(name, namespace)
    if not pods:
        raise RuntimeError(f"no serving pod for {name} in {namespace}")
    pod = pods[0]["metadata"]["name"]
    rid = run_id(pod, label, params) + (f"-{int(time.time())}" if fresh else "")
    base = f"/tmp/bench-{rid}"
    what = f"concurrency={max_concurrency} prompts={num_prompts} {input_len}in/{output_len}out"
    t0 = clock()

    state = _state(name, namespace, base)
    if state["state"] in ("failed", "died"):
        k8s.exec_in_pod(name, namespace, ["sh", "-c", _discard_script(base)], retry=True)
        log(f"[{label}] previous benchmark failed ({_why_failed(state)}); starting a new one")
        state = {"state": "none"}
    if state["state"] == "done":
        log(f"[{label}] this run already finished in the pod; using its result (no new benchmark)")
    elif state["state"] == "running":
        log(f"[{label}] this run is still going in the pod; waiting for it")
    else:
        commands = [FLUSH_CACHE]
        if warmup:
            # Its own seed, so the warm-up's prompts are not the benchmark's first prompts; then flush again.
            w = {**warmup, "seed": warmup.get("seed", seed + 1)}
            if w["seed"] == seed:
                raise ValueError(f"the warm-up must use a different seed than the benchmark ({seed})")
            commands += [
                bench_command(
                    served_model_name, dataset="random-ids", output_file=f"{base}.warmup.jsonl", **w
                ),
                FLUSH_CACHE,
            ]
        commands.append(bench_command(served_model_name, input_len=input_len, output_len=output_len,
                                      num_prompts=num_prompts, max_concurrency=max_concurrency,
                                      request_rate=request_rate, dataset=dataset, output_file=f"{base}.jsonl",
                                      seed=seed, range_ratio=range_ratio, extra=extra))  # fmt: skip
        k8s.exec_in_pod(name, namespace, ["sh", "-c", _start_script(base, commands)], retry=True)
        log(f"[{label}] {'warm-up, then ' if warmup else ''}{what} ...")

    dots = 0
    while state["state"] != "done":
        if state["state"] == "failed":
            raise RuntimeError(
                f"[{label}] the benchmark failed in the pod (exit {state['code']}):\n{state['log']}"
            )
        if state["state"] == "died" or clock() - t0 > timeout_s:
            raise BenchmarkLost(LOST)
        sleep(poll_s)
        if progress:
            progress()
            dots += 1
        try:
            state = _state(name, namespace, base)
        except RuntimeError as exc:
            if "no serving pod" in str(exc):
                raise BenchmarkLost(LOST) from None  # the pod went away
            if not k8s.is_transient(exc):
                raise
            continue  # the network is still down: keep waiting until the time limit
        if state["state"] == "none":
            raise BenchmarkLost(LOST)  # the container restarted, /tmp is empty
    if dots and progress is _dot:
        print()

    line = json.dumps(state["summary"])
    res = parse_result(line, label, max_concurrency, input_len, output_len, dataset, seed, range_ratio)
    res.cache_flushed = True  # "done" means every step ran, the flush right before the benchmark included
    m = res.metrics
    log(
        f"[{label}] done in {clock() - t0:.0f}s: "
        f"{m.get('output_throughput', 0):.0f} tok/s out, "
        f"TTFT p50 {m.get('median_ttft_ms', 0):.0f} ms, TPOT p50 {m.get('median_tpot_ms', 0):.1f} ms"
    )
    return res


#: The one benchmark whose result goes on the leaderboard. Every participant runs exactly this,
#: so results are comparable: 384 prompts of exactly 4096 random token ids in and 512 out, 256
#: requests in flight, a fixed seed. 256 x 4608 tokens is about 1.18 million tokens of KV cache,
#: more than the bf16 pool of Qwen3-8B on one H200 (803,502) and less than the FP8 pool (1.6
#: million), so the benchmark shows what KV capacity is worth under load. About 2 minutes with
#: the warm-up; two baselines on one GPU agreed within 0.5%.
STANDARD: dict[str, Any] = {
    "dataset": "random-ids",
    "input_len": 4096,
    "output_len": 512,
    "num_prompts": 384,
    "max_concurrency": 256,
    "seed": 42,
    "range_ratio": 1.0,
}

#: Short batch run (and discarded) before the standard benchmark, so the first measured
#: requests do not pay for warming up the server.
WARMUP: dict[str, Any] = {
    "num_prompts": 32,
    "max_concurrency": 32,
    "input_len": 1024,
    "output_len": 32,
    "seed": 4242,
}


def benchmark_params(result: BenchResult) -> dict[str, Any]:
    """The parameters a result was measured with, in the shape of :data:`STANDARD`."""
    return {
        "dataset": result.dataset,
        "input_len": result.input_len,
        "output_len": result.output_len,
        "num_prompts": result.num_prompts,
        "max_concurrency": result.concurrency,
        "seed": result.seed,
        "range_ratio": result.range_ratio,
    }


def is_standard(result: BenchResult) -> bool:
    """True if ``result`` is the standard leaderboard benchmark with every request completed, measured
    on an emptied prefix cache."""
    return benchmark_params(result) == STANDARD and result.cache_flushed


def standard_bench(name: str, namespace: str, served_model_name: str, label: str, log=print) -> BenchResult:
    """Warm up, then run the standard leaderboard benchmark (:data:`STANDARD`, about 2 minutes),
    as one detached run in the pod (see :func:`run_bench`)."""
    return run_bench(name, namespace, served_model_name, label, warmup=WARMUP, log=log, **STANDARD)


def save_results(results: Iterable[BenchResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    return path


def load_results(path: str | Path) -> list[BenchResult]:
    return [BenchResult.from_dict(d) for d in json.loads(Path(path).read_text())]


def table(
    results: Iterable[BenchResult],
    keys: Iterable[str] = ("output_throughput", "median_ttft_ms", "median_tpot_ms"),
) -> str:
    """A Markdown table, one row per run."""
    keys = list(keys)
    head = "| run | concurrency | " + " | ".join(LABELS.get(k, k) for k in keys) + " |"
    sep = "|---|---:|" + "|".join("---:" for _ in keys) + "|"
    rows = [head, sep]
    for r in results:
        cells = [f"{r.metrics.get(k, float('nan')):.1f}" for k in keys]
        rows.append(f"| {r.label} | {r.concurrency} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _cell(v: float | None) -> str:
    return "–" if v is None else f"{v:.1f}"


def compare_table(
    baseline: BenchResult,
    results: Iterable[BenchResult],
    keys: Iterable[str] = ("output_throughput", "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms"),
) -> str:
    """A Markdown table of runs against the baseline: each value and its change in percent."""
    keys = list(keys)
    head = "| run | " + " | ".join(LABELS.get(k, k) for k in keys) + " |"
    rows = [head, "|---|" + "|".join("---:" for _ in keys) + "|"]
    base = baseline.metrics
    rows.append(f"| {baseline.label} | " + " | ".join(_cell(base.get(k)) for k in keys) + " |")
    for r in results:
        cells = []
        for k in keys:
            v, b = r.metrics.get(k), base.get(k)
            change = f" ({(v - b) / b * 100:+.0f}%)" if v is not None and b else ""
            cells.append(f"{_cell(v)}{change}")
        rows.append(f"| {r.label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
