import asyncio, json, os, re, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
import httpx, structlog, trafilatura

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "03-agent"))
sys.path.insert(0, str(ROOT / "04-evals"))
sys.path.insert(0, str(ROOT / "05-multi-agent"))
from agent import slugify
from graders import _resolves, grounding_problems
from team import TeamResult, Verdict, build_team, dump, run_team_traced

STATE = Path(os.environ.get("BRIEF_STATE", "state/STATE.json"))
TOPICS = Path(os.environ.get("BRIEF_TOPICS", str(Path(__file__).with_name("topics.txt"))))
DRY_RUN = os.environ.get("BRIEF_DRY_RUN", "1") != "0"
REPO = os.environ.get("BRIEF_REPO", "")
UA = "Mozilla/5.0 (compatible; brief-agent/0.1; +https://github.com/crusoecloud/crusoe-developer-hub)"
log = structlog.get_logger()


@dataclass
class Budget:
    max_attempts: int = 3
    max_usd: float = 2.00
    max_minutes: int = 40


class BudgetExceeded(Exception):
    def __init__(self, spent: float):
        super().__init__(f"budget exceeded: ${spent:.2f}")
        self.spent = spent


def sh(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"spent_usd": 0.0, "topics": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)


def load_topics(path: Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]


def fetch_text(url: str) -> str:
    try:
        html = httpx.get(url, timeout=15, follow_redirects=True, headers={"User-Agent": UA}).text
    except httpx.HTTPError:
        return ""
    return trafilatura.extract(html) or ""


def problem_key(problem: str) -> str:
    m = re.search(r"source\s+(\d+)", problem, flags=re.I)
    kind = "not fetched" if "not fetched" in problem.lower() else "quote" if "quote" in problem.lower() else "other"
    return f"source {m.group(1)}: {kind}" if m else re.sub(r"\W+", " ", problem.lower()).strip()[:80]


async def verify(result: TeamResult) -> Verdict:
    if result.error:
        return Verdict(passed=False, problems=[f"run failed: {result.error}"])
    if result.brief is None:
        return Verdict(passed=False, problems=result.problems or ["writer and verifier did not converge"])
    problems = []
    if result.brief_path is None or not result.brief_path.exists():
        problems.append("brief file missing")
    if len(result.brief.sources) < 3:
        problems.append("fewer than three sources")
    for i, s in enumerate(result.brief.sources, start=1):
        if not _resolves(str(s.url)):
            problems.append(f"source {i}: url does not resolve")
    fetched = {str(s.url): fetch_text(str(s.url)) for s in result.brief.sources}
    problems += grounding_problems(result.brief, fetched)
    if problems:
        return Verdict(passed=False, problems=problems)
    verifier = build_team(result.workdir)["verifier"]
    async with verifier:
        verdict = (await verifier.run(f"Brief:\n{dump(result.brief)}\nFindings: see the sources in the brief.")).output
    return verdict


def publish(result: TeamResult) -> None:
    if DRY_RUN:
        return log.info("dry-run: would publish", path=str(result.brief_path))
    sh("git", "checkout", "-B", "results")
    sh("git", "add", "-f", str(result.brief_path))
    sh("git", "commit", "-m", f"brief: {result.brief.title}")
    sh("git", "push", "origin", "results")


def escalate(topic: str, problems: list[str], trace_path: Path | None) -> None:
    body = f"Topic: {topic}\n\nProblems:\n" + "\n".join(f"- {p}" for p in problems) + f"\n\nTrace: {trace_path}"
    if DRY_RUN:
        return log.info("dry-run: would open issue", title=topic, problems=problems)
    sh("gh", "issue", "create", *(["--repo", REPO] if REPO else []),
       "--title", f"[brief] needs a human: {topic}", "--body", body, "--label", "needs-human")


def iterate(topic: str, state: dict, budget: Budget, started: float) -> None:
    rec = state["topics"].setdefault(topic, {"attempts": 0, "status": "pending", "problems": []})
    if rec["status"] in ("done", "escalated") or rec["attempts"] >= budget.max_attempts:
        return
    handoff = (f"Topic: {topic}\nPrevious attempts: {rec['attempts']}\n"
               f"Problems to fix from last attempt: {rec['problems'] or 'none'}")
    workdir = Path("runs") / "loop" / slugify(topic) / str(rec["attempts"])
    result = asyncio.run(run_team_traced(handoff, workdir))
    rec["attempts"] += 1
    state["spent_usd"] += result.cost_usd
    verdict = asyncio.run(verify(result))
    log.info("iteration", topic=topic, attempt=rec["attempts"], passed=verdict.passed, problems=verdict.problems)
    if verdict.passed:
        rec["status"] = "done"
        publish(result)
    elif sorted(map(problem_key, verdict.problems)) == sorted(map(problem_key, rec["problems"])):
        rec["status"] = "escalated"
        escalate(topic, verdict.problems, result.trace_path)
    else:
        rec["problems"] = verdict.problems
    if rec["attempts"] >= budget.max_attempts and rec["status"] == "pending":
        rec["status"] = "escalated"
        escalate(topic, verdict.problems, result.trace_path)
    if state["spent_usd"] > budget.max_usd or (time.time() - started) / 60 > budget.max_minutes:
        raise BudgetExceeded(state["spent_usd"])


if __name__ == "__main__":
    if Path("STOP").exists():
        sys.exit("kill switch present")
    state = load_state(STATE)
    started = time.time()
    try:
        for topic in load_topics(TOPICS):
            iterate(topic, state, Budget(), started)
            save_state(STATE, state)
    except BudgetExceeded as e:
        save_state(STATE, state)
        escalate("loop", [str(e)], None)
    print(json.dumps(state, indent=1))
