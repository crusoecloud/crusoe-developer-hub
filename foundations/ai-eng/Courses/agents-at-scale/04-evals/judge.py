import asyncio, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "03-agent"))
from config import client
from schemas import Brief
from agent import RunResult

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "zai-org/GLM-5.3-Flash")
QUESTIONS = [
    "Is every sentence of the SUMMARY supported by at least one passage in the SOURCES below? Answer yes or no.",
    "Does the BRIEF make any claim that contradicts the SOURCES? Answer yes or no.",
]


async def judge(brief: Brief, fetched: dict[str, str]) -> dict[str, bool]:
    evidence = "\n\n".join(f"SOURCE {u}\n{text[:6000]}" for u, text in fetched.items())
    answers = {}
    for q in QUESTIONS:
        resp = await client.chat.completions.create(model=JUDGE_MODEL, temperature=0, messages=[
            {"role": "system", "content": "You are a strict fact checker. Answer with exactly one word: yes or no."},
            {"role": "user", "content": f"{q}\n\nBRIEF\n{brief.model_dump_json()}\n\n{evidence}"}])
        answers[q] = resp.choices[0].message.content.strip().lower().startswith("yes")
    return answers


async def judge_run(run_dir: Path) -> dict[str, bool]:
    run = RunResult.load(run_dir)
    if run.brief is None:
        raise SystemExit(f"{run_dir} has no brief")
    return await judge(run.brief, run.fetched())


async def calibrate(labels_path: Path) -> float:
    labels = [json.loads(l) for l in Path(labels_path).read_text().splitlines() if l.strip()]
    agree = 0
    for label in labels:
        answers = await judge_run(Path(label["run_dir"]))
        faithful = answers[QUESTIONS[0]] and not answers[QUESTIONS[1]]
        hit = faithful == label["faithful"]
        agree += hit
        print(f"{'agree ' if hit else 'DIFFER'} {label['run_dir']}: judge={faithful} human={label['faithful']} ({label.get('reason', '')})")
    rate = agree / max(len(labels), 1)
    print(f"agreement: {agree}/{len(labels)} = {rate:.2f}")
    return rate


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--calibrate":
        rate = asyncio.run(calibrate(Path(sys.argv[2])))
        sys.exit(0 if rate >= float(os.environ.get("JUDGE_MIN_AGREEMENT", "0.8")) else 1)
    for q, a in asyncio.run(judge_run(Path(sys.argv[1]))).items():
        print(f"{'yes' if a else 'no ':3}  {q}")
