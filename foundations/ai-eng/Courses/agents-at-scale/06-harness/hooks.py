import os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "04-evals"))
from graders import _norm

WORK = Path(os.environ.get("BRIEF_WORKSPACE", "workspace")).resolve()


class Blocked(Exception):
    pass


def parse_sources(markdown: str) -> list[tuple[str, str]]:
    return re.findall(r"^- (\S+)\n  > (.+)$", markdown, flags=re.M)


def grounding_problems(path: Path, fetched: dict[str, str]) -> list[str]:
    pages = {u: _norm(t) for u, t in fetched.items()}
    problems = []
    for i, (url, quote) in enumerate(parse_sources(Path(path).read_text()), start=1):
        if url not in pages:
            problems.append(f"source {i}: {url} was not fetched in this run")
        elif _norm(quote) not in pages[url]:
            problems.append(f"source {i}: quote not found on page")
    return problems


def before_tool(name: str, args: dict, work: Path = WORK) -> None:
    if name == "write_file" and not (work / args["path"]).resolve().is_relative_to(work):
        raise Blocked(f"write outside workspace: {args['path']}")


def after_tool(name: str, args: dict, result: str, fetched: dict[str, str], work: Path = WORK) -> str | None:
    if name == "write_file":
        problems = grounding_problems(work / args["path"], fetched)
        if problems:
            return "VERIFY FAILED, fix before finishing:\n- " + "\n- ".join(problems)
    return None
