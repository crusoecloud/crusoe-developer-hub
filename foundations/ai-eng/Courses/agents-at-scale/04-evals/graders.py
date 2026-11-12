import re
import httpx

_resolved: dict[str, bool] = {}


PUNCT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " "})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(PUNCT)).strip().lower()


def _resolves(url: str) -> bool:
    if url in _resolved:
        return _resolved[url]
    try:
        r = httpx.head(url, follow_redirects=True, timeout=10)
        if r.status_code >= 400:
            r = httpx.get(url, follow_redirects=True, timeout=10, headers={"Range": "bytes=0-1024"})
        ok = r.status_code < 400
    except httpx.HTTPError:
        ok = False
    _resolved[url] = ok
    return ok


def grounding_problems(brief, fetched: dict[str, str]) -> list[str]:
    pages = {u: _norm(t) for u, t in fetched.items()}
    problems = []
    for i, s in enumerate(brief.sources, start=1):
        url = str(s.url)
        if url not in pages:
            problems.append(f"source {i}: {url} was not fetched in this run")
        elif _norm(s.quote) not in pages[url]:
            problems.append(f"source {i}: quote not found on page")
    return problems


def file_exists(task, run):
    return run.brief_path is not None and run.brief_path.exists()


def min_sources(task, run):
    return run.brief is not None and len(run.brief.sources) >= 3


def urls_resolve(task, run):
    return run.brief is not None and all(_resolves(str(s.url)) for s in run.brief.sources)


def quotes_grounded(task, run):
    return run.brief is not None and not grounding_problems(run.brief, run.fetched())


def facts_present(task, run):
    if run.brief_path is None or not run.brief_path.exists():
        return False
    text = _norm(run.brief_path.read_text())
    return all(_norm(f) in text for f in task.get("must_mention", []))


def trajectory_ok(task, run):
    names = [c.name for c in run.tool_calls]
    sigs = [(c.name, c.arguments) for c in run.tool_calls]
    return ("web_search" in names and "write_file" in names
            and names.index("web_search") < names.index("write_file")
            and len(names) <= 25
            and len(sigs) == len(set(sigs)))


GRADERS = [file_exists, min_sources, urls_resolve, quotes_grounded, facts_present, trajectory_ok]
