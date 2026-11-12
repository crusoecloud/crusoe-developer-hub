import os, sys
from pathlib import Path
import httpx, trafilatura
from fastmcp import FastMCP
from search import search_adapter

WORK = Path(os.environ.get("BRIEF_WORKSPACE", "workspace")).resolve()
UA = "Mozilla/5.0 (compatible; brief-agent/0.1; +https://github.com/crusoecloud/crusoe-developer-hub)"
mcp = FastMCP("brief-tools")


@mcp.tool
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web. Returns title, url and snippet for each result. One focused query per call."""
    return search_adapter(query, max_results)


@mcp.tool
def fetch_url(url: str, max_chars: int = 20000) -> str:
    """Fetch a page and return its readable text, truncated to max_chars."""
    html = httpx.get(url, timeout=15, follow_redirects=True, headers={"User-Agent": UA}).text
    return (trafilatura.extract(html) or "")[:max_chars]


@mcp.tool
def write_file(path: str, content: str) -> str:
    """Write content to a file inside the workspace. Returns the absolute path."""
    target = (WORK / path).resolve()
    if not target.is_relative_to(WORK):
        raise ValueError("path must stay inside the workspace")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return str(target)


@mcp.tool
def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
    """Read lines start_line..end_line of a file in the workspace. Use a range; never read a whole large file."""
    target = (WORK / path).resolve()
    if not target.is_relative_to(WORK):
        raise ValueError("path must stay inside the workspace")
    lines = target.read_text().splitlines()
    return "\n".join(f"{i}: {l}" for i, l in enumerate(lines[start_line - 1:end_line], start=start_line))


if __name__ == "__main__":
    if "--http" in sys.argv:
        port = int(os.environ.get("MCP_PORT", "8000"))
        mcp.run(transport="http", host="127.0.0.1", port=port)
    else:
        mcp.run()
