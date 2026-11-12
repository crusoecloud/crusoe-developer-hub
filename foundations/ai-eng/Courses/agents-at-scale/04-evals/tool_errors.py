import json, sys
from pathlib import Path
import pandas as pd


def read_traces(pattern: str) -> pd.DataFrame:
    rows = []
    for path in Path(".").glob(pattern):
        model = None
        for line in path.read_text().splitlines():
            obj = json.loads(line)
            model = obj.get("model") or model
            if "tool" in obj:
                rows.append({"run_id": obj["run_id"], "model": model, "tool_name": obj["tool"]["name"],
                             "tool_result": obj["tool"]["result"]})
    return pd.DataFrame(rows, columns=["run_id", "model", "tool_name", "tool_result"])


if __name__ == "__main__":
    df = read_traces(sys.argv[1] if len(sys.argv) > 1 else "runs/**/trace.jsonl")
    errors = df[df.tool_result.str.startswith("error:")]
    print(f"{len(df)} tool calls, {len(errors)} errors")
    if len(errors):
        print(errors.groupby(["model", "tool_name"]).size().sort_values(ascending=False))
