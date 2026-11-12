import json, time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    def add(self, u) -> None:
        if u is None:
            return
        self.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(u, "completion_tokens", 0) or 0
        details = getattr(u, "prompt_tokens_details", None)
        self.cached_tokens += (getattr(details, "cached_tokens", None) or 0)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolRecord:
    id: str
    name: str
    arguments: str
    args: dict
    result: str


class Trace:
    def __init__(self, run_id: str, path: Path):
        self.run_id, self.path, self.t0 = run_id, Path(path), time.time()
        self.f = self.path.open("a")
        self.usage = Usage()
        self.steps = 0
        self.tool_calls: list[ToolRecord] = []

    def _write(self, obj: dict) -> None:
        self.f.write(json.dumps({"run_id": self.run_id, "ts": round(time.time() - self.t0, 3), **obj}) + "\n")
        self.f.flush()

    def record(self, step: int, msg, usage, ms: int | None = None, model: str | None = None) -> None:
        self.usage.add(usage)
        self.steps = step + 1
        self._write({"step": step, "message": msg.model_dump(exclude_none=True),
                     "usage": usage.model_dump() if usage is not None else None, "ms": ms, "model": model})

    def tool_result(self, call, args: dict, result: str) -> None:
        rec = ToolRecord(call.id, call.function.name, call.function.arguments or "{}", args, result)
        self.tool_calls.append(rec)
        self._write({"tool": {"id": rec.id, "name": rec.name, "arguments": rec.arguments, "result": result}})

    def note(self, event: str, **fields) -> None:
        self._write({"event": event, **fields})

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def close(self) -> None:
        self.f.close()
