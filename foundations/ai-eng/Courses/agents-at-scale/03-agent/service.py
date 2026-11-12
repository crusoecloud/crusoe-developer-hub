import asyncio, json, shutil, sys
from pathlib import Path
from uuid import uuid4
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
from config import settings
from shared import SLOTS
from agent import run_brief_traced

WORK = Path("workspace").resolve()
RESULTS = Path(settings.results_dir).resolve()
PENDING: set[str] = set()
app = FastAPI()


class BriefRequest(BaseModel):
    topic: str


def new_run_id() -> str:
    return f"svc_{uuid4().hex[:8]}"


async def upload(workdir: Path, bucket: Path, prefix: str) -> None:
    dest = bucket / prefix
    dest.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(shutil.copytree, workdir, dest, dirs_exist_ok=True)


async def run_and_store(run_id: str, topic: str) -> None:
    workdir = WORK / run_id
    try:
        async with SLOTS:
            await run_brief_traced(topic, model=settings.model, workdir=workdir)
        await upload(workdir, bucket=RESULTS, prefix=run_id)
    finally:
        PENDING.discard(run_id)
        shutil.rmtree(workdir, ignore_errors=True)


async def read_status(run_id: str) -> dict:
    meta_path = RESULTS / run_id / "run.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("error"):
            return {"run_id": run_id, "status": "failed", "error": meta["error"]}
        brief = json.loads((RESULTS / run_id / "brief.json").read_text())
        return {"run_id": run_id, "status": "done", "brief": brief, "usage": meta["usage"], "seconds": meta["seconds"]}
    if run_id in PENDING:
        return {"run_id": run_id, "status": "pending"}
    raise HTTPException(status_code=404, detail="unknown run id")


@app.post("/brief", status_code=202)
async def submit(req: BriefRequest, bg: BackgroundTasks) -> dict:
    run_id = new_run_id()
    PENDING.add(run_id)
    bg.add_task(run_and_store, run_id, req.topic)
    return {"run_id": run_id, "status_url": f"/brief/{run_id}"}


@app.get("/brief/{run_id}")
async def status(run_id: str) -> dict:
    return await read_status(run_id)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "model": settings.model}
