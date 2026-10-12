"""Small HTTP API for running simulations.

    uvicorn service.app:app --port 8000

    curl -X POST localhost:8000/jobs -H 'content-type: application/json' \
         -d '{"n": 1024, "steps": 5000, "init": "hotspot"}'
    curl localhost:8000/jobs/<id>
    open http://localhost:8000/jobs/<id>/image
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field, field_validator

from heatsim.grid import INITIAL_CONDITIONS
from heatsim.solvers import available_backends
from service import settings
from service.jobs import JobManager

BACKENDS = available_backends()
manager = JobManager()


@asynccontextmanager
async def lifespan(_app):
    os.makedirs(manager.results_dir, exist_ok=True)
    manager.start()
    yield
    manager.stop()


app = FastAPI(title="heatsim", version="0.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class JobRequest(BaseModel):
    n: int = Field(1024, ge=16, le=settings.MAX_N)
    steps: int = Field(1000, ge=1, le=settings.MAX_STEPS)
    r: float = Field(0.24, gt=0, le=0.25)
    init: Literal[INITIAL_CONDITIONS] = "hotspot"
    backend: str | None = None

    @field_validator("backend")
    @classmethod
    def check_backend(cls, v):
        if v is not None and v not in BACKENDS:
            raise ValueError(f"backend '{v}' not available here, options: {BACKENDS}")
        return v


def _default_backend():
    if settings.DEFAULT_BACKEND in BACKENDS:
        return settings.DEFAULT_BACKEND
    # e.g. inside the linux container there's no metal, fall back to the fastest CPU one
    return "numba" if "numba" in BACKENDS else "numpy"


def _job_or_404(job_id):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/backends")
def backends():
    return {"available": BACKENDS, "default": _default_backend()}


@app.post("/jobs", status_code=202)
def submit(req: JobRequest):
    params = req.model_dump()
    params["backend"] = params["backend"] or _default_backend()
    job = manager.submit(params)
    return {"id": job.id, "status": job.status, "params": params}


@app.get("/jobs")
def list_jobs(limit: int = 50):
    return [j.to_dict() for j in manager.list()[:limit]]


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _job_or_404(job_id)
    out = job.to_dict()
    if job.status == "queued":
        out["queue_position"] = manager.queue_position(job_id)
    return out


def _result_file(job_id, name):
    job = _job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"job is {job.status}")
    path = os.path.join(manager.results_dir, job_id, name)
    if not os.path.exists(path):
        raise HTTPException(410, "output file is gone")
    return path


@app.get("/jobs/{job_id}/image")
def job_image(job_id: str):
    return FileResponse(_result_file(job_id, "final.png"), media_type="image/png")


@app.get("/jobs/{job_id}/data")
def job_data(job_id: str):
    return FileResponse(_result_file(job_id, "final.npy"), media_type="application/octet-stream",
                        filename=f"{job_id}.npy")
