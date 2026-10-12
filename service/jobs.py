"""In-memory job queue.

Jobs go into a queue.Queue. A few dispatcher threads pull from it and hand each
job to a process pool, so the simulation runs in its own process (no GIL fights
with the API, and a crashing kernel can't take the server down). The threads
just wait on the result and update job state + metrics.

Everything lives in memory, so restarting the server forgets old jobs (the
output files stay on disk though).
"""

import multiprocessing as mp
import queue
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

from service import metrics, settings
from service.worker import run_job


@dataclass
class Job:
    id: str
    params: dict
    status: str = "queued"  # queued -> running -> done | failed
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None

    def to_dict(self):
        return asdict(self)


class JobManager:
    def __init__(self, workers=settings.WORKERS, results_dir=settings.RESULTS_DIR):
        self.workers = workers
        self.results_dir = results_dir
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self.pool = None
        self.threads = []

    def start(self):
        # spawn instead of fork: forking a process that already touched Metal/numba threads is asking for trouble
        self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"))
        for i in range(self.workers):
            t = threading.Thread(target=self._dispatch_loop, name=f"dispatch-{i}", daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        for _ in self.threads:
            self.q.put(None)
        for t in self.threads:
            t.join(timeout=5)
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def submit(self, params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], params=params)
        with self.lock:
            self.jobs[job.id] = job
        metrics.JOBS_SUBMITTED.labels(params["backend"]).inc()
        metrics.JOBS_QUEUED.inc()
        self.q.put(job.id)
        return job

    def get(self, job_id) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def list(self):
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: j.submitted_at, reverse=True)

    def queue_position(self, job_id):
        with self.lock:
            queued = sorted((j for j in self.jobs.values() if j.status == "queued"), key=lambda j: j.submitted_at)
        for i, j in enumerate(queued):
            if j.id == job_id:
                return i
        return None

    def _dispatch_loop(self):
        while True:
            job_id = self.q.get()
            if job_id is None:
                return
            job = self.get(job_id)
            backend = job.params["backend"]

            metrics.JOBS_QUEUED.dec()
            metrics.JOBS_RUNNING.inc()
            job.status = "running"
            job.started_at = time.time()
            metrics.JOB_WAIT.observe(job.started_at - job.submitted_at)

            try:
                res = self.pool.submit(run_job, job.id, job.params, self.results_dir).result()
                job.result = res
                job.status = "done"

                cost = res["runtime_seconds"] / 3600 * settings.COST_PER_HOUR
                res["estimated_cost_dollars"] = cost
                metrics.JOB_RUNTIME.labels(backend).observe(res["runtime_seconds"])
                metrics.CELL_UPDATES.labels(backend).inc(res["cell_updates"])
                metrics.LAST_MLUPS.labels(backend).set(res["mlups"])
                metrics.COST.labels(backend).inc(cost)
                metrics.JOB_COST.labels(backend).observe(cost)
            except Exception as e:  # noqa: BLE001 - want to record any failure on the job
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
            finally:
                job.finished_at = time.time()
                metrics.JOBS_RUNNING.dec()
                metrics.JOBS_FINISHED.labels(backend, job.status).inc()
                metrics.JOB_WALL.labels(backend).observe(job.finished_at - job.submitted_at)
