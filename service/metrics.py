from prometheus_client import Counter, Gauge, Histogram

DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)

JOBS_SUBMITTED = Counter("heatsim_jobs_submitted_total", "Jobs accepted by the API", ["backend"])
JOBS_FINISHED = Counter("heatsim_jobs_finished_total", "Jobs that finished", ["backend", "status"])

JOBS_QUEUED = Gauge("heatsim_jobs_queued", "Jobs waiting for a worker")
JOBS_RUNNING = Gauge("heatsim_jobs_running", "Jobs currently running")

JOB_RUNTIME = Histogram(
    "heatsim_job_runtime_seconds", "Time spent stepping the simulation", ["backend"], buckets=DURATION_BUCKETS
)
JOB_WALL = Histogram(
    "heatsim_job_wall_seconds", "Submit to finish, including queueing and output", ["backend"],
    buckets=DURATION_BUCKETS,
)
JOB_WAIT = Histogram("heatsim_job_queue_wait_seconds", "Time a job sat in the queue", buckets=DURATION_BUCKETS)

CELL_UPDATES = Counter("heatsim_cell_updates_total", "Grid cell updates computed", ["backend"])
LAST_MLUPS = Gauge("heatsim_last_job_mlups", "Throughput of the most recent job", ["backend"])

COST = Counter("heatsim_estimated_cost_dollars_total", "Estimated compute cost at HEATSIM_COST_PER_HOUR", ["backend"])
JOB_COST = Histogram(
    "heatsim_job_cost_dollars", "Estimated cost per job", ["backend"],
    buckets=(1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2),
)
