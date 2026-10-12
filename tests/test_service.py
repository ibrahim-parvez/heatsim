import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from service import app as app_module

    app_module.manager.results_dir = str(tmp_path_factory.mktemp("jobs"))
    with TestClient(app_module.app) as c:
        yield c


def wait_for(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(job_id)


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_job_roundtrip(client):
    r = client.post("/jobs", json={"n": 64, "steps": 50, "backend": "numpy"})
    assert r.status_code == 202
    job = wait_for(client, r.json()["id"])
    assert job["status"] == "done", job["error"]
    assert job["result"]["cell_updates"] == 62 * 62 * 50

    img = client.get(f"/jobs/{job['id']}/image")
    assert img.status_code == 200
    assert img.content[:4] == b"\x89PNG"


def test_bad_requests(client):
    assert client.post("/jobs", json={"n": 10**6}).status_code == 422
    assert client.post("/jobs", json={"r": 0.5}).status_code == 422
    assert client.post("/jobs", json={"backend": "cuda"}).status_code == 422
    assert client.post("/jobs", json={"init": "volcano"}).status_code == 422
    assert client.get("/jobs/nope").status_code == 404


def test_metrics_exposed(client):
    text = client.get("/metrics/").text
    assert "heatsim_jobs_submitted_total" in text
    assert "heatsim_job_runtime_seconds_bucket" in text
