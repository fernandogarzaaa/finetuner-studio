import threading
import time

import pytest
from fastapi.testclient import TestClient

from finetuner_studio.server.app import create_app
from finetuner_studio.server.settings import Settings
from finetuner_studio.server.worker import Worker

SPEC = """
version: 1
name: api-test
model: {base_model: sshleifer/tiny-gpt2}
dataset: {path: x/y}
training: {max_steps: 5, logging_steps: 1}
"""


def _settings(tmp_path, token=None):
    return Settings(home=tmp_path, db_url=f"sqlite:///{tmp_path / 'db.sqlite'}", api_token=token,
                    poll_interval=0.1)


@pytest.fixture
def client(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_validate_reports_field_errors(client):
    r = client.post("/api/spec/validate", json={"spec_yaml": "version: 1\nname: x\n"}).json()
    assert r["valid"] is False
    locs = {e["loc"] for e in r["errors"]}
    assert "model" in locs and "dataset" in locs


def test_create_rejects_invalid_spec(client):
    r = client.post("/api/runs", json={"spec_yaml": "name: [oops"})
    assert r.status_code == 422


def test_examples_listed(client):
    names = [e["name"] for e in client.get("/api/spec/examples").json()]
    assert "qwen2.5-1.5b-alpaca-t4" in names


def test_auth_required_when_token_set(tmp_path):
    app = create_app(_settings(tmp_path, token="s3cret"))
    with TestClient(app) as c:
        assert c.get("/api/runs").status_code == 401
        assert c.get("/api/runs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/api/runs?token=s3cret").status_code == 200
        assert c.get("/api/health").status_code == 200


def test_cancel_queued_run(client):
    run = client.post("/api/runs", json={"spec_yaml": SPEC, "simulate": True}).json()
    r = client.post(f"/api/runs/{run['id']}/cancel").json()
    assert r["status"] == "cancelled"
    assert client.post(f"/api/runs/{run['id']}/cancel").status_code == 409


def test_end_to_end_simulated_run(client):
    run = client.post("/api/runs", json={"spec_yaml": SPEC, "simulate": True}).json()
    assert run["status"] == "queued"

    worker: Worker = client.app.state.worker
    t = threading.Thread(target=worker.serve_forever, daemon=True)
    t.start()
    deadline = time.time() + 60
    while time.time() < deadline:
        status = client.get(f"/api/runs/{run['id']}").json()
        if status["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.2)
    worker.stop()

    assert status["status"] == "succeeded", status
    assert status["decision"] == "SHIP"
    assert status["progress"] == 1.0

    events = client.get(f"/api/runs/{run['id']}/events").json()
    assert any(e["type"] == "train_metrics" for e in events)
    assert client.get(f"/api/runs/{run['id']}/report").json()["verdict"]["decision"] == "SHIP"
    assert "SHIP" in client.get(f"/api/runs/{run['id']}/report.md").text

    with client.stream("GET", f"/api/runs/{run['id']}/stream") as s:
        body = "".join(s.iter_text())
    assert "event: run_event" in body and "event: run_status" in body

    assert client.delete(f"/api/runs/{run['id']}").status_code == 204
    assert client.get(f"/api/runs/{run['id']}").status_code == 404
