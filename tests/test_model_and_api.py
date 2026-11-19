import json

import numpy as np
from fastapi.testclient import TestClient

from comms_triage.api.main import create_app
from comms_triage.classify.train import choose_threshold, routing_decisions


def test_unclear_is_never_auto_routed():
    auto = routing_decisions(["PR", "Unclear", "Design"], np.array([0.99, 0.99, 0.5]), 0.8)
    assert auto.tolist() == [True, False, False]


def test_choose_threshold_picks_lowest_that_meets_target():
    curve = [{"threshold": 0.5, "auto_accuracy": 0.90}, {"threshold": 0.7, "auto_accuracy": 0.96},
             {"threshold": 0.9, "auto_accuracy": 0.99}]
    assert choose_threshold(curve, 0.95) == 0.7
    assert choose_threshold(curve, 0.999) == 0.9  # nothing meets it: take the most accurate


def test_training_writes_metrics(model_dir):
    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert metrics["n_test"] >= 40
    assert 0.0 < metrics["test"]["accuracy"] <= 1.0
    assert metrics["best_model"] in metrics["cv"]
    assert (model_dir / "model.joblib").exists()


def test_api_flow(service):
    app = create_app(service)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"

        r = client.post("/triage", json={"text": "Can design make a banner for the launch blog post by Wednesday?",
                                         "source": "web"})
        assert r.status_code == 200
        req = r.json()
        assert req["predicted_team"] in ("PR", "Social", "Design", "Internal Comms", "Exec Comms", "Unclear")
        assert req["route"] in ("auto", "review")
        assert req["deadline"] == "wednesday"
        assert "banner" in req["assets"]
        assert req["draft_reply"]

        vague = client.post("/triage", json={"text": "hello? is anyone around"}).json()
        assert vague["route"] == "review" and vague["status"] == "pending_review"

        bad = client.post(f"/requests/{vague['id']}/review", json={"final_team": "Marketing"})
        assert bad.status_code == 422
        done = client.post(f"/requests/{vague['id']}/review", json={"final_team": "PR", "reviewer": "me"}).json()
        assert done["status"] == "resolved" and done["final_team"] == "PR"
        assert client.post("/requests/9999/review", json={"final_team": "PR"}).status_code == 404

        stats = client.get("/stats").json()
        assert stats["total"] == 2 and stats["reviewed"] == 1
        pending_ids = [r["id"] for r in client.get("/requests", params={"status": "pending_review"}).json()]
        assert vague["id"] not in pending_ids

        feedback = [json.loads(line) for line in client.get("/feedback").text.splitlines()]
        assert feedback == [{"id": f"fb{vague['id']}", "text": vague["text"], "team": "PR", "source": "feedback",
                             "predicted_team": vague["predicted_team"]}]

        hits = client.get("/search", params={"q": "When did we launch Edge Shield?", "kind": "doc"}).json()
        assert hits and hits[0]["doc_id"] == "kb01"
        assert client.post("/triage", json={"text": ""}).status_code == 422
