import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from rasterio.warp import transform

from conftest import TRANSFORM
from urbanheat.app import assistant, server
from urbanheat.efficient import lite


def lonlat(r, c):
    """EPSG:4326 position of pixel (row, col) centre of the fake pairs (EPSG:32643)."""
    xs, ys = transform("EPSG:32643", "EPSG:4326", [TRANSFORM[2] + 60 * (c + 0.5)], [TRANSFORM[5] - 60 * (r + 0.5)])
    return [xs[0], ys[0]]


SQUARE = {"type": "Polygon", "coordinates": [[lonlat(4.5, 4.5), lonlat(4.5, 14.5), lonlat(14.5, 14.5), lonlat(14.5, 4.5), lonlat(4.5, 4.5)]]}


@pytest.fixture
def client(fake_h5):
    torch.manual_seed(0)
    return TestClient(server.create_app(fake_h5, model=lite.lite_cnn("fusion_cnn", 8, 4).eval()))


def test_cities_and_layers(client):
    meta = client.get("/api/cities").json()
    assert len(meta["cities"]) == 4 and meta["cities"][0]["split"] == "test"
    pid = meta["cities"][0]["pid"]
    for name in ["lst", "pred", "anomaly", "rgb", "NDVI", "attr_NDVI", "gradcam_thermal"]:
        r = client.get(f"/api/layer/{pid}/{name}")
        assert r.status_code == 200, name
        lay = r.json()
        assert lay["url"].startswith("data:image/png;base64,")
        (s, w), (n, e) = lay["bounds"]
        assert s < n and w < e
    assert client.get(f"/api/layer/{pid}/nope").status_code == 404
    assert client.get("/api/layer/nope/lst").status_code == 404


def test_scenario_needs_a_region_and_stays_local(client):
    pid = client.get("/api/cities").json()["cities"][0]["pid"]
    body = dict(pid=pid, name="built_to_green", changes=server.PRESETS["built_to_green"])
    assert client.post("/api/scenario", json=body).status_code == 400
    r = client.post("/api/scenario", json={**body, "geometry": SQUARE}).json()
    assert r["stats"]["pixels_inside"] == 100 and "not a causal guarantee" in r["note"]
    assert client.post("/api/scenario", json={**body, "geometry": SQUARE, "changes": {"albedo": 0.1}}).status_code == 400


def test_explain_region_gives_facts_and_factors(client):
    pid = client.get("/api/cities").json()["cities"][0]["pid"]
    whole = client.post("/api/explain", json={"pid": pid}).json()
    region = client.post("/api/explain", json={"pid": pid, "geometry": SQUARE}).json()
    assert region["facts"]["pixels"] == 100 < whole["facts"]["pixels"]
    assert {f["feature"] for f in region["factors"]} >= {"NDVI", "built_share", "thermal context"}
    assert len(region["layers"]) == 2


def test_ask_answers_offline_without_credentials(client, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent")
    pid = client.get("/api/cities").json()["cities"][0]["pid"]
    r = client.post("/api/ask", json={"question": "What does a high NDBI value indicate?", "pid": pid}).json()
    assert r["mode"] == "offline" and r["sources"] and "surface" in r["answer"]


def test_retrieval_finds_the_right_passage():
    kb = assistant.Knowledge()
    assert kb.search("what does a high NDBI value indicate")[0]["title"].startswith("NDBI")
    assert kb.search("is this air temperature?")[0]["title"].startswith("Land surface temperature")
    assert any("Limits" in h["title"] for h in kb.search("can I trust the scenario, is it causal"))


def test_ask_uses_claude_when_available():
    class Fake:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    assert kw["model"] == "claude-opus-5-5" and "FACTS" in kw["messages"][0]["content"]
                    block = type("B", (), {"type": "text", "text": "Hot because of roofs [1]."})
                    return type("R", (), {"stop_reason": "end_turn", "content": [block]})

    r = assistant.ask("why hot?", {"top_factors": "NDBI +1.2 °C"}, client=Fake())
    assert r["mode"] == "claude" and r["answer"].startswith("Hot")
