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


def test_insights_need_explain_tables(client):
    assert client.get("/api/insights").json()["available"] is False


def test_insights_from_a_run_folder(fake_h5, tmp_path):
    import json
    from urbanheat.data.dataset import PatchDataset
    from urbanheat.efficient.distill import Prepared
    from urbanheat.explain import xai
    from urbanheat.models import train

    torch.manual_seed(0)
    m = lite.lite_cnn("fusion_cnn", 8, 4).eval()
    run = tmp_path / "runs" / "m3_tiny"
    train.save(m, run, {"test": {"r2": 0.5, "mae": 1.0, "n": 10, "per_climate": {"tropical": {"r2": 0.5, "mae": 1.0, "n": 10}}}},
               [], dict(lite=True, variant="fusion_cnn", width=8, thermal_width=4, separable=True))
    ds = Prepared(PatchDataset(fake_h5, "test"))
    table, _ = xai.global_importance(m, ds, {p: {"climate_group": "tropical"} for p in ds.pairs}, steps=4)
    (run / "explain").mkdir()
    table.to_csv(run / "explain" / "global_importance.csv", index=False)
    (run / "explain" / "response_tests.csv").write_text(
        "model,scenario,change,mean_dlst_c,cities_cooling\nstudent m3_tiny,densify,{'built_share': 0.1},0.006,0.64\n")
    ins = TestClient(server.create_app(fake_h5, run_dir=run)).get("/api/insights").json()
    assert ins["available"] and set(ins["importance"]) == {"all", "tropical"}
    assert abs(sum(r["share_pct"] for r in ins["importance"]["all"]) - 100) < 0.1
    assert len(ins["findings"]) == 2 and "+0.006" in ins["findings"][1]
    assert [a["group"] for a in ins["accuracy"]] == ["all", "tropical"]
    json.dumps(ins)                                                   # no NaN leaks into the JSON


def test_export_geotiffs(client, tmp_path):
    import rasterio
    pid = client.get("/api/cities").json()["cities"][0]["pid"]
    for kind, first in [("pred", "predicted_lst_c"), ("attr", "hyperspectral")]:
        r = client.get(f"/api/export/{pid}/{kind}.tif")
        assert r.status_code == 200 and r.headers["content-type"] == "image/tiff"
        (tmp_path / "x.tif").write_bytes(r.content)
        with rasterio.open(tmp_path / "x.tif") as t:
            assert t.descriptions[0] == first
    assert client.get(f"/api/export/{pid}/nope.tif").status_code == 404
