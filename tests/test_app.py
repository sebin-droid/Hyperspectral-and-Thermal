import json

import pytest
import rasterio
import torch
from fastapi.testclient import TestClient
from rasterio.warp import transform

from conftest import TRANSFORM
from urbanheat.app import assistant, server
from urbanheat.data.dataset import PatchDataset
from urbanheat.efficient import lite
from urbanheat.efficient.distill import Prepared
from urbanheat.explain import xai
from urbanheat.models import train


def latlng(r, c):
    """EPSG:4326 position of pixel (row, col) centre of the fake pairs (EPSG:32643)."""
    xs, ys = transform("EPSG:32643", "EPSG:4326", [TRANSFORM[2] + 60 * (c + 0.5)], [TRANSFORM[5] - 60 * (r + 0.5)])
    return ys[0], xs[0]


@pytest.fixture
def client(fake_h5):
    torch.manual_seed(0)
    return TestClient(server.create_app(fake_h5, model=lite.lite_cnn("fusion_cnn", 8, 4).eval()))


@pytest.fixture
def pid(client):
    return client.get("/api/places").json()["places"][0]["images"][0]["pid"]


def test_places_group_images_and_list_layers(client):
    meta = client.get("/api/places").json()
    assert meta["n_images"] == 4 and sum(len(p["images"]) for p in meta["places"]) == 4
    assert {"rgb", "hotspots", "NDVI"} <= {l["name"] for l in meta["layers"] if l["basic"]}
    assert set(meta["presets"]) == set(server.PRESETS)


def test_layers(client, pid):
    for name in ["rgb", "hotspots", "lst", "pred", "NDVI", "built_share", "attr_NDVI", "gradcam_thermal"]:
        r = client.get(f"/api/layer/{pid}/{name}")
        assert r.status_code == 200, name
        lay = r.json()
        assert lay["url"].startswith("data:image/png;base64,")
        (s, w), (n, e) = lay["bounds"]
        assert s < n and w < e
    assert client.get(f"/api/layer/{pid}/nope").status_code == 404
    assert client.get("/api/layer/nope/lst").status_code == 404


def test_summary_and_its_spots_lie_inside_the_image(client, pid):
    s = client.get(f"/api/summary/{pid}").json()
    assert s["coolest_c"] < s["typical_c"] < s["hottest_c"] and 0 <= s["built_pct"] <= 100
    (south, west), (north, east) = client.get(f"/api/layer/{pid}/lst").json()["bounds"]
    for k in ("hottest", "coolest"):
        p = s["spots"][k]
        assert south <= p["lat"] <= north and west <= p["lng"] <= east
    assert s["spots"]["hottest"]["diff_c"] > s["spots"]["coolest"]["diff_c"]


def test_spot_explains_a_circle_around_the_click(client, pid):
    lat, lng = latlng(20, 20)
    small = client.post("/api/spot", json=dict(pid=pid, lat=lat, lng=lng, radius_m=60)).json()
    big = client.post("/api/spot", json=dict(pid=pid, lat=lat, lng=lng, radius_m=600)).json()
    assert small["facts"]["pixels"] < big["facts"]["pixels"] <= 21 * 21
    assert {r["label"] for r in big["reasons"]} >= {"Greenery", "Buildings and pavement", "Heat seen on other days"}
    off = client.post("/api/spot", json=dict(pid=pid, lat=lat + 1, lng=lng, radius_m=600))
    assert off.status_code == 400 and "No data" in off.json()["detail"]


def test_whatif_presets_stay_local(client, pid):
    lat, lng = latlng(15, 15)
    r = client.post("/api/whatif", json=dict(pid=pid, lat=lat, lng=lng, radius_m=300, preset="built_to_green")).json()
    assert r["label"].startswith("Turn built-up") and "not a causal guarantee" in r["note"]
    assert 0 < r["stats"]["pixels"] < 40 * 40
    assert client.post("/api/whatif", json=dict(pid=pid, lat=lat, lng=lng, preset="nope")).status_code == 400


def test_ask_answers_offline_without_credentials(client, pid, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent")
    lat, lng = latlng(20, 20)
    for body in [dict(question="What does a high NDBI value indicate?", pid=pid),
                 dict(question="Why is this spot hot?", pid=pid, lat=lat, lng=lng)]:
        r = client.post("/api/ask", json=body).json()
        assert r["mode"] == "offline" and "ground" in r["answer"]


def test_retrieval_finds_the_right_passage():
    kb = assistant.Knowledge()
    assert "NDBI" in kb.search("what does a high NDBI value indicate")[0]["title"]
    assert kb.search("is this air temperature?")[0]["title"].startswith("Is this the air temperature")
    assert kb.search("can I trust the what if results")[0]["title"].startswith("Can I trust")


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


def test_export_geotiffs(client, pid, tmp_path):
    for kind, first in [("pred", "predicted_lst_c"), ("attr", "hyperspectral")]:
        r = client.get(f"/api/export/{pid}/{kind}.tif")
        assert r.status_code == 200 and r.headers["content-type"] == "image/tiff"
        (tmp_path / "x.tif").write_bytes(r.content)
        with rasterio.open(tmp_path / "x.tif") as t:
            assert t.descriptions[0] == first
    assert client.get(f"/api/export/{pid}/nope.tif").status_code == 404


def test_offline_answers_follow_the_question(client, pid, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent")
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    lat, lng = latlng(20, 20)
    spot = dict(pid=pid, lat=lat, lng=lng)
    cool = client.post("/api/ask", json={"question": "is this place cool?", **spot}).json()["answer"]
    assert cool.split(".")[0] in ("Yes", "No") or cool.startswith("Neither")
    assert "Starting from an average place" in client.post("/api/ask", json={"question": "Why is this spot hot?", **spot}).json()["answer"]
    assert client.post("/api/ask", json={"question": "What does NDVI mean?"}).json()["answer"].startswith("NDVI")
    assert "off by" in client.post("/api/ask", json={"question": "How sure is the model?", "pid": pid}).json()["answer"]
    assert "step 4" in client.post("/api/ask", json={"question": "Would trees help here?", **spot}).json()["answer"]
    air = client.post("/api/ask", json={"question": "Is this the air temperature?", "pid": pid}).json()
    assert air["answer"].startswith("No.") and "Details" not in air["answer"] and len(air["sources"]) == 1
    hot = client.post("/api/ask", json={"question": "Which parts of this place are hottest?", "pid": pid}).json()["answer"]
    assert "Hottest spot" in hot
