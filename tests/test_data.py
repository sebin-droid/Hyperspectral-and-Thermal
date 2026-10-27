import pandas as pd
import pytest

from urbanheat.cli import handover_filter, pair_of
from urbanheat.contracts import Scenario
from urbanheat.data import catalog, drive

HTML = """
<div class="flip-entry"><a href="https://drive.google.com/drive/folders/FOLDER1?usp=x">
<div class="flip-entry-title">PASSED-PAIRS</div></a><div class="flip-entry-last-modified"><div>Oct 24</div></div></div>
<div class="flip-entry"><a href="https://drive.google.com/file/d/FILE_2/view">
<div class="flip-entry-title">aux_layers.tif</div></a><div class="flip-entry-last-modified"><div>Oct 23</div></div></div>
"""


def test_parse_folder_html():
    assert drive.parse_folder_html(HTML) == [("FOLDER1", "PASSED-PAIRS", True, "Oct 24"),
                                             ("FILE_2", "aux_layers.tif", False, "Oct 23")]


def test_handover_filter():
    skip = handover_filter(["EMIT"])
    assert not skip("FINAL-DATASET/dataset_index.csv")
    assert not skip("EMIT-LANDSAT")
    assert not skip("EMIT-LANDSAT/PASSED-PAIRS/12/emit_reflectance.tif")
    assert skip("EMIT-LANDSAT/pairs")
    assert skip("EMIT-LANDSAT/PASSED-PAIRS/_to_delete")
    assert skip("ENMAP-LANDSAT/PASSED-PAIRS/3/aux_layers.tif")
    assert skip("screening/cell_fit.csv")
    assert pair_of("EMIT-LANDSAT/PASSED-PAIRS/12/aux_layers.tif") == ("EMIT", "12")
    assert pair_of("FINAL-DATASET/dataset_index.csv") is None


def _index(tmp_path, splits):
    rows = [dict(sensor="EMIT", pair_no=i, rel_dir=f"EMIT-LANDSAT/PASSED-PAIRS/{i}", reflectance_file="emit_reflectance.tif",
                 landsat_file="landsat_x.tif", aux_file="aux_layers.tif", cell_id=c, split=s)
            for i, (c, s) in enumerate(splits, 1)]
    path = tmp_path / "dataset_index.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_index_and_split(tmp_path):
    path = _index(tmp_path, [("a", "train"), ("a", "train"), ("b", "test")])
    pair_dir = tmp_path / "raw" / "EMIT-LANDSAT/PASSED-PAIRS/1"
    pair_dir.mkdir(parents=True)
    for f in ["emit_reflectance.tif", "landsat_x.tif", "aux_layers.tif"]:
        (pair_dir / f).write_bytes(b"x")
    idx = catalog.load_index(path, tmp_path / "raw")
    assert idx.pair_id.tolist() == ["EMIT_1", "EMIT_2", "EMIT_3"]
    assert idx.on_disk.tolist() == [True, False, False]
    catalog.check_split(idx)
    assert catalog.summary(idx).loc["total", "cells"] == 2


def test_split_leak_is_caught(tmp_path):
    idx = catalog.load_index(_index(tmp_path, [("a", "train"), ("a", "test")]))
    with pytest.raises(ValueError, match="more than one split"):
        catalog.check_split(idx)


def test_scenario_rejects_non_levers():
    Scenario("greening", {"NDVI": 0.1})
    with pytest.raises(ValueError):
        Scenario("bad", {"elevation_m": -10})
