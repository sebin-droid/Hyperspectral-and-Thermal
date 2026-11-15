import numpy as np
import torch

from urbanheat.data.dataset import PatchDataset
from urbanheat.efficient import benchmark, distill, lite
from urbanheat.models import cnn, train


def _batch(b=2, h=32, w=32):
    return {"hs": torch.randn(b, 122, h, w), "ls": torch.randn(b, 9, h, w), "env": torch.randn(b, 3, h, w),
            "thermal": torch.randn(b, 1, h, w), "sensor": torch.tensor([0, 2][:b])}


def test_lite_is_much_cheaper_and_keeps_any_scene_size():
    big, small = cnn.HeatCNN("fusion_cnn").eval(), lite.lite_cnn().eval()
    assert cnn.count_params(small) < cnn.count_params(big) / 8
    assert cnn.count_flops(small, _batch()) < cnn.count_flops(big, _batch()) / 8
    assert small(_batch(1, 131, 97)).shape == (1, 131, 97)
    assert all(isinstance(b, lite.SepBlock) for b in small.surface.blocks)


def test_prepared_clips_indices_and_blends_the_teacher(fake_h5):
    ds = PatchDataset(fake_h5, "train", patch=16)
    teacher = {pid: np.full((40, 40), 7.0, "float16") for pid in ds.pairs}
    p = distill.Prepared(ds, teacher, alpha=0.25)
    ch = distill.INDEX_CH
    raw = lambda o: o["ls"][ch] * p.norm["ls_std"][ch, None, None] + p.norm["ls_mean"][ch, None, None]
    s = p.scene(ds.pairs[0])
    assert np.isclose(raw(s)[0, 0, 1], 1.0) and np.isclose(raw(s)[0, 0, 0], -1.0)   # outliers 1452, -478 clipped
    assert raw(s)[:, s["mask"]].min() >= -1 - 1e-5 and raw(s)[:, s["mask"]].max() <= 1 + 1e-5
    assert np.array_equal(s["y"], ds.scene(ds.pairs[0])["y"])                        # scenes keep the true target
    i = 0
    item, orig = p[i], ds[i]
    m = orig["mask"]
    np.testing.assert_allclose(item["y"][m], 0.25 * orig["y"][m] + 0.75 * 7.0, rtol=1e-5)
    assert (item["ls"][:, ~m] == 0).all()


def test_benchmark_onnx_and_int8_match_torch(tmp_path):
    m = lite.lite_cnn().eval()
    b = _batch()
    r = benchmark.measure(m, b)
    assert r["params"] == cnn.count_params(m) and r["flops"] > 0 and r["latency_ms"] > 0 and r["size_mb"] > 0
    path = benchmark.export_onnx(m, b, tmp_path / "m.onnx")
    other = {k: v.numpy() for k, v in _batch(1, 45, 37).items()}                   # dynamic batch and scene size
    _, out = benchmark.onnx_run(path, other, reps=1)
    with torch.no_grad():
        ref = m({k: torch.as_tensor(v) for k, v in other.items()}).numpy()
    np.testing.assert_allclose(out, ref, atol=1e-4)
    q = benchmark.quantize_int8(path, tmp_path / "m_int8.onnx")
    _, outq = benchmark.onnx_run(q, other, reps=1)
    assert outq.shape == ref.shape and np.isfinite(outq).all()


def test_distill_run_saves_a_loadable_student(fake_h5, tmp_path):
    teacher_dir = tmp_path / "cnn_robust_fusion_cnn_s0"
    t = cnn.HeatCNN("fusion_cnn")
    train.save(t, teacher_dir, {}, [], {})                                         # config without "variant"
    metrics, student = distill.run(fake_h5, tmp_path / "student", teacher_dir, alpha=0.5, device="cpu",
                                   epochs=1, samples_per_epoch=8, batch_size=4)
    assert {"val", "test", "params", "flops", "latency_ms"} <= set(metrics) and metrics["alpha"] == 0.5
    loaded = lite.load(tmp_path / "student")
    b = _batch(1)
    with torch.no_grad():
        assert torch.allclose(loaded(b), student.cpu().eval()(b), atol=1e-5)
        assert lite.load(teacher_dir)(b).shape == (1, 32, 32)
