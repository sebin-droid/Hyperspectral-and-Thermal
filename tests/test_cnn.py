import torch

from urbanheat.models import cnn, train


def _batch(b=2, h=32, w=32):
    return {"hs": torch.randn(b, 122, h, w), "ls": torch.randn(b, 9, h, w), "env": torch.randn(b, 3, h, w),
            "thermal": torch.randn(b, 1, h, w), "sensor": torch.tensor([0, 2][:b])}


def _model(v):
    return cnn.HeatCNN(v, hs_proj=torch.randn(16, 122) if cnn.VARIANTS[v].get("pca") else None)


def test_every_variant_keeps_the_input_size_for_any_scene_size():
    for v in cnn.VARIANTS:
        m = _model(v).eval()
        assert m(_batch()).shape == (2, 32, 32)
        assert m(_batch(1, 131, 97)).shape == (1, 131, 97)          # whole scenes of any size


def test_branches_use_only_their_inputs():
    m = cnn.HeatCNN("landsat_cnn").eval()
    b = _batch(1)
    p0 = m(b)
    b["hs"] = b["hs"] * 100
    b["thermal"] = b["thermal"] * 100
    assert torch.allclose(p0, m(b))                                   # landsat_cnn ignores hs and thermal
    f = cnn.HeatCNN("fusion_cnn").eval()
    assert not torch.allclose(f(_batch(1)), f({**_batch(1), "thermal": torch.zeros(1, 1, 32, 32)}))


def test_pca_variant_projects_and_keeps_the_projection_fixed():
    m = _model("spectral_pca_cnn")
    assert m.surface.stem[0].in_channels == 16 + 9 + 3
    assert "hs_proj" in dict(m.named_buffers()) and all(p is not m.hs_proj for p in m.parameters())


def test_masked_mse_ignores_invalid_pixels():
    pred, y = torch.zeros(1, 2, 2), torch.tensor([[[1.0, 100.0], [1.0, 1.0]]])
    mask = torch.tensor([[[True, False], [True, True]]])
    assert train.masked_mse(pred, y, mask).item() == 1.0


def test_augment_applies_the_same_transform_to_inputs_and_target():
    b = {k: v for k, v in _batch(1).items()}
    b["y"], b["mask"] = b["ls"][:, 0].clone(), torch.ones(1, 32, 32, dtype=torch.bool)
    out = train.augment(b)
    assert torch.equal(out["y"], out["ls"][:, 0])


def test_flops_and_params():
    m = cnn.HeatCNN("thermal_cnn")
    assert cnn.count_params(m) > 0 and cnn.count_flops(m.eval(), _batch(2)) > 0
