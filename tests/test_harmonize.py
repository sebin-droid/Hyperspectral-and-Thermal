import numpy as np
import pytest
from affine import Affine

from urbanheat.harmonize import spatial, spectral


def test_wavelengths_from_descriptions():
    np.testing.assert_allclose(spectral.wavelengths(["381 nm (EMIT b000)", "434.7 nm (EnMAP b004)", "458 nm (B011)"]),
                               [381, 434.7, 458])


def test_interp_matrix_linear_and_gaps():
    w = spectral.interp_matrix([400, 410, 500], [400, 405, 450], max_gap=25)
    np.testing.assert_allclose(w[0], [1, 0, 0])
    np.testing.assert_allclose(w[1], [0.5, 0.5, 0])
    assert np.isnan(w[2]).all()                      # 410 -> 500 is a 90 nm gap: not covered


def test_interp_matrix_unsorted_source():
    w = spectral.interp_matrix([410, 400], [405], max_gap=25)
    np.testing.assert_allclose(w[0], [0.5, 0.5])


def test_common_grid_keeps_only_bands_every_sensor_covers():
    a = np.arange(400, 1001, 10)                     # covers 400-1000
    b = np.r_[np.arange(450, 701, 10), np.arange(800, 1001, 10)]   # gap 700-800
    grid = spectral.common_grid([a, b], 400, 1000, 10, max_gap=25)
    assert grid.min() == 450 and grid.max() == 1000
    assert not ((grid > 700) & (grid < 800)).any()


def test_apply_reproduces_a_linear_spectrum():
    src = np.array([400.0, 410, 420])
    cube = (src / 1000)[:, None, None] * np.ones((3, 2, 2))
    out = spectral.apply(spectral.interp_matrix(src, [405, 415], 25), cube)
    np.testing.assert_allclose(out[:, 0, 0], [0.405, 0.415], rtol=1e-6)


def test_block_mean_needs_enough_valid_pixels():
    a = np.array([[1, 2, 3, np.nan], [3, 4, np.nan, np.nan]], "float32")
    out = spatial.block_mean(a, 2, min_frac=0.75)
    assert out[0, 0] == 2.5 and np.isnan(out[0, 1])
    assert spatial.block_mean(a[None], 2).shape == (1, 1, 2)       # works with a band axis


def test_block_mode():
    a = np.array([[10, 10, 50, 0], [20, 10, 0, 0]], "uint8")
    np.testing.assert_array_equal(spatial.block_mode(a, 2), [[10, 50]])


def test_factor_and_transform():
    assert spatial.factor(30, 60) == 2 and spatial.factor(60, 60) == 1
    assert spatial.factor(29.97, 60) == 2
    with pytest.raises(ValueError):
        spatial.factor(45, 60)
    t = spatial.transform(Affine(30, 0, 1000, 0, -30, 2000), 2)
    assert (t.a, t.e, t.c, t.f) == (60, -60, 1000, 2000)


def test_spectral_indices_stay_in_range():
    from urbanheat.harmonize.build import spectral_indices
    opt = np.array([[0.05, 0.05, 0.05], [0.08, 0.08, np.nan], [0.10, 0.0011, 0.1],
                    [0.30, -0.001, 0.3], [0.20, 0.2, 0.2], [0.15, 0.15, 0.15]], "float32")[:, :, None]
    ndvi, ndbi, mndwi = spectral_indices(opt)[:, :, 0]
    assert np.isclose(ndvi[0], 0.5) and np.isclose(ndbi[0], -0.2)   # normal pixel unchanged
    assert ndvi[1] == -1                                              # (-0.001-0.0011)/(0.0001) -> clipped
    assert np.isnan(mndwi[2])                                         # missing band stays missing
    assert np.nanmax(np.abs(spectral_indices(opt))) <= 1
