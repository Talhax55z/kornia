"""Patch extraction from LAFs is translation-equivariant and consistent across extractors."""

import pytest
import torch

import kornia.feature as KF
from kornia.feature import laf as laf_mod

EXTRACTORS = [KF.extract_patches_simple, KF.extract_patches_from_pyramid]


def _laf(cx, cy, scale, dtype=torch.float64):
    return KF.laf_from_center_scale_ori(
        torch.tensor([[[cx, cy]]], dtype=dtype), torch.full((1, 1, 1, 1), scale, dtype=dtype)
    )


@pytest.fixture(params=["atlas", "levelwise"])
def pyramid_mode(request, monkeypatch):
    """Run pyramid tests through both the atlas path and the level-by-level fallback."""
    if request.param == "levelwise":
        monkeypatch.setattr(laf_mod, "_pyramid_atlas_fits", lambda *_: False)
    return request.param


@pytest.mark.parametrize("extract", EXTRACTORS)
def test_nonsquare_rotated_laf(extract, pyramid_mode):
    """A 90 degree LAF on a non-square image returns the transposed-and-flipped crop (catches W/H mix-ups)."""
    g = torch.Generator().manual_seed(1)
    img = torch.rand(1, 1, 50, 80, generator=g, dtype=torch.float64)
    ps, cx, cy = 21, 40, 25
    laf = KF.laf_from_center_scale_ori(
        torch.tensor([[[float(cx), float(cy)]]], dtype=torch.float64),
        torch.full((1, 1, 1, 1), ps / 2, dtype=torch.float64),
        torch.full((1, 1, 1), 90.0, dtype=torch.float64),
    )
    out = extract(img, laf, ps)[0, 0, 0]
    crop = img[0, 0, cy - 10 : cy + 11, cx - 10 : cx + 11]
    # Compare against both rotation directions; whichever kornia's convention is, one must match exactly.
    err = min((out - crop.rot90(1)).abs().max(), (out - crop.rot90(-1)).abs().max())
    assert err < 1e-10


def test_simple_and_pyramid_agree_at_level0(pyramid_mode):
    g = torch.Generator().manual_seed(2)
    img = torch.rand(2, 3, 64, 48, generator=g, dtype=torch.float64)
    xy = torch.tensor([[[20.3, 30.7], [10.0, 40.0]], [[24.0, 32.0], [30.2, 12.9]]], dtype=torch.float64)
    laf = KF.laf_from_center_scale_ori(xy, torch.full((2, 2, 1, 1), 8.0, dtype=torch.float64))
    a = KF.extract_patches_simple(img, laf, 16)
    b = KF.extract_patches_from_pyramid(img, laf, 16)
    assert torch.allclose(a, b, atol=1e-10)


@pytest.mark.parametrize("pad", [7, 16])
def test_affine_shape_estimate_is_invariant_to_padding(pad):
    """Padding the image and shifting the LAF by the same amount must not change the estimate."""
    import torch.nn.functional as F

    est = KF.LAFAffineShapeEstimator(32, preserve_orientation=False)
    inp = torch.zeros(1, 1, 32, 32)
    inp[:, :, 15:-15, 9:-9] = 1

    def run(img, c):
        laf = torch.tensor([[[[20.0, 0.0, c], [0.0, 20.0, c]]]])
        out = est(laf, img)[0, 0].clone()
        out[:, 2] -= c - 16.0
        return out

    base = run(inp, 16.0)
    padded = run(F.pad(inp, (pad,) * 4), 16.0 + pad)
    assert (padded - base).abs().max() < 1e-3


@pytest.mark.parametrize("pad", [7, 16])
@pytest.mark.parametrize("extract", EXTRACTORS)
def test_patches_are_invariant_to_padding(extract, pad):
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(3)
    img = torch.rand(1, 1, 40, 56, generator=g, dtype=torch.float64)
    xy = torch.tensor([[[20.3, 17.6]]], dtype=torch.float64)
    scale = torch.full((1, 1, 1, 1), 6.0, dtype=torch.float64)
    base = extract(img, KF.laf_from_center_scale_ori(xy, scale), 15)
    padded = extract(F.pad(img, (pad,) * 4, mode="replicate"), KF.laf_from_center_scale_ori(xy + pad, scale), 15)
    assert torch.allclose(base, padded, atol=1e-10)
