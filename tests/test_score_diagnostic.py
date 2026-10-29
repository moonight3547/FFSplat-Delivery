import pytest
from scripts.rescore_historical import quality_scores, time_score


def test_timeout_cannot_be_presented_as_zero_points_but_valid_total():
    assert time_score(10, 100) == 20
    assert time_score(55, 100) == pytest.approx(10)
    assert time_score(100, 100) is None
    assert time_score(110, 100) is None


def test_quality_score_scale_and_clipping():
    assert quality_scores(dict(psnr=30, ssim=.9, lpips=.1)) == pytest.approx(dict(psnr=12, ssim=16, lpips=30))
    assert quality_scores(dict(psnr=45, ssim=1, lpips=0)) == dict(psnr=20, ssim=20, lpips=40)


def test_nonfinite_time_does_not_receive_a_score():
    with pytest.raises(ValueError): time_score(float('nan'), 100)
    with pytest.raises(ValueError): time_score(10, float('inf'))
