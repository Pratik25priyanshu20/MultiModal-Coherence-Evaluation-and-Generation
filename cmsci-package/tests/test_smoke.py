"""Smoke tests: package imports, assets resolve, scorer builds and scores."""

import numpy as np
import pytest


def test_import_is_light():
    import cmsci

    assert cmsci.__version__


def test_assets_resolve():
    from cmsci import resolve_assets

    assets = resolve_assets()
    assert assets.calibration is not None, "bundled calibration missing"
    assert assets.bridge is not None, "bundled bridge weights missing"
    assert assets.image_index is not None, "bundled image index missing"
    assert assets.audio_index is not None, "bundled audio index missing"


@pytest.fixture(scope="module")
def scorer():
    from cmsci import CoherenceScorer

    return CoherenceScorer()


def test_score_text_image(scorer, tmp_path_factory):
    from PIL import Image

    tmp = tmp_path_factory.mktemp("data")
    img = tmp / "green.png"
    Image.new("RGB", (224, 224), (34, 139, 34)).save(img)

    result = scorer.score(text="a plain green field", image=str(img))
    assert result.cmsci is not None
    assert result.st_i is not None
    assert -1.0 <= result.st_i <= 1.0


def test_score_full_triple(scorer, tmp_path_factory):
    import soundfile as sf
    from PIL import Image

    tmp = tmp_path_factory.mktemp("data")
    img = tmp / "blue.png"
    Image.new("RGB", (224, 224), (30, 60, 200)).save(img)

    sr = 48000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    noise = (0.2 * np.random.default_rng(0).standard_normal(t.shape)).astype("float32")
    wav = tmp / "noise.wav"
    sf.write(wav, noise, sr)

    result = scorer.score(
        text="ocean waves under a blue sky", image=str(img), audio=str(wav)
    )
    assert result.cmsci is not None
    assert result.st_i is not None
    assert result.st_a is not None
