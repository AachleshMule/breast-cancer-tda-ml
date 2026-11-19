"""Unit and integration tests (run with: pytest -q). Uses only generated synthetic images."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_manifest  # noqa: E402
import make_synthetic_demo as demo  # noqa: E402
import run_experiments  # noqa: E402
import tda_pipeline as tp  # noqa: E402


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    rng = np.random.default_rng(0)
    for cls, mal in (("benign", False), ("malignant", True)):
        (d / cls).mkdir()
        for i in range(24):
            Image.fromarray(demo.make_image(rng, mal, 64)).save(d / cls / f"p{i // 2:03d}_{cls}_{i}.png")
    man = build_manifest.folder(d, patient_regex=r"(p\d+)")
    man.to_csv(d / "manifest.csv", index=False)
    return d


def test_normalization_range_and_integer_levels():
    img = np.random.default_rng(1).random((50, 50)) * 4000 + 17
    out = tp.normalize_0_100(img)
    assert out.min() == 0 and out.max() == 100
    assert np.allclose(out, np.round(out))


def test_constant_image_does_not_crash():
    assert tp.normalize_0_100(np.full((8, 8), 5.0)).max() == 0


def test_crop_to_mask_uses_bounding_box():
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:8] = 1
    assert tp.crop_to_mask(img, mask).shape == (3, 5)


def test_preprocess_output_shape_and_range(tmp_path):
    arr = (np.random.default_rng(2).random((90, 120)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(tmp_path / "a.png")
    m = np.zeros((90, 120), dtype=np.uint8)
    m[10:50, 20:80] = 255
    Image.fromarray(m).save(tmp_path / "a_mask.png")
    out = tp.preprocess_image(tmp_path / "a.png", [tmp_path / "a_mask.png"], size=64)
    assert out.shape == (64, 64) and out.min() >= 0 and out.max() <= 100


def test_persistence_diagrams_are_valid():
    X = np.round(np.random.default_rng(3).random((3, 32, 32)) * 100).astype(np.float32)
    D = tp.compute_diagrams(X, n_jobs=1)
    assert D.ndim == 3 and D.shape[0] == 3 and D.shape[2] == 3
    assert (D[:, :, 1] >= D[:, :, 0]).all()
    assert set(np.unique(D[:, :, 2])) <= {0.0, 1.0}


@pytest.mark.parametrize("name,kw,expected", [("pi", {"sigma": 1.0, "n_bins": 20}, 800), ("betti", {}, 200),
                                              ("landscape", {}, 1000), ("silhouette", {}, 200)])
def test_feature_dimensions_match_paper(name, kw, expected):
    X = np.round(np.random.default_rng(4).random((4, 32, 32)) * 100).astype(np.float32)
    D = tp.compute_diagrams(X, n_jobs=1)
    F = tp.flatten(tp.make_vectorizer(name, n_jobs=1, **kw).fit_transform(D))
    assert F.shape == (4, expected)


def test_select_dims_keeps_requested_dimension():
    X = np.round(np.random.default_rng(5).random((3, 32, 32)) * 100).astype(np.float32)
    D = tp.select_dims(tp.compute_diagrams(X, n_jobs=1), (1,))
    assert set(np.unique(D[:, :, 2])) == {1.0}


def test_patient_grouped_split_has_no_leakage():
    y = np.array([0, 1] * 40)
    groups = np.array([f"p{i // 2}" for i in range(80)])
    tr, te = tp.holdout_split(y, groups)
    assert not set(groups[tr]) & set(groups[te])
    assert abs(len(te) / len(y) - 0.2) < 0.1


def test_metrics_are_correct():
    m = tp.metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), np.array([0.1, 0.6, 0.7, 0.9]))
    assert m["accuracy"] == 0.75 and m["recall"] == 1.0 and abs(m["precision"] - 2 / 3) < 1e-9
    assert m["auc"] == 1.0


def test_manifest_parsing_skips_normal(tmp_path):
    (tmp_path / "m.csv").write_text("image_path,label\na.png,benign\nb.png,normal\nc.png,MALIGNANT\n")
    df = tp.load_manifest(tmp_path / "m.csv", tmp_path)
    assert list(df["y"]) == [0, 1]


def test_end_to_end_fast_run_writes_all_outputs(synthetic, tmp_path):
    out = run_experiments.main(["--manifest", str(synthetic / "manifest.csv"), "--root", str(synthetic),
                                "--name", "demo", "--out", str(tmp_path), "--fast", "--n-jobs", "1", "--size", "64"])
    for f in ("table3_classifiers.csv", "table4_vectorizations.csv", "table5_ablation.csv",
              "table6_extended_metrics.csv", "table7_hyperparameters.csv"):
        assert (out / "tables" / f).exists()
    for f in ("fig_persistence_diagrams.png", "fig_roc.png", "fig_ablation.png", "fig_median_persistence_images.png"):
        assert (out / "figures" / f).exists()
    assert (out / "run_log.json").exists() and (out / "summary.md").exists()
