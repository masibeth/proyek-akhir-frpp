"""Uji anti-bocor protokol F4. Semua uji memakai data sintetis, bukan data nyata."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.load_data import read_config
from src.protocol import feature_matrix, fold_splits, iter_folds, make_pipeline

CHANNELS = ["Mq-2", "Mq-3", "Mq-4", "Mq-5", "Mq-135", "Mq-136", "Mq-137", "Mq-138"]


@pytest.fixture(scope="module")
def cfg():
    return read_config()


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(7)
    n_v = 100
    v_label = np.array(["fresh"] * 50 + ["spoiled"] * 50)
    v = pd.DataFrame(
        {"visual_id": np.arange(n_v), "label": v_label, "group_id": np.arange(n_v) // 5}
    )
    for i in range(20):
        v[f"f{i:02d}"] = rng.normal(0, 1, n_v) + (v_label == "spoiled") * 2.0

    n_s = 120
    minutes = np.arange(1, n_s + 1)
    s_label = np.array(["fresh"] * 60 + ["spoiled"] * 60)
    s = pd.DataFrame(
        {"sensor_id": minutes, "label": s_label, "group_id": (minutes - 1) // 6}
    )
    for c in CHANNELS:
        s[c] = rng.normal(0, 1, n_s) + (minutes / n_s) * 3.0
    return v, s


def test_tidak_ada_visual_id_menyeberang(cfg, synthetic):
    v, s = synthetic
    for _, _, tr, te in iter_folds(cfg, v, s):
        assert set(tr["visual_id"]) & set(te["visual_id"]) == set()


def test_tidak_ada_sensor_id_menyeberang(cfg, synthetic):
    v, s = synthetic
    for _, _, tr, te in iter_folds(cfg, v, s):
        assert set(tr["sensor_id"]) & set(te["sensor_id"]) == set()


def test_group_tidak_menyeberang(cfg, synthetic):
    v, s = synthetic
    for _, _, tr, te in iter_folds(cfg, v, s):
        assert set(tr["v_group"]) & set(te["v_group"]) == set()
        assert set(tr["s_group"]) & set(te["s_group"]) == set()


def test_pasangan_berlabel_sama(cfg, synthetic):
    v, s = synthetic
    for _, _, tr, te in iter_folds(cfg, v, s):
        for frame in (tr, te):
            assert (frame["v_label"] == frame["s_label"]).all()
            assert (frame["label"] == frame["v_label"]).all()


def test_pairing_tanpa_pengembalian(cfg, synthetic):
    v, s = synthetic
    for _, _, tr, te in iter_folds(cfg, v, s):
        for frame in (tr, te):
            assert frame["visual_id"].is_unique
            assert frame["sensor_id"].is_unique


def test_kardinalitas_min_per_kelas(cfg, synthetic):
    v, s = synthetic
    splits = fold_splits(cfg, v, s)
    seeds = cfg["protocol"]["pairing_seeds"]
    frames = list(iter_folds(cfg, v, s))
    assert len(frames) == len(seeds) * len(splits)
    for (_, _, tr, te), (v_tr, v_te, s_tr, s_te) in zip(frames, splits * len(seeds)):
        for frame, v_pos, s_pos in ((tr, v_tr, s_tr), (te, v_te, s_te)):
            for cls in ("fresh", "spoiled"):
                n_v = int((v["label"].to_numpy()[v_pos] == cls).sum())
                n_s = int((s["label"].to_numpy()[s_pos] == cls).sum())
                assert int((frame["label"] == cls).sum()) == min(n_v, n_s)


def test_scaler_fit_hanya_di_latih(cfg, synthetic):
    v, s = synthetic
    _, _, tr, te = next(iter(iter_folds(cfg, v, s)))
    x_tr = feature_matrix(tr, "fused", cfg)
    x_te = feature_matrix(te, "fused", cfg)
    pipe = make_pipeline(cfg, "random_forest")
    pre = Pipeline(pipe.steps[:-1]).fit(x_tr)
    assert np.allclose(pre.transform(x_tr).mean(axis=0), 0, atol=1e-8)
    clip = pre.named_steps["clip"]
    train_mean = clip.transform(x_tr).mean(axis=0)
    global_mean = clip.transform(np.concatenate([x_tr, x_te])).mean(axis=0)
    assert np.allclose(pre.named_steps["scale"].mean_, train_mean, atol=1e-10)
    assert not np.allclose(pre.named_steps["scale"].mean_, global_mean, atol=1e-9)
    assert np.abs(pre.transform(x_te).mean(axis=0)).max() > 1e-3


def test_unimodal_memakai_baris_yang_sama(cfg, synthetic):
    v, s = synthetic
    for _, _, tr, te in iter_folds(cfg, v, s):
        for frame in (tr, te):
            x_fused = feature_matrix(frame, "fused", cfg)
            x_v = feature_matrix(frame, "visual", cfg)
            x_s = feature_matrix(frame, "sensor", cfg)
            assert x_fused.shape[0] == x_v.shape[0] == x_s.shape[0]
            assert np.array_equal(x_fused[:, : x_v.shape[1]], x_v)
            assert np.array_equal(x_fused[:, x_v.shape[1] :], x_s)