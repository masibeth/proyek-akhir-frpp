"""Uji F5b: pasangan diskordan hanya dari split uji, tanpa ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.load_data import read_config
from src.probe_discordant import cross_pairs, discordant_frames, measure_probe
from src.protocol import feature_matrix, fold_splits, iter_folds, make_pipeline

CHANNELS = ["Mq-2", "Mq-3", "Mq-4", "Mq-5", "Mq-135", "Mq-136", "Mq-137", "Mq-138"]


@pytest.fixture(scope="module")
def cfg():
    return read_config()


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(5)
    n_v, n_s = 100, 120
    v_label = np.array(["fresh"] * 50 + ["spoiled"] * 50)
    v = pd.DataFrame({"visual_id": np.arange(n_v), "label": v_label, "group_id": np.arange(n_v) // 5})
    for i in range(20):
        v[f"f{i:02d}"] = rng.normal(0, 1, n_v) + (v_label == "spoiled") * 2.0
    minutes = np.arange(1, n_s + 1)
    s_label = np.array(["fresh"] * 60 + ["spoiled"] * 60)
    s = pd.DataFrame({"sensor_id": minutes, "label": s_label, "group_id": (minutes - 1) // 6})
    for c in CHANNELS:
        s[c] = rng.normal(0, 1, n_s) + (minutes / n_s) * 3.0
    return v, s


def test_diskordan_hanya_dari_split_uji(cfg, synthetic):
    v, s = synthetic
    splits = fold_splits(cfg, v, s)
    seeds = cfg["protocol"]["pairing_seeds"]
    for (seed, fold, tr, te), (v_tr, v_te, s_tr, s_te) in zip(iter_folds(cfg, v, s), splits * len(seeds)):
        rng = np.random.default_rng(seed * 1000 + fold)
        discs = discordant_frames(cfg, v, s, v_te, s_te, rng)
        assert len(discs) == len(cfg["probe"]["discordant_pairs"])
        for frame in discs.values():
            assert "label" not in frame.columns
            assert (frame["v_label"] != frame["s_label"]).all()
            assert frame["visual_id"].is_unique and frame["sensor_id"].is_unique
            assert set(frame["visual_id"]) <= set(v.iloc[v_te]["visual_id"])
            assert set(frame["sensor_id"]) <= set(s.iloc[s_te]["sensor_id"])
            assert not set(frame["visual_id"]) & set(tr["visual_id"])
            assert not set(frame["sensor_id"]) & set(tr["sensor_id"])


def test_kardinalitas_cross_pairing(cfg):
    rng = np.random.default_rng(1)
    v = np.array(["fresh"] * 7 + ["spoiled"] * 3)
    s = np.array(["fresh"] * 2 + ["spoiled"] * 9)
    pairs = cross_pairs(v, s, "fresh", "spoiled", rng)
    assert len(pairs) == min(7, 9) == 7
    assert len({a for a, _ in pairs}) == 7
    assert len({b for _, b in pairs}) == 7
    assert all(v[a] == "fresh" and s[b] == "spoiled" for a, b in pairs)


def test_ukuran_probe_konsisten(cfg, synthetic):
    v, s = synthetic
    seed, fold, tr, te = next(iter(iter_folds(cfg, v, s)))
    _, v_te, _, s_te = fold_splits(cfg, v, s)[0]
    discs = discordant_frames(cfg, v, s, v_te, s_te, np.random.default_rng(0))
    pos = cfg["metrics"]["positive_label"]
    pipe = make_pipeline(cfg, "random_forest").fit(
        feature_matrix(tr, "fused", cfg), (tr["label"].to_numpy() == pos).astype(int)
    )
    for frame in discs.values():
        pred = pipe.predict(feature_matrix(frame, "fused", cfg))
        m = measure_probe(pred, frame["v_label"].to_numpy(), frame["s_label"].to_numpy(), pos)
        assert m["agreement_visual"] + m["agreement_sensor"] == pytest.approx(1.0, abs=1e-12)
        assert m["share_predicted_busuk"] == pytest.approx(float(pred.mean()), abs=1e-12)
        assert 0.0 <= m["agreement_visual"] <= 1.0