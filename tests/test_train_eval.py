"""Uji F5: grid eksperimen dan semantik agregasi (mean±sd antar seed, bukan antar fit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.load_data import read_config
from src.train_eval import METRICS, build_grid, paired_delta, summarize


@pytest.fixture(scope="module")
def cfg():
    return read_config()


def test_grid_300_inti_plus_varian_jarak(cfg):
    grid = build_grid(cfg)
    default = cfg["fusion"]["post_concat_scaling"]
    inti = [g for g in grid if g[2] == default]
    varian = [g for g in grid if g[2] != default]
    n_combo = len(cfg["protocol"]["pairing_seeds"]) * cfg["protocol"]["n_folds"]
    assert len(inti) * n_combo == 300
    assert len(varian) * n_combo == 50
    assert all(fs == "fused" and m in cfg["fusion"]["distance_models"] for fs, m, _ in varian)
    assert len({(fs, m) for fs, m, _ in inti}) == 12


def test_ringkasan_memakai_seed_sebagai_unit(cfg):
    """Fold dirata-ratakan dulu di dalam seed; sd dihitung antar seed (ddof=1)."""
    rows = []
    for seed, nilai in ((1, (0.8, 1.0)), (2, (0.6, 0.8))):
        for fold, v in enumerate(nilai, start=1):
            rows.append(
                {"feature_set": "fused", "model": "knn", "scaling": "standard",
                 "seed": seed, "fold": fold, **{m: v for m in METRICS}}
            )
    s = summarize(pd.DataFrame(rows))
    assert len(s) == 1
    r = s.iloc[0]
    assert r.n_seed == 2
    assert r.macro_f1_mean == pytest.approx(0.8, abs=1e-12)
    assert r.macro_f1_std == pytest.approx(np.std([0.9, 0.7], ddof=1), abs=1e-12)
    assert r.macro_f1_std != pytest.approx(np.std([0.8, 1.0, 0.6, 0.8], ddof=1), abs=1e-6)


def test_selisih_berpasangan_per_seed(cfg):
    rows = []
    for seed in (1, 2):
        for fs, v in (("visual", 0.90), ("sensor", 0.80), ("fused", 0.95)):
            rows.append(
                {"feature_set": fs, "model": "knn", "scaling": "standard",
                 "seed": seed, "fold": 1, **{m: v + 0.01 * seed for m in METRICS}}
            )
    d = paired_delta(pd.DataFrame(rows), cfg)
    assert len(d) == 2
    assert d.delta_vs_visual.tolist() == pytest.approx([0.05, 0.05], abs=1e-12)
    assert d.delta_vs_sensor.tolist() == pytest.approx([0.15, 0.15], abs=1e-12)