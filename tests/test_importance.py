"""Uji F5c: permutasi blok menjaga korelasi internal, dan blok sinyal lebih penting."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.importance import drops, macro_f1, permute_block
from src.load_data import read_config


@pytest.fixture(scope="module")
def cfg():
    return read_config()


def test_permutasi_blok_menjaga_korelasi_internal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (50, 4))
    x[:, 1] = x[:, 0] * 2.0                      # kolom 0 dan 1 satu blok, berkorelasi sempurna
    out = permute_block(x, [0, 1], np.random.default_rng(7))
    assert np.array_equal(out[:, 2:], x[:, 2:])  # kolom di luar blok tidak berubah
    assert np.allclose(out[:, 1], out[:, 0] * 2.0)
    assert np.array_equal(np.sort(out[:, 0]), np.sort(x[:, 0]))
    assert not np.array_equal(out[:, 0], x[:, 0])
    sendiri = permute_block(x, [0], np.random.default_rng(7))
    assert not np.allclose(sendiri[:, 1], sendiri[:, 0] * 2.0)


def test_blok_sinyal_penurunan_lebih_besar(cfg):
    rng = np.random.default_rng(2)
    n = 400
    y = (rng.normal(0, 1, n) > 0).astype(int)
    x = np.column_stack([y * 3.0 + rng.normal(0, 0.3, n), rng.normal(0, 1, n)])
    pipe = RandomForestClassifier(n_estimators=50, random_state=cfg["seed"]).fit(x, y)
    baseline = macro_f1(y, pipe.predict(x))
    repeat = cfg["importance"]["block_repeat"]
    d_sinyal = drops(pipe, x, y, [0], repeat, np.random.default_rng(3), baseline)
    d_noise = drops(pipe, x, y, [1], repeat, np.random.default_rng(3), baseline)
    assert len(d_sinyal) == len(d_noise) == repeat
    assert np.mean(d_sinyal) > np.mean(d_noise)
    assert np.mean(d_sinyal) > 0.1
    assert abs(np.mean(d_noise)) < 0.05