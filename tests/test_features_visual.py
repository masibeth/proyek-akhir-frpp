"""Uji F2: spesifikasi fitur visual 20-D diuji pada citra sintetis, bukan data nyata."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.features_visual import extract, feature_columns
from src.load_data import read_config


@pytest.fixture(scope="module")
def cfg():
    return read_config()


def simpan(path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


@pytest.fixture
def warna(cfg, tmp_path):
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 256, (256, 256, 3))
    p = tmp_path / "warna.png"
    simpan(p, arr)
    return p, arr


def test_dimensi_tepat_20(cfg, warna):
    p, _ = warna
    vec = extract(p, cfg)
    assert vec.shape == (cfg["visual"]["expected_dim"],) == (20,)
    assert np.isfinite(vec).all()
    assert (vec >= 0).all()
    assert feature_columns(20) == [f"f{i:02d}" for i in range(20)]


def test_histogram_per_kanal_berjumlah_satu(cfg, warna):
    p, _ = warna
    vec = extract(p, cfg)
    awal = 0
    for channel in ("h", "s", "v"):
        lebar = cfg["visual"]["hist_bins"][channel]
        assert vec[awal : awal + lebar].sum() == pytest.approx(1.0, abs=1e-6)
        awal += lebar
    assert awal == 16


def test_citra_identik_vektor_identik(cfg, warna, tmp_path):
    p, arr = warna
    q = tmp_path / "salinan.png"
    simpan(q, arr)
    assert np.array_equal(extract(p, cfg), extract(q, cfg))


def test_roi_membuang_latar(cfg, tmp_path):
    sisi = cfg["visual"]["resize"][0]
    potong = int(sisi * cfg["visual"]["roi_center_fraction"])
    margin = (sisi - potong) // 2
    arr = np.zeros((sisi, sisi, 3), dtype=np.uint8)
    arr[..., 1] = 255                      # seluruh bidang hijau
    arr[:margin, :, 0] = 255               # latar merah di tepi
    arr[-margin:, :, 0] = 255
    arr[:, :margin, 0] = 255
    arr[:, -margin:, 0] = 255
    arr[:margin, :, 1] = 0
    arr[-margin:, :, 1] = 0
    arr[:, :margin, 1] = 0
    arr[:, -margin:, 1] = 0
    p = tmp_path / "latar.png"
    simpan(p, arr)

    vec = extract(p, cfg)
    bins_h = cfg["visual"]["hist_bins"]["h"]
    assert vec[0] == pytest.approx(0.0, abs=1e-9)          # bin H merah kosong
    assert vec[int(bins_h / 3)] == pytest.approx(1.0, abs=1e-6)  # seluruh ROI hijau


def test_glcm_merespons_tekstur(cfg, tmp_path):
    sisi = 256
    seragam = np.full((sisi, sisi, 3), 128, dtype=np.uint8)
    kotak = np.indices((sisi, sisi)).sum(axis=0) // 8 % 2
    papan = np.stack([kotak * 255] * 3, axis=-1).astype(np.uint8)
    a, b = tmp_path / "seragam.png", tmp_path / "papan.png"
    simpan(a, seragam)
    simpan(b, papan)

    va, vb = extract(a, cfg), extract(b, cfg)
    contrast, _, energy, _ = va[16:20]
    assert vb[16] > contrast + 1e-6        # contrast papan catur lebih tinggi
    assert vb[18] < energy - 1e-6          # energy papan catur lebih rendah


def test_rotasi_90_derajat_tidak_mengubah_fitur(cfg, tmp_path):
    """Rata-rata 4 sudut + ROI persegi membuat vektor 20-D tahan rotasi 90 derajat."""
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 256, (256, 256, 3))
    a, b = tmp_path / "asli.png", tmp_path / "putar.png"
    simpan(a, arr)
    simpan(b, np.rot90(arr))
    assert np.allclose(extract(a, cfg), extract(b, cfg), atol=1e-6)