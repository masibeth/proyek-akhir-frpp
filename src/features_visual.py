"""F2: fitur visual 20-D (16 histogram HSV + 4 GLCM) dan cache parquet.

Urutan pra-pemrosesan beku (rencana §3.1): resize 256x256 -> center crop 80%
-> Gaussian sigma 1.0 -> HSV. Urutan fitur: f00-f07 H, f08-f11 S, f12-f15 V,
f16-f19 contrast/homogeneity/energy/correlation.

Cache menyimpan fitur apa adanya; penskalaan antar-blok hidup di fold latih (protocol.py).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian

from src.load_data import ROOT, load_visual, read_config

RESAMPLE = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "lanczos": Image.LANCZOS}


def preprocess(img: Image.Image, vcfg: dict) -> np.ndarray:
    w, h = vcfg["resize"]
    img = img.convert("RGB").resize((w, h), RESAMPLE[vcfg["resize_resample"]])
    arr = np.asarray(img, dtype=np.float64) / 255.0
    cw, ch = int(w * vcfg["roi_center_fraction"]), int(h * vcfg["roi_center_fraction"])
    left, top = (w - cw) // 2, (h - ch) // 2
    roi = arr[top : top + ch, left : left + cw]
    roi = gaussian(roi, sigma=vcfg["gaussian_sigma"], channel_axis=-1, preserve_range=True)
    return np.clip(roi, 0.0, 1.0)


def extract(path, cfg: dict) -> np.ndarray:
    vcfg = cfg["visual"]
    with Image.open(path) as img:
        roi = preprocess(img, vcfg)

    hsv = rgb2hsv(roi)
    blocks = []
    for i, channel in enumerate(("h", "s", "v")):
        hist, _ = np.histogram(hsv[..., i], bins=vcfg["hist_bins"][channel], range=(0.0, 1.0))
        hist = hist.astype(float)
        blocks.append(hist / hist.sum())

    gcfg = vcfg["glcm"]
    levels = gcfg["levels"]
    gray = rgb2gray(roi)
    quantized = np.clip((gray * levels).astype(np.int64), 0, levels - 1).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=gcfg["distances"],
        angles=[np.deg2rad(a) for a in gcfg["angles_deg"]],
        levels=levels,
        symmetric=gcfg["symmetric"],
        normed=gcfg["normed"],
    )
    for prop in gcfg["properties"]:
        blocks.append(np.array([graycoprops(glcm, prop).mean()]))

    return np.concatenate(blocks)


def feature_columns(dim: int) -> list[str]:
    return [f"f{i:02d}" for i in range(dim)]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = read_config()
    visual = load_visual(cfg)
    if visual["status"] != "ok" or visual["issues"]:
        print("visual belum lolos audit:", visual["issues"] or visual["status"])
        return 2

    frame = visual["frame"]
    dim = cfg["visual"]["expected_dim"]
    cols = feature_columns(dim)
    rows, failed = [], []
    for n, r in enumerate(frame.itertuples(index=False), start=1):
        try:
            vec = extract(ROOT / r.path, cfg)
            if vec.shape != (dim,) or not np.isfinite(vec).all():
                raise ValueError(f"bentuk/nilai tidak sah: {vec.shape}")
        except Exception as exc:
            failed.append((r.path, repr(exc)))
            continue
        rows.append({"visual_id": r.visual_id, "label": r.label, "group_id": r.group_id, "path": r.path, **dict(zip(cols, vec))})
        if n % 200 == 0:
            print(f"  {n}/{len(frame)} citra diproses...", flush=True)

    if failed:
        print(f"GAGAL {len(failed)} citra (tidak dimasukkan ke cache):")
        for path, err in failed[:10]:
            print("  ", path, err)

    if not rows:
        print("tidak ada fitur yang berhasil dihitung")
        return 2

    out = pd.DataFrame(rows)
    if out[cols].isna().any().any():
        print("ada NaN pada fitur")
        return 2
    constant = [c for c in cols if out[c].nunique() <= 1]
    if constant:
        print("kolom konstan:", constant)
        return 2
    awal = 0
    for channel in ("h", "s", "v"):
        lebar = cfg["visual"]["hist_bins"][channel]
        jumlah = out[cols[awal : awal + lebar]].to_numpy().sum(axis=1)
        if not np.allclose(jumlah, 1.0, atol=1e-6):
            print(f"histogram {channel.upper()} tidak berjumlah 1 (simpangan maks {np.abs(jumlah - 1).max():.2e})")
            return 2
        awal += lebar

    path = ROOT / cfg["paths"]["cache_visual"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    print(f"fitur visual: {len(out)} baris x {len(cols)} kolom -> {path.relative_to(ROOT)}")
    print("per kelas:", out["label"].value_counts().to_dict())
    print("blok waktu per kelas:", out.groupby("label")["group_id"].nunique().to_dict())
    print("rentang fitur: min", float(out[cols].to_numpy().min()), "maks", float(out[cols].to_numpy().max()))
    print("catatan: penskalaan antar-blok dilakukan di dalam fold latih, bukan di cache")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())