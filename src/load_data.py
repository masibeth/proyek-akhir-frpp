"""Audit data mentah kedua modalitas dan tulis data/manifest.json.

Kode keluar: 0 = kedua modalitas ada dan lolos kriteria; 2 = ada modalitas
yang hadir tetapi gagal kriteria; 3 = ada modalitas yang belum diunduh.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sensor(cfg: dict) -> dict:
    """Baca CSV sensor, agregasi ke rata-rata per menit, dan kumpulkan masalah."""
    scfg = cfg["sensor"]
    raw_dir = ROOT / cfg["paths"]["raw_sensor"]
    files = sorted(raw_dir.glob("*.csv"))
    out: dict = {"status": "missing", "issues": []}
    if not files:
        return out
    if len(files) > 1:
        out["issues"].append(f"lebih dari satu CSV di {raw_dir}: {[f.name for f in files]}")

    path = files[0]
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    out["status"] = "ok"
    out["file"] = path.name
    out["sha256"] = sha256_file(path)
    out["rows_raw"] = int(len(raw))
    out["columns"] = list(raw.columns)

    channels = scfg["channels"]
    missing_ch = [c for c in channels if c not in raw.columns]
    if missing_ch:
        out["issues"].append(f"kanal tidak ada di CSV: {missing_ch}")
        return out

    publisher = scfg["source"].get("publisher_sha256")
    out["sha256_cocok_penerbit"] = publisher is None or out["sha256"] == publisher
    if not out["sha256_cocok_penerbit"]:
        out["issues"].append("sha256 unduhan berbeda dari hash penerbit")

    if raw[channels].isna().any().any():
        out["issues"].append("ada nilai hilang pada kanal gas")

    label_col = "class"
    fresh_ids = scfg["class_map"]["fresh"]
    spoiled_ids = scfg["class_map"]["spoiled"]
    known = set(fresh_ids) | set(spoiled_ids)
    unknown = sorted(set(raw[label_col].unique()) - known)
    if unknown:
        out["issues"].append(f"kelas di luar class_map: {unknown}")

    raw["label"] = raw[label_col].map(
        lambda v: "fresh" if v in fresh_ids else ("spoiled" if v in spoiled_ids else None)
    )
    out["counts_raw"] = {k: int(v) for k, v in raw["label"].value_counts().items()}

    mixed = raw.groupby("minutes")["label"].nunique()
    if (mixed > 1).any():
        out["issues"].append("ada menit dengan lebih dari satu kelas; agregasi tidak sah")
        return out

    agg = raw.groupby("minutes", as_index=False).agg(
        {**{c: "mean" for c in channels}, "label": "first"}
    )
    block = scfg["block_minutes"]
    agg["group_id"] = (agg["minutes"] - agg["minutes"].min()) // block
    out["rows_aggregated"] = int(len(agg))
    out["counts_aggregated"] = {k: int(v) for k, v in agg["label"].value_counts().items()}
    out["groups_per_class"] = {
        k: int(v) for k, v in agg.groupby("label")["group_id"].nunique().items()
    }
    out["minutes_range"] = [int(agg["minutes"].min()), int(agg["minutes"].max())]
    out["block_minutes"] = block
    out["channels"] = channels
    out["excluded_columns"] = scfg["excluded_columns"]
    out["class_map"] = scfg["class_map"]
    out["source"] = scfg["source"]

    if set(out["counts_aggregated"]) != {"fresh", "spoiled"}:
        out["issues"].append("kelas biner tidak lengkap setelah agregasi")
    out["frame"] = agg
    return out


def _visual_timestamp(name: str, vcfg: dict) -> datetime | None:
    m = re.match(vcfg["timestamp"]["pattern"], name)
    if m is None:
        return None
    return datetime.strptime("".join(m.groups()), vcfg["timestamp"]["format"])


def load_visual(cfg: dict) -> dict:
    """Audit citra: timestamp dari nama berkas, blok waktu, dan pembuangan blok batas kelas."""
    vcfg = cfg["visual"]
    raw_dir = ROOT / cfg["paths"]["raw_visual"]
    out: dict = {"status": "missing", "issues": []}
    if not raw_dir.is_dir():
        return out

    wanted = {v.lower(): k for k, v in vcfg["class_dirs"].items()}
    dirs = {}
    for sub in sorted(raw_dir.iterdir()):
        if sub.is_dir() and sub.name.lower() in wanted:
            dirs[wanted[sub.name.lower()]] = sub
    if not dirs:
        nested = [p for p in raw_dir.iterdir() if p.is_dir()]
        if len(nested) == 1:
            for sub in sorted(nested[0].iterdir()):
                if sub.is_dir() and sub.name.lower() in wanted:
                    dirs[wanted[sub.name.lower()]] = sub
    if not dirs:
        return out

    rows: list[dict] = []
    ignored: list[str] = []
    counts: dict[str, int] = {}
    exif: dict[str, str] = {}
    for label, sub in sorted(dirs.items()):
        imgs = sorted(p for p in sub.rglob("*") if p.suffix.lower() in IMAGE_EXT)
        counts[label] = len(imgs)
        if not imgs:
            out["issues"].append(f"folder kelas {label} kosong")
        with_dt = 0
        for p in imgs[:10]:
            try:
                from PIL import Image

                e = Image.open(p).getexif()
                if e.get(36867) or e.get(306):
                    with_dt += 1
            except Exception:
                continue
        exif[label] = f"{with_dt}/{min(10, len(imgs))} sampel punya EXIF datetime"
        for p in imgs:
            ts = _visual_timestamp(p.name, vcfg)
            if ts is None:
                ignored.append(p.relative_to(ROOT).as_posix())
                continue
            rows.append({"label": label, "path": p.relative_to(ROOT).as_posix(), "timestamp": ts})

    out["status"] = "ok"
    out["counts"] = counts
    out["exif_datetime"] = exif
    out["ignored_count"] = len(ignored)
    out["ignored_files"] = ignored[:20]
    if ignored:
        out["issues"].append(f"{len(ignored)} berkas tidak cocok pola timestamp dan diabaikan")
    if not rows:
        out["issues"].append("tidak ada citra dengan timestamp terbaca")
        return out

    frame = pd.DataFrame(rows).sort_values(["timestamp", "path"]).reset_index(drop=True)
    frame["visual_id"] = frame.index.astype(int)

    gap = vcfg["timestamp"]["session_gap_minutes"]
    trajectory = {}
    for label, g in frame.groupby("label", sort=True):
        ts = g["timestamp"]
        deltas = ts.diff().dt.total_seconds().div(60).dropna()
        trajectory[label] = {
            "n": int(len(g)),
            "mulai": ts.min().isoformat(sep=" "),
            "selesai": ts.max().isoformat(sep=" "),
            "interval_menit_modus": float(deltas.mode().iloc[0]) if len(deltas) else None,
            "sesi": int((deltas > gap).sum()) + 1,
        }
    out["trajectory"] = trajectory

    block = vcfg["group_minutes"]
    frame["block_in_class"] = 0
    for label, g in frame.groupby("label"):
        elapsed = (g["timestamp"] - g["timestamp"].min()).dt.total_seconds() / 60
        frame.loc[g.index, "block_in_class"] = (elapsed // block).astype(int)

    rule = vcfg["boundary"]["rule"]
    boundary: dict = {"rule": rule, "dibuang": 0}
    order = frame.groupby("label")["timestamp"].min().sort_values().index.tolist()
    if len(order) == 2:
        awal, akhir = order
        t_awal = frame.loc[frame["label"] == awal, "timestamp"].max()
        t_akhir = frame.loc[frame["label"] == akhir, "timestamp"].min()
        boundary["kelas_awal"] = awal
        boundary["kelas_akhir"] = akhir
        boundary["frame_terakhir_kelas_awal"] = t_awal.isoformat(sep=" ")
        boundary["frame_pertama_kelas_akhir"] = t_akhir.isoformat(sep=" ")
        boundary["jeda_menit"] = round((t_akhir - t_awal).total_seconds() / 60, 2)
        if rule == "drop_adjacent_blocks":
            if t_awal >= t_akhir:
                out["issues"].append("kelas tidak terpisah waktu; blok batas tidak dibuang")
            else:
                b_awal = int(frame.loc[frame["label"] == awal, "block_in_class"].max())
                mask = ((frame["label"] == awal) & (frame["block_in_class"] == b_awal)) | (
                    (frame["label"] == akhir) & (frame["block_in_class"] == 0)
                )
                boundary["blok_dibuang"] = {awal: b_awal, akhir: 0}
                boundary["dibuang"] = int(mask.sum())
                boundary["dibuang_per_kelas"] = {
                    k: int(v) for k, v in frame.loc[mask, "label"].value_counts().items()
                }
                frame = frame.loc[~mask].reset_index(drop=True)
    out["boundary"] = boundary

    keys = list(dict.fromkeys(zip(frame["label"], frame["block_in_class"])))
    nomor = {k: i for i, k in enumerate(sorted(keys))}
    frame["group_id"] = [nomor[k] for k in zip(frame["label"], frame["block_in_class"])]

    out["rows_used"] = int(len(frame))
    out["counts_used"] = {k: int(v) for k, v in frame["label"].value_counts().items()}
    out["groups_per_class"] = {
        k: int(v) for k, v in frame.groupby("label")["group_id"].nunique().items()
    }
    out["block_minutes"] = block
    out["sample_names"] = {
        label: [Path(p).name for p in g["path"].head(5)] for label, g in frame.groupby("label")
    }
    out["source"] = vcfg["source"]

    if set(out["counts_used"]) != {"fresh", "spoiled"}:
        out["issues"].append(f"kelas visual tidak lengkap: {sorted(out['counts_used'])}")
    n_fold = cfg["protocol"]["n_folds"]
    for label, n in out["groups_per_class"].items():
        if n < n_fold:
            out["issues"].append(f"blok {label} = {n} < jumlah fold {n_fold}")
    out["frame"] = frame
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = read_config()
    sensor = load_sensor(cfg)
    visual = load_visual(cfg)

    print("=== SENSOR (e-nose) ===")
    if sensor["status"] == "missing":
        print("BELUM DIUNDUH: letakkan CSV Mendeley di", cfg["paths"]["raw_sensor"])
    else:
        print(f"berkas          : {sensor['file']}")
        print(f"sha256 cocok    : {sensor['sha256_cocok_penerbit']}")
        print(f"baris mentah    : {sensor['rows_raw']}  {sensor.get('counts_raw', {})}")
        print(f"baris agregasi  : {sensor['rows_aggregated']}  {sensor.get('counts_aggregated', {})}")
        print(f"blok per kelas  : {sensor.get('groups_per_class', {})} (blok {sensor.get('block_minutes')} menit)")
        print(f"rentang menit   : {sensor.get('minutes_range')}")
        print(f"kanal dipakai   : {len(sensor.get('channels', []))} -> {sensor.get('channels')}")
    print()
    print("=== VISUAL (citra) ===")
    if visual["status"] == "missing":
        print("BELUM DIUNDUH: letakkan folder kelas di", cfg["paths"]["raw_visual"])
    else:
        print(f"jumlah ditemukan: {visual['counts']}")
        for label, t in visual.get("trajectory", {}).items():
            print(
                f"  {label}: {t['n']} frame, {t['mulai']} -> {t['selesai']}, "
                f"interval {t['interval_menit_modus']} menit, {t['sesi']} sesi"
            )
        b = visual.get("boundary", {})
        if b:
            print(
                f"batas kelas     : {b.get('frame_terakhir_kelas_awal')} ({b.get('kelas_awal')}) -> "
                f"{b.get('frame_pertama_kelas_akhir')} ({b.get('kelas_akhir')}), jeda {b.get('jeda_menit')} menit"
            )
            print(f"blok dibuang    : {b.get('blok_dibuang', {})} -> {b.get('dibuang', 0)} frame")
        print(f"jumlah dipakai  : {visual.get('rows_used')}  {visual.get('counts_used', {})}")
        print(
            f"blok per kelas  : {visual.get('groups_per_class', {})} "
            f"(blok {visual.get('block_minutes')} menit)"
        )
        for label, names in visual.get("sample_names", {}).items():
            print(f"  contoh {label}: {names}")
        print(f"EXIF datetime   : {visual['exif_datetime']}")
        if visual.get("ignored_count"):
            print(f"diabaikan       : {visual['ignored_count']} berkas (pola timestamp tidak cocok)")
    print()

    issues = sensor["issues"] + visual["issues"]
    for msg in issues:
        print("MASALAH:", msg)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_sha256": sha256_file(ROOT / "config.yaml"),
        "sensor": {k: v for k, v in sensor.items() if k != "frame"},
        "visual": {k: v for k, v in visual.items() if k != "frame"},
        "keputusan": {
            "class_map_sensor": cfg["sensor"]["class_map_reason"],
            "agregasi_sensor": "rata-rata per menit; split blok waktu, bukan baris acak",
            "grup_visual": (
                f"blok {cfg['visual']['group_minutes']} menit dari timestamp nama berkas; "
                f"batas kelas: {cfg['visual']['boundary']['rule']}"
            ),
            "dimensi_fused": cfg["fusion"]["expected_dim"],
        },
    }
    manifest_path = ROOT / cfg["paths"]["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest tertulis:", manifest_path.relative_to(ROOT))

    if issues:
        return 2
    if sensor["status"] == "missing" or visual["status"] == "missing":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())