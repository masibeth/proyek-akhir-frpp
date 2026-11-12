"""F3: cache kanal sensor teragregasi per menit.

Yang di-cache adalah nilai kanal mentah teragregasi, BUKAN fitur terskala:
clip outlier, imputasi, dan Z-score wajib fit di fold latih (hidup di protocol.py).
"""

from __future__ import annotations

import sys

import pandas as pd

from src.load_data import ROOT, load_sensor, read_config


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = read_config()
    sensor = load_sensor(cfg)
    if sensor["status"] != "ok" or sensor["issues"]:
        print("sensor belum lolos audit:", sensor["issues"] or sensor["status"])
        return 2

    channels = cfg["sensor"]["channels"]
    if len(channels) != cfg["sensor"]["expected_dim"]:
        print("jumlah kanal tidak sama dengan expected_dim")
        return 2

    agg = sensor["frame"]
    out = agg[["minutes", "group_id", "label"] + channels].rename(columns={"minutes": "sensor_id"})

    if out[channels].isna().any().any():
        print("ada NaN pada kanal")
        return 2
    constant = [c for c in channels if out[c].nunique() <= 1]
    if constant:
        print("kolom konstan:", constant)
        return 2

    path = ROOT / cfg["paths"]["cache_sensor"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)

    print(f"fitur sensor: {len(out)} baris x {len(channels)} kanal -> {path.relative_to(ROOT)}")
    print("per kelas:", out["label"].value_counts().to_dict())
    print("blok waktu per kelas:", out.groupby("label")["group_id"].nunique().to_dict())
    print("catatan: penskalaan/clip/imputasi dilakukan di dalam fold latih, bukan di cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())