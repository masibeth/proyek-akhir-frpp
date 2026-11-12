"""F5c: dominasi modalitas — block-permutation importance pada data uji.

Seluruh kolom satu modalitas dipermutasi BERSAMA-SAMA dengan urutan baris yang sama,
sehingga korelasi internal blok (mis. antar-bin HSV) tetap utuh. Permutasi per-fitur
dihitung terpisah sebagai pelengkap, lalu arah blok dibandingkan dengan selisih
drop-one dari F5 (results/paired_delta.csv).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.load_data import ROOT, read_config
from src.protocol import feature_matrix, iter_folds, make_pipeline, visual_columns

KEY = ["model", "target"]


def permute_block(x: np.ndarray, idx: list[int], rng) -> np.ndarray:
    out = x.copy()
    order = rng.permutation(x.shape[0])
    out[:, idx] = x[order][:, idx]
    return out


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, average="macro"))


def drops(pipe, x: np.ndarray, y: np.ndarray, idx: list[int], repeat: int, rng, baseline: float) -> list[float]:
    return [baseline - macro_f1(y, pipe.predict(permute_block(x, idx, rng))) for _ in range(repeat)]


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    per_seed = rows.groupby(KEY + ["seed"], as_index=False)["drop_mean"].mean()
    agg = per_seed.groupby(KEY)["drop_mean"].agg(["mean", "std"])
    agg.columns = ["drop_mean", "drop_sd"]
    agg["n_seed"] = per_seed.groupby(KEY)["seed"].nunique()
    return agg.reset_index()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = read_config()
    pos = cfg["metrics"]["positive_label"]
    repeat = cfg["importance"]["block_repeat"]
    visual = pd.read_parquet(ROOT / cfg["paths"]["cache_visual"])
    sensor = pd.read_parquet(ROOT / cfg["paths"]["cache_sensor"])
    v_cols = visual_columns(visual)
    s_cols = cfg["sensor"]["channels"]
    blocks = {"visual": list(range(len(v_cols))), "sensor": list(range(len(v_cols), len(v_cols) + len(s_cols)))}
    semua = v_cols + s_cols

    rows_block: list[dict] = []
    rows_fitur: list[dict] = []
    mulai = time.perf_counter()
    for seed, fold, tr, te in iter_folds(cfg, visual, sensor):
        x_tr = feature_matrix(tr, "fused", cfg)
        x_te = feature_matrix(te, "fused", cfg)
        y_tr = (tr["label"].to_numpy() == pos).astype(int)
        y_te = (te["label"].to_numpy() == pos).astype(int)
        for model in cfg["models"]:
            pipe = make_pipeline(cfg, model).fit(x_tr, y_tr)
            baseline = macro_f1(y_te, pipe.predict(x_te))
            rng = np.random.default_rng(seed * 1000 + fold)
            for name, idx in blocks.items():
                d = drops(pipe, x_te, y_te, idx, repeat, rng, baseline)
                rows_block.append(
                    {"seed": seed, "fold": fold, "model": model, "target": name,
                     "baseline": baseline, "drop_mean": float(np.mean(d)), "drop_sd": float(np.std(d, ddof=1))}
                )
            for j, name in enumerate(semua):
                d = drops(pipe, x_te, y_te, [j], repeat, rng, baseline)
                rows_fitur.append(
                    {"seed": seed, "fold": fold, "model": model, "target": name,
                     "baseline": baseline, "drop_mean": float(np.mean(d)), "drop_sd": float(np.std(d, ddof=1))}
                )
        print(f"  seed {seed} fold {fold}: {len(cfg['models'])} model x ({len(blocks)} blok + {len(semua)} fitur) x {repeat} ulang | {time.perf_counter() - mulai:.1f}s", flush=True)

    blok = pd.DataFrame(rows_block)
    fitur = pd.DataFrame(rows_fitur)
    if blok[["drop_mean"]].isna().any().any() or fitur[["drop_mean"]].isna().any().any():
        print("ada NaN pada importance")
        return 2

    ringkas_blok = summarize(blok)
    ringkas_fitur = summarize(fitur)
    out_dir = ROOT / cfg["paths"]["results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    blok.to_csv(out_dir / "importance.csv", index=False)
    fitur.to_csv(out_dir / "importance_per_feature.csv", index=False)
    ringkas_blok.to_csv(out_dir / "importance_summary.csv", index=False)

    print(f"\n=== penurunan macro-F1 per blok (mean±sd antar seed, {repeat} permutasi/uji) ===")
    for model in cfg["models"]:
        s = ringkas_blok[ringkas_blok["model"] == model].set_index("target")
        print(
            f"  {model:14s} blok visual {s.loc['visual', 'drop_mean']:+.4f} ± {s.loc['visual', 'drop_sd']:.4f} "
            f"| blok sensor {s.loc['sensor', 'drop_mean']:+.4f} ± {s.loc['sensor', 'drop_sd']:.4f}"
        )

    top = []
    print(f"\n=== top-10 per fitur (mean penurunan macro-F1 antar seed) ===")
    for model in cfg["models"]:
        s = ringkas_fitur[ringkas_fitur["model"] == model].sort_values("drop_mean", ascending=False).head(10)
        top.append(s.assign(peringkat=range(1, len(s) + 1)))
        print(f"  {model}: " + ", ".join(f"{r.target} {r.drop_mean:+.4f}" for _, r in s.iterrows()))
    pd.concat(top).to_csv(out_dir / "importance_top10.csv", index=False)

    delta_path = out_dir / "paired_delta.csv"
    if delta_path.is_file():
        delta = pd.read_csv(delta_path).groupby("model")[["delta_vs_visual", "delta_vs_sensor"]].mean()
        print("\n=== konsistensi arah dengan drop-one F5 ===")
        print("  (drop-one: delta_vs_sensor = kontribusi visual; delta_vs_visual = kontribusi sensor)")
        for model in cfg["models"]:
            s = ringkas_blok[ringkas_blok["model"] == model].set_index("target")
            pv, ps = s.loc["visual", "drop_mean"], s.loc["sensor", "drop_mean"]
            dv, ds = delta.loc[model, "delta_vs_sensor"], delta.loc[model, "delta_vs_visual"]
            setuju = (pv > ps) == (dv > ds)
            print(
                f"  {model:14s} permutasi visual {pv:+.4f} vs sensor {ps:+.4f} | "
                f"drop-one visual {dv:+.4f} vs sensor {ds:+.4f} -> {'searah' if setuju else 'BERTENTANGAN'}"
            )
            if not setuju:
                print("    tulis kedua angka di laporan dan jelaskan; jangan pilih yang menguntungkan.")
    else:
        print("paired_delta.csv tidak ada — jalankan F5 dulu untuk konfirmasi drop-one")

    print(f"\nhasil tertulis di {out_dir.relative_to(ROOT)}: importance.csv, importance_per_feature.csv, importance_summary.csv, importance_top10.csv")
    print(f"total waktu {time.perf_counter() - mulai:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())