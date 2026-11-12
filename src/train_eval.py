"""F5: evaluasi komparatif 3 set fitur x 4 model x 25 kombinasi (5 seed x 5 fold).

Satu-satunya jalur pelatihan adalah protocol.iter_folds + make_pipeline, sehingga
setiap set fitur memakai baris berpasangan yang sama persis. Varian tambahan sesuai
rencana §3.3: tanpa penskalaan gabungan untuk KNN dan SVM-RBF pada fused.

Keluaran: results/metrics.csv (satu baris per fit), results/summary.csv (mean±sd
antar seed pairing), results/confusion.csv (teragregasi atas fold),
results/paired_delta.csv (selisih macro-F1 fused vs unimodal per seed).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.load_data import ROOT, read_config
from src.protocol import feature_matrix, iter_folds, make_pipeline

METRICS = ["accuracy", "macro_f1", "precision_macro", "recall_macro", "roc_auc"]
KEY = ["feature_set", "model", "scaling"]


def build_grid(cfg: dict) -> list[tuple[str, str, str]]:
    sets = cfg["fusion"]["concat_order"] + ["fused"]
    default = cfg["fusion"]["post_concat_scaling"]
    distance = set(cfg["fusion"]["distance_models"])
    variants = cfg["fusion"]["distance_scaling_variants"]
    grid = []
    for fs in sets:
        for model in cfg["models"]:
            scalings = variants if fs == "fused" and model in distance else [default]
            grid.extend((fs, model, sc) for sc in scalings)
    return grid


def decision_score(pipe, x: np.ndarray) -> np.ndarray:
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(x)
    return pipe.predict_proba(x)[:, 1]


def measure(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, score),
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata over fold dulu di dalam tiap seed, baru mean±sd antar seed."""
    per_seed = metrics.groupby(KEY + ["seed"], as_index=False)[METRICS].mean()
    agg = per_seed.groupby(KEY)[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seed"] = per_seed.groupby(KEY)["seed"].nunique()
    return agg.reset_index()


def paired_delta(metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    default = cfg["fusion"]["post_concat_scaling"]
    per_seed = metrics.groupby(KEY + ["seed"], as_index=False)[METRICS].mean()
    per_seed = per_seed[per_seed["scaling"] == default]
    wide = per_seed.pivot(index="seed", columns=["feature_set", "model"], values="macro_f1")
    rows = []
    for model in sorted(wide.columns.get_level_values("model").unique()):
        for seed in wide.index:
            rows.append(
                {
                    "seed": int(seed),
                    "model": model,
                    "fused": wide.loc[seed, ("fused", model)],
                    "delta_vs_visual": wide.loc[seed, ("fused", model)] - wide.loc[seed, ("visual", model)],
                    "delta_vs_sensor": wide.loc[seed, ("fused", model)] - wide.loc[seed, ("sensor", model)],
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = read_config()
    pos = cfg["metrics"]["positive_label"]
    visual = pd.read_parquet(ROOT / cfg["paths"]["cache_visual"])
    sensor = pd.read_parquet(ROOT / cfg["paths"]["cache_sensor"])
    grid = build_grid(cfg)
    n_combo = len(cfg["protocol"]["pairing_seeds"]) * cfg["protocol"]["n_folds"]
    print(f"grid: {len(grid)} konfigurasi x {n_combo} kombinasi = {len(grid) * n_combo} fit", flush=True)

    rows: list[dict] = []
    cm: dict[tuple, np.ndarray] = {}
    mulai = time.perf_counter()
    for seed, fold, tr, te in iter_folds(cfg, visual, sensor):
        y_tr = (tr["label"].to_numpy() == pos).astype(int)
        y_te = (te["label"].to_numpy() == pos).astype(int)
        for fs, model, scaling in grid:
            x_tr = feature_matrix(tr, fs, cfg)
            x_te = feature_matrix(te, fs, cfg)
            pipe = make_pipeline(cfg, model, scaling=scaling)
            t0 = time.perf_counter()
            pipe.fit(x_tr, y_tr)
            pred = pipe.predict(x_te)
            detik = time.perf_counter() - t0
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "feature_set": fs,
                    "model": model,
                    "scaling": scaling,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "detik": round(detik, 4),
                    **measure(y_te, pred, decision_score(pipe, x_te)),
                }
            )
            key = (fs, model, scaling)
            cm[key] = cm.get(key, np.zeros((2, 2), dtype=int)) + confusion_matrix(y_te, pred, labels=[0, 1])
        print(
            f"  seed {seed} fold {fold}: {len(grid)} fit | latih {len(tr)} uji {len(te)} "
            f"| total {time.perf_counter() - mulai:.1f}s",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    wajib = KEY + ["seed", "fold", "n_train", "n_test", "detik"] + METRICS
    if set(metrics.columns) != set(wajib):
        print("kolom metrics.csv tidak sesuai:", list(metrics.columns))
        return 2
    if len(metrics) != len(grid) * n_combo:
        print(f"jumlah baris {len(metrics)} != seharusnya {len(grid) * n_combo}")
        return 2
    if metrics[METRICS].isna().any().any():
        print("ada NaN pada metrik")
        return 2

    summary = summarize(metrics)
    delta = paired_delta(metrics, cfg)
    confusion = pd.DataFrame(
        [{"feature_set": k[0], "model": k[1], "scaling": k[2], "tn": int(v[0, 0]), "fp": int(v[0, 1]),
          "fn": int(v[1, 0]), "tp": int(v[1, 1])} for k, v in sorted(cm.items())]
    )

    out_dir = ROOT / cfg["paths"]["results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    confusion.to_csv(out_dir / "confusion.csv", index=False)
    delta.to_csv(out_dir / "paired_delta.csv", index=False)

    inti = metrics[metrics["scaling"] == cfg["fusion"]["post_concat_scaling"]]
    print(f"\nmetrics.csv: {len(metrics)} baris ({len(inti)} inti + {len(metrics) - len(inti)} varian penskalaan)")
    print("\n=== ringkasan macro-F1 mean±sd antar seed (5 seed) ===")
    for _, r in summary.iterrows():
        print(f"  {r.feature_set:7s} {r.model:14s} {r.scaling:8s} {r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f}")

    print("\n=== sanity: fused vs unimodal (macro-F1, penskalaan standar) ===")
    rusak = []
    for model in cfg["models"]:
        s = summary[(summary["model"] == model) & (summary["scaling"] == cfg["fusion"]["post_concat_scaling"])]
        nilai = {r.feature_set: r.macro_f1_mean for _, r in s.iterrows()}
        fused, vis, sen = nilai["fused"], nilai["visual"], nilai["sensor"]
        tanda = "OK" if fused >= max(vis, sen) else ("di bawah keduanya" if fused < min(vis, sen) else "di antara")
        if fused < min(vis, sen):
            rusak.append(model)
        print(f"  {model:14s} fused {fused:.4f} | visual {vis:.4f} | sensor {sen:.4f} -> {tanda}")
    if len(rusak) == len(cfg["models"]):
        print("PERINGATAN: fused di bawah kedua unimodal pada semua model — curigai penskalaan blok (§3.3), bukan langsung disimpulkan.")

    print("\n=== selisih macro-F1 fused - unimodal, mean±sd antar seed ===")
    for model, g in delta.groupby("model", sort=False):
        print(
            f"  {model:14s} vs visual {g.delta_vs_visual.mean():+.4f} ± {g.delta_vs_visual.std():.4f} "
            f"| vs sensor {g.delta_vs_sensor.mean():+.4f} ± {g.delta_vs_sensor.std():.4f}"
        )

    print(f"\nhasil tertulis di {out_dir.relative_to(ROOT)}: metrics.csv, summary.csv, confusion.csv, paired_delta.csv")
    print(f"total waktu {time.perf_counter() - mulai:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())