"""F5b: probe pasangan diskordan — HANYA uji, tidak pernah masuk data latih.

Pasangan visual segar <-> sensor busuk (dan sebaliknya) dibuat dari baris split uji
fold yang sama, lalu diuji dengan model yang dilatih pada pasangan konkordan fold itu.
Tidak ada ground truth untuk pasangan diskordan, maka akurasi TIDAK dihitung:
kolom `label` sengaja dibuang dari frame diskordan.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.load_data import ROOT, read_config
from src.protocol import _join, feature_matrix, fold_splits, iter_folds, make_pipeline

KEY = ["direction", "model"]


def cross_pairs(v_labels: np.ndarray, s_labels: np.ndarray, want_v: str, want_s: str, rng) -> list[tuple]:
    iv = np.flatnonzero(v_labels == want_v)
    is_ = np.flatnonzero(s_labels == want_s)
    n = min(len(iv), len(is_))
    rng.shuffle(iv)
    rng.shuffle(is_)
    return list(zip(iv[:n], is_[:n]))


def discordant_frames(cfg: dict, visual: pd.DataFrame, sensor: pd.DataFrame, v_te, s_te, rng) -> dict:
    """Arah -> frame diskordan tanpa kolom label, seluruhnya dari split uji."""
    display = cfg["labels"]["visual"]
    inverse = {v: k for k, v in display.items()}
    v_lab = visual["label"].to_numpy()
    s_lab = sensor["label"].to_numpy()
    out = {}
    for v_disp, s_disp in cfg["probe"]["discordant_pairs"]:
        want_v, want_s = inverse[v_disp], inverse[s_disp]
        pairs = cross_pairs(v_lab[v_te], s_lab[s_te], want_v, want_s, rng)
        frame = _join(visual, sensor, v_te, s_te, pairs, cfg).drop(columns=["label"])
        out[f"{v_disp}-{s_disp}"] = frame
    return out


def confidence(pipe, x: np.ndarray) -> tuple[float, str]:
    if hasattr(pipe, "predict_proba"):
        return float(pipe.predict_proba(x).max(axis=1).mean()), "max_proba"
    return float(np.abs(pipe.decision_function(x)).mean()), "abs_margin"


def measure_probe(pred: np.ndarray, v_label: np.ndarray, s_label: np.ndarray, pos: str) -> dict:
    return {
        "agreement_visual": float((pred == (v_label == pos).astype(int)).mean()),
        "agreement_sensor": float((pred == (s_label == pos).astype(int)).mean()),
        "share_predicted_busuk": float(pred.mean()),
    }


def summarize(probe: pd.DataFrame, nilai: list[str]) -> pd.DataFrame:
    per_seed = probe.groupby(KEY + ["seed"], as_index=False)[nilai].mean()
    agg = per_seed.groupby(KEY)[nilai].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seed"] = per_seed.groupby(KEY)["seed"].nunique()
    agg["n_pairs"] = probe.groupby(KEY)["n_pairs"].sum()
    return agg.reset_index()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = read_config()
    pos = cfg["metrics"]["positive_label"]
    visual = pd.read_parquet(ROOT / cfg["paths"]["cache_visual"])
    sensor = pd.read_parquet(ROOT / cfg["paths"]["cache_sensor"])
    splits = fold_splits(cfg, visual, sensor)
    seeds = cfg["protocol"]["pairing_seeds"]
    nilai = ["agreement_visual", "agreement_sensor", "share_predicted_busuk", "confidence"]

    rows: list[dict] = []
    for (seed, fold, tr, te), (v_tr, v_te, s_tr, s_te) in zip(
        iter_folds(cfg, visual, sensor), splits * len(seeds)
    ):
        if not set(tr["visual_id"]) <= set(visual.iloc[v_tr]["visual_id"]):
            print("frame latih tidak cocok dengan split latih")
            return 2
        rng = np.random.default_rng(seed * 1000 + fold)
        discs = discordant_frames(cfg, visual, sensor, v_te, s_te, rng)
        y_tr = (tr["label"].to_numpy() == pos).astype(int)
        x_tr = feature_matrix(tr, "fused", cfg)
        for model in cfg["models"]:
            pipe = make_pipeline(cfg, model).fit(x_tr, y_tr)
            for direction, frame in discs.items():
                if len(frame) == 0:
                    print(f"arah {direction} kosong pada seed {seed} fold {fold}")
                    return 2
                if (frame["v_label"] == frame["s_label"]).any():
                    print("ada pasangan diskordan berlabel sama")
                    return 2
                if set(frame["visual_id"]) & set(tr["visual_id"]) or set(frame["sensor_id"]) & set(tr["sensor_id"]):
                    print("PASANGAN LATIH TERPAKAI DI PROBE — protokol batal")
                    return 2
                x = feature_matrix(frame, "fused", cfg)
                pred = pipe.predict(x)
                conf, kind = confidence(pipe, x)
                rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "direction": direction,
                        "model": model,
                        "n_pairs": len(frame),
                        "confidence_kind": kind,
                        **measure_probe(pred, frame["v_label"].to_numpy(), frame["s_label"].to_numpy(), pos),
                        "confidence": conf,
                    }
                )
        print(f"  seed {seed} fold {fold}: {len(discs)} arah x {len(cfg['models'])} model selesai", flush=True)

    probe = pd.DataFrame(rows)
    if probe[nilai].isna().any().any():
        print("ada NaN pada ukuran probe")
        return 2
    if not np.allclose(probe["agreement_visual"] + probe["agreement_sensor"], 1.0):
        print("agreement visual + sensor tidak berjumlah 1")
        return 2

    summary = summarize(probe, nilai)
    out_dir = ROOT / cfg["paths"]["results"]
    out_dir.mkdir(parents=True, exist_ok=True)
    probe.to_csv(out_dir / "probe.csv", index=False)
    summary.to_csv(out_dir / "probe_summary.csv", index=False)

    print(f"\nprobe.csv: {len(probe)} baris | total pasangan uji diskordan: {int(probe['n_pairs'].sum())}")
    print("nol pasangan latih terpakai (diperiksa per fold); akurasi tidak dihitung karena tiada ground truth")
    for direction in summary["direction"].unique():
        print(f"\n=== arah {direction} (mean±sd antar {len(seeds)} seed) ===")
        s = summary[summary["direction"] == direction]
        for _, r in s.iterrows():
            print(
                f"  {r.model:14s} ikut visual {r.agreement_visual_mean:.4f} ± {r.agreement_visual_std:.4f} "
                f"| ikut sensor {r.agreement_sensor_mean:.4f} ± {r.agreement_sensor_std:.4f} "
                f"| prediksi busuk {r.share_predicted_busuk_mean:.4f} | confidence {r.confidence_mean:.4f}"
            )
    print(f"\nhasil tertulis di {out_dir.relative_to(ROOT)}: probe.csv, probe_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())