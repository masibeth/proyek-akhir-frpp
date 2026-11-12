"""Protokol evaluasi anti-bocor: split -> pairing dalam fold -> scaler latih -> model latih.

Satu-satunya jalur yang boleh melatih model. Unimodal memakai baris berpasangan
yang sama persis dengan fused supaya perbandingannya berpasangan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.load_data import read_config

MODELS = {
    "random_forest": RandomForestClassifier,
    "svm_rbf": SVC,
    "knn": KNeighborsClassifier,
    "mlp": MLPClassifier,
}


class RobustZClip(BaseEstimator, TransformerMixin):
    """Clip outlier pada robust z (median/MAD); statistik fit hanya dari data latih."""

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.center_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.center_), axis=0)
        self.scale_ = np.where(mad > 0, mad, 1.0)
        return self

    def transform(self, X):
        z = (np.asarray(X, dtype=float) - self.center_) / self.scale_
        return np.clip(z, -self.threshold, self.threshold) * self.scale_ + self.center_


def visual_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("f"))


def feature_matrix(df: pd.DataFrame, kind: str, cfg: dict) -> np.ndarray:
    v = visual_columns(df)
    s = cfg["sensor"]["channels"]
    cols = {"visual": v, "sensor": s, "fused": v + s}[kind]
    return df[cols].to_numpy(dtype=float)


def _splitter(cfg: dict, groups):
    n = cfg["protocol"]["n_folds"]
    if groups is not None:
        return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=cfg["seed"])
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=cfg["seed"])


def fold_splits(cfg: dict, visual: pd.DataFrame, sensor: pd.DataFrame) -> list[tuple]:
    """Split tiap modalitas terpisah; group visual dipakai bila tersedia."""
    v_groups = visual["group_id"] if "group_id" in visual.columns else None
    if v_groups is not None and v_groups.isna().any():
        v_groups = None
    v = list(_splitter(cfg, v_groups).split(visual, visual["label"], v_groups))
    s = list(_splitter(cfg, sensor["group_id"]).split(sensor, sensor["label"], sensor["group_id"]))
    if len(v) != len(s):
        raise ValueError("jumlah fold visual dan sensor berbeda")
    return [(a, b, c, d) for (a, b), (c, d) in zip(v, s)]


def _pair_within(v_labels: np.ndarray, s_labels: np.ndarray, rng) -> list[tuple]:
    pairs = []
    for cls in sorted(set(v_labels) & set(s_labels)):
        iv = np.flatnonzero(v_labels == cls)
        is_ = np.flatnonzero(s_labels == cls)
        n = min(len(iv), len(is_))
        rng.shuffle(iv)
        rng.shuffle(is_)
        pairs.extend(zip(iv[:n], is_[:n]))
    return pairs


def _join(visual, sensor, v_pos, s_pos, pairs, cfg) -> pd.DataFrame:
    vi = np.array([v_pos[a] for a, _ in pairs], dtype=int)
    si = np.array([s_pos[b] for _, b in pairs], dtype=int)
    v = visual.iloc[vi].reset_index(drop=True)
    s = sensor.iloc[si].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "visual_id": v["visual_id"].to_numpy(),
            "sensor_id": s["sensor_id"].to_numpy(),
            "label": v["label"].to_numpy(),
            "v_label": v["label"].to_numpy(),
            "s_label": s["label"].to_numpy(),
            "v_group": v["group_id"].to_numpy() if "group_id" in v.columns else np.nan,
            "s_group": s["group_id"].to_numpy(),
        }
    )
    for c in visual_columns(visual):
        out[c] = v[c].to_numpy()
    for c in cfg["sensor"]["channels"]:
        out[c] = s[c].to_numpy()
    return out


def iter_folds(cfg: dict, visual: pd.DataFrame, sensor: pd.DataFrame):
    """Yield (seed_pairing, fold, train_berpasangan, test_berpasangan)."""
    splits = fold_splits(cfg, visual, sensor)
    for seed in cfg["protocol"]["pairing_seeds"]:
        for fold, (v_tr, v_te, s_tr, s_te) in enumerate(splits, start=1):
            rng = np.random.default_rng(seed * 1000 + fold)
            tr = _join(visual, sensor, v_tr, s_tr,
                       _pair_within(visual["label"].to_numpy()[v_tr],
                                    sensor["label"].to_numpy()[s_tr], rng), cfg)
            te = _join(visual, sensor, v_te, s_te,
                       _pair_within(visual["label"].to_numpy()[v_te],
                                    sensor["label"].to_numpy()[s_te], rng), cfg)
            yield seed, fold, tr, te


def build_model(cfg: dict, name: str):
    params = dict(cfg["models"][name])
    if params.get("random_state") == "global":
        params["random_state"] = cfg["seed"]
    return MODELS[name](**params)


def make_pipeline(cfg: dict, model_name: str, scaling: str | None = None) -> Pipeline:
    scaling = scaling or cfg["fusion"]["post_concat_scaling"]
    steps = [("clip", RobustZClip(cfg["sensor"]["outlier"]["threshold"]))]
    if scaling == "standard":
        steps.append(("scale", StandardScaler()))
    steps.append(("model", build_model(cfg, model_name)))
    return Pipeline(steps)