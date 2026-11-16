"""Option models, cross-fitted by game (PLAN 4.2).

Every model is fitted K times on game folds; a game is always scored by the model that did
not see it (`CrossFitted.predict(match_id, X)`), so values for the pose games are out-of-sample
and the evaluation numbers are honest.

* xG:      logistic(distance, angle) on shots
* P(pass): logistic on physics features (time-to-intercept margins, lane, length, ...)
* P(carry): logistic on carry physics features
* V(state): LightGBM regression of E[own xG within 10 s] on ball-location state features
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sightline.options.geometry import CARRY_FEATURES, PASS_FEATURES, SHOT_FEATURES, STATE_FEATURES


def game_folds(games: list[int], k: int = 5) -> dict[int, int]:
    """Deterministic game -> fold assignment (sorted ids, round robin)."""
    return {g: i % k for i, g in enumerate(sorted(games))}


@dataclass
class CrossFitted:
    features: list[str]
    make: Callable[[], object]
    kind: str  # "proba" or "reg"
    folds: dict[int, int] = field(default_factory=dict)
    models: dict[int, object] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, y: str, folds: dict[int, int]) -> CrossFitted:
        self.folds = folds
        fold = df.match_id.map(folds)
        for k in sorted(set(folds.values())):
            tr = df[fold != k]
            m = self.make()
            m.fit(tr[self.features], tr[y])
            self.models[k] = m
        return self

    def _pred(self, m, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        if self.kind == "proba":
            return m.predict_proba(X)[:, 1]
        return np.maximum(m.predict(X), 0.0)

    def predict(self, match_id: int, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return self._pred(self.models[self.folds[match_id]], X)

    def oof(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), np.nan)
        fold = df.match_id.map(self.folds).to_numpy()
        for k, m in self.models.items():
            sel = fold == k
            if sel.any():
                out[sel] = self._pred(m, df.loc[sel])
        return out


def _logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def _lgbm_reg():
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.9, reg_lambda=1.0, verbose=-1, random_state=0)


def xg_model() -> CrossFitted:
    return CrossFitted(SHOT_FEATURES, _logit, "proba")


def pass_model() -> CrossFitted:
    return CrossFitted(PASS_FEATURES, _logit, "proba")


def carry_model() -> CrossFitted:
    return CrossFitted(CARRY_FEATURES, _logit, "proba")


def value_model() -> CrossFitted:
    return CrossFitted(STATE_FEATURES, _lgbm_reg, "reg")


# ---- evaluation helpers ------------------------------------------------------------------


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    """Calibration table on predicted-probability deciles."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"),
                                 observed=("y", "mean")).round(4)
