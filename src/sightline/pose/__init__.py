"""Facing directions from pose joints, with Monte Carlo uncertainty.

Angles are radians in the pitch frame: 0 = +x, counter-clockwise positive, range (-pi, pi].
* Head yaw  = direction from the ears' midpoint to the nose (ground plane).
* Torso yaw = left->right shoulder axis rotated +90 deg (SkillCorner baseline convention).

Uncertainty: SkillCorner's `p90_mae_cm` is the radius containing 90% of errors in 3D. For an
isotropic 3D Gaussian that radius is 2.5003 sigma (Maxwell), so sigma per axis = p90 / 2.5003.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

P90_TO_SIGMA = 1.0 / 2.5003  # Maxwell distribution, 90% quantile
MIN_HEAD_VEC_M = 0.02
SHOULDER_WIDTH_M = (0.15, 0.60)


def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return np.pi - np.mod(np.pi - np.asarray(a, float), 2 * np.pi)


def _xy(df: pd.DataFrame, j: str) -> np.ndarray:
    return df[[f"{j}_x", f"{j}_y"]].to_numpy(float)


def _head(nose, lear, rear):
    v = nose - (lear + rear) / 2
    yaw = np.arctan2(v[..., 1], v[..., 0])
    return np.where(np.hypot(v[..., 0], v[..., 1]) >= MIN_HEAD_VEC_M, yaw, np.nan)


def _torso(lsh, rsh):
    d = rsh - lsh
    yaw = np.arctan2(d[..., 0], -d[..., 1])
    w = np.hypot(d[..., 0], d[..., 1])
    return np.where((w >= SHOULDER_WIDTH_M[0]) & (w <= SHOULDER_WIDTH_M[1]), yaw, np.nan)


HEAD_JOINTS = ("nose", "lEar", "rEar")
TORSO_JOINTS = ("lShoulder", "rShoulder")


def _quality(df: pd.DataFrame, joints, max_err_cm: float) -> np.ndarray:
    errs = df[[f"{j}_err" for j in joints]].to_numpy(float)
    return np.all(errs <= max_err_cm, axis=1)  # NaN (missing joint) -> False


def facing(df: pd.DataFrame, max_err_cm: float = 15.0) -> pd.DataFrame:
    """Point estimates `head_yaw`, `torso_yaw` (NaN where gated out)."""
    out = pd.DataFrame(index=df.index)
    h = _head(_xy(df, "nose"), _xy(df, "lEar"), _xy(df, "rEar"))
    t = _torso(_xy(df, "lShoulder"), _xy(df, "rShoulder"))
    out["head_yaw"] = np.where(_quality(df, HEAD_JOINTS, max_err_cm), h, np.nan)
    out["torso_yaw"] = np.where(_quality(df, TORSO_JOINTS, max_err_cm), t, np.nan)
    return out


def sample_yaw(df: pd.DataFrame, kind: str = "head", n_draws: int = 200,
               rng: np.random.Generator | None = None,
               max_err_cm: float = 15.0) -> np.ndarray:
    """Monte Carlo yaw draws, shape (len(df), n_draws). Joint xy ~ N(obs, sigma^2 I)."""
    rng = rng or np.random.default_rng(0)
    joints = HEAD_JOINTS if kind == "head" else TORSO_JOINTS
    pts = []
    for j in joints:
        xy = _xy(df, j)[:, None, :]
        sigma = (df[f"{j}_err"].to_numpy(float) / 100.0 * P90_TO_SIGMA)[:, None, None]
        pts.append(xy + rng.standard_normal((len(df), n_draws, 2)) * sigma)
    draws = _head(*pts) if kind == "head" else _torso(*pts)
    ok = _quality(df, joints, max_err_cm)
    draws[~ok] = np.nan
    return draws


def circ_std(draws: np.ndarray, axis: int = -1) -> np.ndarray:
    """Circular standard deviation (radians), ignoring NaNs."""
    c = np.nanmean(np.cos(draws), axis=axis)
    s = np.nanmean(np.sin(draws), axis=axis)
    r = np.clip(np.hypot(c, s), 1e-12, 1.0)
    return np.sqrt(-2 * np.log(r))
