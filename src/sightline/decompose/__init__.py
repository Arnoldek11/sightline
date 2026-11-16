"""Vision Loss / Decision Loss (PLAN 0 and 4.5).

Per decision, per Monte Carlo draw d of what the carrier saw (seen set S_d):
    best_all    = max_o V(o)
    best_seen_d = max_{o in S_d ∪ {chosen}} V(o)     (the chosen option was, by definition,
                                                       perceived: he played it; D-010)
    Vision Loss_d   = best_all - best_seen_d   >= 0
    Decision Loss_d = best_seen_d - V(chosen)  >= 0
    Total Loss      = best_all - V(chosen)     = VL_d + DL_d  (exact for every draw)
Reported losses are means over draws.

Null model (winner's curse control): in each draw, replace S_d by a random set of the same
size (chosen still included). Excess Vision Loss = VL - VL_null. If carriers' attention
tracks value, VL < VL_null; if seen sets were random, VL ~ VL_null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_losses(values: np.ndarray, seen: np.ndarray, chosen: int,
                 rng: np.random.Generator | None = None) -> dict:
    """values (M,), seen (M, D) bool, chosen index -> loss dict (means over draws)."""
    values = np.asarray(values, float)
    seen = np.asarray(seen, bool).copy()
    seen[chosen] = True
    M, D = seen.shape
    best_all = values.max()
    v_chosen = values[chosen]
    masked = np.where(seen, values[:, None], -np.inf)
    best_seen = masked.max(axis=0)  # (D,)
    vl = best_all - best_seen
    dl = best_seen - v_chosen
    out = {"v_best_all": best_all, "v_chosen": v_chosen, "v_best_seen": best_seen.mean(),
           "total_loss": best_all - v_chosen, "vision_loss": vl.mean(),
           "decision_loss": dl.mean(), "p_best_seen": float(seen[values.argmax()].mean()),
           "n_seen": float(seen.sum(0).mean()), "n_options": M}
    if rng is not None:
        k = seen.sum(axis=0)  # per draw, includes chosen
        others = np.delete(np.arange(M), chosen)
        null_best = np.empty(D)
        for d in range(D):
            pick = rng.choice(others, size=max(int(k[d]) - 1, 0), replace=False)
            null_best[d] = max(v_chosen, values[pick].max() if len(pick) else -np.inf)
        out["vision_loss_null"] = float((best_all - null_best).mean())
    return out


def shrink(means: pd.Series, ses: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Empirical-Bayes normal-normal shrinkage of player means towards their group mean.

    tau^2 (between-player variance within a group) by method of moments, floored at 0.
    Returns posterior mean, posterior sd, and the shrinkage weight.
    """
    df = pd.DataFrame({"m": means, "se": ses, "g": groups})
    out = []
    for _, sub in df.groupby("g"):
        mu = np.average(sub.m, weights=1 / sub.se.clip(lower=1e-9) ** 2)
        tau2 = max(float(np.var(sub.m, ddof=1) - np.mean(sub.se ** 2)), 0.0) if len(sub) > 1 \
            else 0.0
        w = tau2 / (tau2 + sub.se ** 2) if tau2 > 0 else pd.Series(0.0, index=sub.index)
        post = w * sub.m + (1 - w) * mu
        sd = np.sqrt(w * sub.se ** 2) if tau2 > 0 else pd.Series(0.0, index=sub.index)
        out.append(pd.DataFrame({"post_mean": post, "post_sd": sd, "weight": w,
                                 "group_mean": mu}))
    return pd.concat(out).reindex(df.index)
