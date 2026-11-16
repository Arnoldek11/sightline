"""Phase 3: build datasets, fit cross-fitted option models, evaluate, value pose-game options.

Outputs: <data>/processed/models/*.parquet + option_models.pkl,
         <data>/processed/<pose id>/options.parquet, reports/values_validation.json
Usage:   uv run python scripts/fit_option_models.py
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sightline.counters import tag_decisions
from sightline.io.events import decision_points, load_events
from sightline.options import data as D
from sightline.options.models import (
    brier,
    carry_model,
    game_folds,
    pass_model,
    reliability,
    value_model,
    xg_model,
)
from sightline.options.values import OptionModels, option_table
from sightline.paths import data_dir, reports_dir
from sightline.pipeline import POSE_GAMES, load_config, load_processed, processed_dir


def games() -> list[int]:
    full = json.loads((reports_dir() / "data_layer_full.json").read_text())
    return [g["match_id"] for g in full["games"]]


def logit_baseline(train_cols, df, y, folds):
    """Cross-fitted logistic on a few columns (baseline)."""
    out = np.full(len(df), np.nan)
    f = df.match_id.map(folds).to_numpy()
    for k in set(folds.values()):
        m = LogisticRegression(max_iter=1000).fit(df.loc[f != k, train_cols], df.loc[f != k, y])
        out[f == k] = m.predict_proba(df.loc[f == k, train_cols])[:, 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-s", type=float, default=None,
                    help="override value.horizon_s from configs/default.yaml")
    ap.add_argument("--tag", default="", help="suffix for outputs (sensitivity runs)")
    a = ap.parse_args()
    horizon_s = a.horizon_s or load_config("default")["value"]["horizon_s"]
    tag = a.tag
    t0 = time.perf_counter()
    gs = games()
    folds = game_folds(gs)
    mdir = data_dir() / "processed" / f"models{tag}"
    mdir.mkdir(parents=True, exist_ok=True)

    # 0. refresh decision tables (new columns) and collect SkillCorner targeted-option xpass
    sc_xpass = []
    for m in gs:
        ev = load_events(m)
        d = tag_decisions(decision_points(ev), load_processed(m, "counters"))
        d.to_parquet(processed_dir(m) / "decisions.parquet")
        po = ev[(ev.event_type == "passing_option") & (ev.targeted == True)]  # noqa: E712
        sc_xpass.append(po[["associated_player_possession_event_id", "xpass_completion"]]
                        .rename(columns={"associated_player_possession_event_id": "event_id"})
                        .assign(match_id=m))
    sc_xpass = pd.concat(sc_xpass).dropna().drop_duplicates(["match_id", "event_id"])

    # 1. shots/passes/carries datasets, game by game
    S, P, C = [], [], []
    for m in gs:
        g = D.GameData.load(m)
        S.append(D.shots(g))
        P.append(D.passes(g))
        C.append(D.carries(g))
        del g
        gc.collect()
    shots, passes, carries = (pd.concat(x, ignore_index=True) for x in (S, P, C))
    print(f"shots {len(shots)}, passes {len(passes)}, carries {len(carries)} "
          f"({time.perf_counter() - t0:.0f}s)")

    xg = xg_model().fit(shots, "goal", folds)
    shots["xg"] = xg.oof(shots)
    # 2. value states (labels use out-of-fold xG); event ids are only unique within a game
    V = []
    for m in gs:
        g = D.GameData.load(m)
        V.append(D.states(g, shots[shots.match_id == m].set_index("event_id").xg,
                          horizon=int(horizon_s * 10)))
        del g
        gc.collect()
    states = pd.concat(V, ignore_index=True)
    print(f"states {len(states):,} ({time.perf_counter() - t0:.0f}s)")

    pm = pass_model().fit(passes, "completed", folds)
    cm = carry_model().fit(carries, "kept", folds)
    vm = value_model().fit(states, "y_xg", folds)
    passes["p"] = pm.oof(passes)
    carries["p"] = cm.oof(carries)
    states["v"] = vm.oof(states)
    for name, df in (("shots", shots), ("passes", passes), ("carries", carries),
                     ("states", states)):
        df.to_parquet(mdir / f"{name}.parquet")
    models = OptionModels(xg, pm, cm, vm)
    with open(mdir / "option_models.pkl", "wb") as fh:
        pickle.dump(models, fh)

    # 3. evaluation (all out-of-fold)
    rep: dict = {"n": {"shots": len(shots), "goals": int(shots.goal.sum()),
                       "passes": len(passes), "carries": len(carries), "states": len(states)}}
    passes["p_len"] = logit_baseline(["length"], passes, "completed", folds)
    rep["pass"] = {
        "brier_model": round(brier(passes.completed, passes.p), 4),
        "brier_length_only": round(brier(passes.completed, passes.p_len), 4),
        "brier_base_rate": round(brier(passes.completed, np.full(len(passes),
                                                                 passes.completed.mean())), 4),
        "auc_model": round(roc_auc_score(passes.completed, passes.p), 3),
        "auc_length_only": round(roc_auc_score(passes.completed, passes.p_len), 3),
        "reliability": reliability(passes.completed, passes.p).to_dict("records"),
    }
    j = passes.merge(sc_xpass, on=["match_id", "event_id"])
    rep["pass"]["vs_skillcorner_xpass"] = {
        "n": len(j), "spearman": round(float(j[["p", "xpass_completion"]].corr("spearman")
                                             .iloc[0, 1]), 3),
        "auc_skillcorner": round(roc_auc_score(j.completed, j.xpass_completion), 3),
        "auc_ours_same_rows": round(roc_auc_score(j.completed, j.p), 3)}
    rep["carry"] = {
        "brier_model": round(brier(carries.kept, carries.p), 4),
        "brier_base_rate": round(brier(carries.kept, np.full(len(carries),
                                                             carries.kept.mean())), 4),
        "auc_model": round(roc_auc_score(carries.kept, carries.p), 3)}
    rep["xg"] = {
        "brier_model": round(brier(shots.goal, shots.xg), 4),
        "brier_base_rate": round(brier(shots.goal, np.full(len(shots), shots.goal.mean())), 4),
        "auc_model": round(roc_auc_score(shots.goal, shots.xg), 3),
        "sum_xg_vs_goals": [round(float(shots.xg.sum()), 1), int(shots.goal.sum())]}
    ss_res = float(((states.y_xg - states.v) ** 2).sum())
    ss_tot = float(((states.y_xg - states.y_xg.mean()) ** 2).sum())
    loc_only = states.copy()
    rep["value"] = {
        "r2_oof": round(1 - ss_res / ss_tot, 4),
        "auc_any_shot_10s": round(roc_auc_score(states.y_shot, states.v), 3),
        "mean_pred_vs_obs": [round(float(states.v.mean()), 5),
                             round(float(states.y_xg.mean()), 5)],
        "reliability": reliability(states.y_xg, states.v).to_dict("records"),
    }
    loc_only["xbin"] = pd.cut(loc_only.x, 10)
    base = loc_only.groupby("xbin", observed=True).y_xg.transform("mean")
    rep["value"]["r2_x_bins_baseline"] = round(1 - float(((states.y_xg - base) ** 2).sum())
                                               / ss_tot, 4)

    # 4. options for the pose games + SkillCorner EPV cross-check
    opt_all = []
    for m in POSE_GAMES:
        g = D.GameData.load(m)
        dec = g.decisions[g.decisions.end_type.isin(["pass", "shot"])]
        ot = option_table(g, dec, models)
        ot.to_parquet(processed_dir(m) / f"options{tag}.parquet")
        opt_all.append(ot.assign(match_id=m))
        cur = ot.drop_duplicates("event_id")[["event_id", "v_current"]].merge(
            g.decisions[["event_id", "sc_epv_start"]], on="event_id").dropna()
        rep.setdefault("value_vs_skillcorner_epv", {})[str(m)] = {
            "n": len(cur), "spearman_v_release_vs_epv_start": round(float(
                cur[["v_current", "sc_epv_start"]].corr("spearman").iloc[0, 1]), 3)}
    ot = pd.concat(opt_all, ignore_index=True)
    ot["key"] = ot.match_id.astype(str) + "_" + ot.event_id  # event ids repeat across games
    per = ot.groupby("key")
    best = per.value.max()
    chosen = ot[ot.is_chosen].set_index("key").value
    shot = ot[ot.option_type == "shot"].set_index("key")
    other = ot[ot.option_type != "shot"].groupby("key").value.max()
    rep["options_pose_games"] = {
        "decisions": int(ot.key.nunique()),
        "options_per_decision": round(len(ot) / ot.key.nunique(), 1),
        "share_with_chosen_option": round(float(per.is_chosen.any().mean()), 3),
        "shot_best_when_available": round(float((shot.value > other.reindex(shot.index))
                                                .mean()), 3),
        "shot_chosen_when_available": round(float(shot.is_chosen.mean()), 3),
        "mean_total_loss": round(float((best.reindex(chosen.index) - chosen).mean()), 5),
        "share_chosen_is_best": round(float((best.reindex(chosen.index) - chosen <= 1e-12)
                                            .mean()), 3),
        "best_option_type": ot.loc[per.value.idxmax(), "option_type"].value_counts(
            normalize=True).round(3).to_dict(),
    }
    rep["elapsed_s"] = round(time.perf_counter() - t0)
    rep["horizon_s"] = horizon_s
    (reports_dir() / f"values_validation{tag}.json").write_text(json.dumps(rep, indent=2,
                                                                      default=float))
    short = {k: ({kk: vv for kk, vv in v.items() if kk != "reliability"}
                 if isinstance(v, dict) else v) for k, v in rep.items()}
    print(json.dumps(short, indent=2, default=float))


if __name__ == "__main__":
    main()
