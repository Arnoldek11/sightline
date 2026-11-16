"""Phase 5 (quick pass): Vision Loss / Decision Loss on the two pose games.

Decision set (D-001, D-009): pass decisions with an intended receiver, options = passes +
carries, window pose coverage >= 50%. Writes <data>/processed/<id>/losses.parquet and
reports/decomposition_quick.json. Usage: uv run python scripts/decompose_pose_games.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sightline.decompose import shrink, split_losses
from sightline.io.tracking import load_meta
from sightline.paths import reports_dir
from sightline.pipeline import POSE_GAMES, load_config, load_processed, processed_dir
from sightline.vision import VisionConfig, build_geometry, seen_draws, with_static_options

GRID = {f"c{c}_m{m}": (c, m) for c in (45, 60, 90) for m in (1.0, 1.5, 2.5)}


def run_game(mid: int, cfg: VisionConfig, rng: np.random.Generator) -> pd.DataFrame:
    dec = load_processed(mid, "decisions")
    dec = dec[(dec.end_type == "pass") & dec.target_id.notna()]
    opts = load_processed(mid, "options")
    opts = opts[opts.option_type != "shot"]
    players = load_processed(mid, "players")
    geoms = build_geometry(dec, players, load_processed(mid, "pose"), cfg)
    meta = load_meta(mid)
    pos_group = meta.players.set_index("player_id").position_group
    d_by = dec.set_index("event_id")
    o_by = {e: g for e, g in opts.groupby("event_id")}
    rows = []
    for g in geoms:
        o = o_by.get(g.event_id)
        if o is None:
            continue
        d = d_by.loc[g.event_id]
        a = int(d.attack_dir)
        carries = o[o.option_type == "carry"]
        g2 = with_static_options(g, carries[["x_att", "y_att"]].to_numpy() * a,
                                 list(carries.option_id))
        ids = [str(int(i)) if not isinstance(i, str) else i for i in g2.option_ids]
        val = o.set_index("option_id").value.reindex(ids).to_numpy()
        keep = ~np.isnan(val)
        chosen_id = str(int(d.target_id))
        if chosen_id not in ids or not keep[ids.index(chosen_id)]:
            continue
        seen, cov = seen_draws(g2, cfg.cone_deg, cfg.memory_s, cfg.max_move_m)
        if cov < cfg.min_coverage:
            continue
        vals, seen = val[keep], seen[keep]
        kept_ids = [i for i, k in zip(ids, keep, strict=True) if k]
        c = kept_ids.index(chosen_id)
        r = split_losses(vals, seen, c, rng=rng)
        r |= {"match_id": mid, "event_id": g.event_id, "player_id": int(d.player_id),
              "position_group": pos_group.get(int(d.player_id)), "coverage": cov,
              "counter": bool(d.counter), "frame_end": int(d.frame_end),
              "carrier_x_att": float(o.x_att.iloc[0] * 0 + np.nan),
              "best_type": "pass" if not kept_ids[int(np.argmax(vals))].startswith("carry")
              else "carry"}
        for name, (cone, mem) in GRID.items():
            s, _ = seen_draws(g2, cone, mem, cfg.max_move_m)
            rr = split_losses(vals, s[keep], c)
            r[f"vl_{name}"], r[f"dl_{name}"] = rr["vision_loss"], rr["decision_loss"]
        s, _ = seen_draws(g2, cfg.cone_deg, cfg.memory_s, cfg.max_move_m, placebo=True)
        r["vl_placebo"] = split_losses(vals, s[keep], c)["vision_loss"]
        rows.append(r)
    out = pd.DataFrame(rows)
    # carrier location in the attacking frame at release
    pl = players.set_index(["frame", "player_id"])[["x"]]
    x = pl.reindex(pd.MultiIndex.from_arrays([out.frame_end, out.player_id])).x.to_numpy()
    out["carrier_x_att"] = x * d_by.attack_dir.reindex(out.event_id).to_numpy()
    out.to_parquet(processed_dir(mid) / "losses.parquet")
    return out


def summary(df: pd.DataFrame) -> dict:
    tl, vl, dl = df.total_loss.mean(), df.vision_loss.mean(), df.decision_loss.mean()
    return {"n": len(df), "total_loss": round(tl, 5), "vision_loss": round(vl, 5),
            "decision_loss": round(dl, 5), "vision_share": round(vl / tl, 3) if tl else None,
            "vision_loss_null": round(df.vision_loss_null.mean(), 5),
            "vision_loss_placebo": round(df.vl_placebo.mean(), 5),
            "p_best_seen": round(df.p_best_seen.mean(), 3),
            "share_chosen_best": round(float((df.total_loss <= 1e-12).mean()), 3)}


def main() -> None:
    cfg_d = load_config("default")
    cfg = VisionConfig.from_dict(cfg_d)
    rng = np.random.default_rng(cfg.seed)
    df = pd.concat([run_game(m, cfg, rng) for m in POSE_GAMES], ignore_index=True)
    rep: dict = {"overall": summary(df)}
    rep["subsets"] = {
        "attacking_half": summary(df[df.carrier_x_att > 0]),
        "own_half": summary(df[df.carrier_x_att <= 0]),
        "final_third": summary(df[df.carrier_x_att > 52.5 / 3]),
        "counter": summary(df[df.counter]),
        "best_is_pass": summary(df[df.best_type == "pass"]),
        "best_is_carry": summary(df[df.best_type == "carry"]),
    }
    rep["sensitivity_vision_share"] = {
        k: round(df[f"vl_{k}"].mean() / df.total_loss.mean(), 3) for k in GRID}

    # player profiles
    min_n = cfg_d["profiles"]["min_decisions"]
    g = df.groupby("player_id")
    pl = pd.DataFrame({"n": g.size(), "vl": g.vision_loss.mean(), "dl": g.decision_loss.mean(),
                       "vl_se": g.vision_loss.std() / np.sqrt(g.size()),
                       "dl_se": g.decision_loss.std() / np.sqrt(g.size()),
                       "pos": g.position_group.first()})
    pl = pl[pl.n >= min_n].copy()
    for k in ("vl", "dl"):
        s = shrink(pl[k], pl[f"{k}_se"], pl.pos)
        pl[f"{k}_eb"], pl[f"{k}_eb_sd"], pl[f"{k}_w"] = s.post_mean, s.post_sd, s.weight
    rep["players"] = {"n_players": len(pl), "min_decisions": min_n,
                      "mean_shrink_weight_vl": round(float(pl.vl_w.mean()), 3),
                      "mean_shrink_weight_dl": round(float(pl.dl_w.mean()), 3),
                      "spearman_vl_dl": round(float(pl[["vl", "dl"]].corr("spearman")
                                                    .iloc[0, 1]), 3)}
    # split-half stability: alternate decisions per player
    df["half"] = df.groupby("player_id").cumcount() % 2
    sh = df.groupby(["player_id", "half"])[["vision_loss", "decision_loss"]].mean().unstack()
    sh = sh[df.groupby("player_id").size().reindex(sh.index) >= 2 * min_n]
    rep["players"]["split_half"] = {
        "n_players": len(sh),
        "vl_spearman": round(float(sh.vision_loss.corr(method="spearman").iloc[0, 1]), 3),
        "dl_spearman": round(float(sh.decision_loss.corr(method="spearman").iloc[0, 1]), 3)}
    # construct validity: scanning before receiving vs Vision Loss, within position
    sc = pd.concat([load_processed(m, "scans").assign(match_id=m) for m in POSE_GAMES])
    j = df.merge(sc[sc.pose_share >= 0.5][["match_id", "event_id", "n_scans"]],
                 on=["match_id", "event_id"])
    for c in ("n_scans", "vision_loss"):
        j[c + "_dm"] = j[c] - j.groupby("position_group")[c].transform("mean")
    rep["construct_scanning"] = {
        "n": len(j), "spearman_within_position": round(float(
            j[["n_scans_dm", "vision_loss_dm"]].corr("spearman").iloc[0, 1]), 3)}
    pl.to_csv(reports_dir() / "players_quick.csv")
    (reports_dir() / "decomposition_quick.json").write_text(json.dumps(rep, indent=2,
                                                                        default=float))
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
