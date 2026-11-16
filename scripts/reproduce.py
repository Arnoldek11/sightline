"""End-to-end pipeline -> data/processed + reports/. Stages are added phase by phase.

    uv run python scripts/reproduce.py --quick   # one pose game (1925299)
    uv run python scripts/reproduce.py --full    # all 20 games
"""

from __future__ import annotations

import argparse
import json
import time

import orjson

from sightline.paths import data_dir, reports_dir
from sightline.pipeline import build_game

QUICK_GAME = 1925299


def all_games() -> list[int]:
    m = orjson.loads((data_dir() / "opendata" / "data" / "matches.json").read_bytes())
    return sorted(int(x["id"]) for x in m)


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="one pose game")
    g.add_argument("--full", action="store_true", help="all games")
    a = p.parse_args()
    games = [QUICK_GAME] if a.quick else all_games()
    t = time.perf_counter()
    summaries = [build_game(m) for m in games]
    out = reports_dir() / ("data_layer_quick.json" if a.quick else "data_layer_full.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"elapsed_s": round(time.perf_counter() - t, 1),
                               "games": summaries}, indent=2))
    print(f"done in {time.perf_counter() - t:.0f}s -> {out}")


if __name__ == "__main__":
    main()
