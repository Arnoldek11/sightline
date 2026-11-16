"""Stream SkillCorner body pose (25 fps, 29 joints) from the Hugging Face zip.

The full JSON is ~3.3 GB per match, so records are streamed line by line from inside the zip
and only the joints needed for orientation are kept. Output: one row per (pose_frame,
player) with joints, columns `<joint>_x`, `<joint>_y`, `<joint>_err` (p90 error radius, cm).
z is dropped: it is centroid-relative, not pitch-referenced.
"""

from __future__ import annotations

import gzip
import os
import zipfile
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

from sightline.paths import data_dir

POSE_FPS = 25
KEY_JOINTS = ("nose", "lEar", "rEar", "lEye", "rEye", "neck", "lShoulder", "rShoulder",
              "midHip", "lHip", "rHip")


def pose_zip_path(match_id: int) -> Path:
    return data_dir() / "pose" / f"{match_id}.jsonl.zip"


def iter_records(path: Path) -> Iterator[dict]:
    """Yield pose frame records from a `.jsonl.zip`, `.jsonl.gz` or `.jsonl` file."""
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".jsonl"))
            with zf.open(name) as fh:
                for line in fh:
                    yield orjson.loads(line)
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as fh:
            for line in fh:
                yield orjson.loads(line)


def records_to_table(records: Iterable[dict], joints: tuple[str, ...] = KEY_JOINTS,
                     frame_filter: set[int] | None = None) -> pd.DataFrame:
    """Flatten pose records into one row per player-frame that has joints."""
    cols: dict[str, list] = {"pose_frame": [], "period": [], "player_id": []}
    for j in joints:
        for s in ("x", "y", "err"):
            cols[f"{j}_{s}"] = []
    nan = float("nan")
    for r in records:
        f = r["frame"]
        if frame_filter is not None and f not in frame_filter:
            continue
        for p in r["player_data"]:
            jd = p.get("joints")
            if not jd:
                continue
            cols["pose_frame"].append(f)
            cols["period"].append(r["period"])
            cols["player_id"].append(p["player_id"])
            for j in joints:
                v = jd.get(j)
                if v is None:
                    cols[f"{j}_x"].append(nan)
                    cols[f"{j}_y"].append(nan)
                    cols[f"{j}_err"].append(nan)
                else:
                    cols[f"{j}_x"].append(v["xyz"][0])
                    cols[f"{j}_y"].append(v["xyz"][1])
                    cols[f"{j}_err"].append(v["p90_mae_cm"])
    df = pd.DataFrame(cols)
    for c in df.columns:
        if c.endswith(("_x", "_y", "_err")):
            df[c] = df[c].astype(np.float32)
    return df


def _iter_line_batches(path: Path, batch: int) -> Iterator[list[bytes]]:
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".jsonl"))
        with zf.open(name) as fh:
            buf: list[bytes] = []
            for line in fh:
                buf.append(line)
                if len(buf) == batch:
                    yield buf
                    buf = []
            if buf:
                yield buf


def _parse_batch(lines: list[bytes]) -> pd.DataFrame:
    return records_to_table(orjson.loads(x) for x in lines)


def load_pose(match_id: int, path: Path | None = None, workers: int | None = None,
              batch: int = 1000) -> pd.DataFrame:
    """Whole-match pose table. JSON parsing is CPU-bound, so batches go to a process pool."""
    path = Path(path or pose_zip_path(match_id))
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    if workers <= 1 or path.suffix != ".zip":
        return records_to_table(iter_records(path))
    parts: list[pd.DataFrame] = []
    pending: deque = deque()
    with ProcessPoolExecutor(workers) as ex:  # bounded queue: never holds the whole file
        for lines in _iter_line_batches(path, batch):
            pending.append(ex.submit(_parse_batch, lines))
            if len(pending) >= 2 * workers:
                parts.append(pending.popleft().result())
        parts.extend(f.result() for f in pending)
    return pd.concat(parts, ignore_index=True)
