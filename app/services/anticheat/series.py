"""Graph-ready analysis series over a parsed replay.

The detectors in ``detectors.py`` reduce a replay to scalar signals for the
review queue. This module does the complementary thing: it turns the *same*
parsed ``.osr`` (plus its extracted ``ReplayFeatures``) into JSON-able arrays a
staff "Analyze" panel can plot and the replay renderer (Phase 6) can overlay --
the cursor path, per-step velocity, the frametime series + histogram behind D1,
the holdtime histogram behind B1, and the D2 aim-correction snap markers.

Everything here is **pure** (same replay in, same arrays out) and **computed on
demand, never stored**: the flag ``evidence`` column stays scalar-only, and a
staff-gated endpoint recomputes these arrays from the retained ``.osr`` when a
reviewer opens a score. Re-analysis is idempotent, so recompute is always safe
and the tables stay lean.

To keep the graphed numbers identical to the ones the detectors judged, the
per-step math is reused from ``features.py`` (``_absolute_samples`` /
``_step_velocities`` / ``_frametimes`` / ``_hold_durations``) rather than
re-derived here -- a second implementation would be free to drift from the
extractor's, and then a graph could disagree with the flag it is supposed to
explain. Series are emitted columnar (parallel ``t`` + value arrays) because
that is both compact over JSON and what typed-array plotters and PIXI want.

Hit-error / rolling-UR / judgement series are deliberately absent: they need
beatmap geometry (Phase 4) and land with the relax/UR detector in Phase 5.
"""

from __future__ import annotations

import math
from typing import Any

from app.adapters.osr_replay import Replay
from app.services.anticheat.features import ReplayFeatures
from app.services.anticheat.features import _absolute_samples
from app.services.anticheat.features import _frametimes
from app.services.anticheat.features import _hold_durations
from app.services.anticheat.features import _step_velocities

# default histogram bin widths (ms). Frametimes cluster tightly around the
# engine's ~16.67 ms step, so a 1 ms bin resolves the timewarp tells D1 keys on;
# holdtimes spread over tens of ms, so a coarser bin keeps the shape legible.
_FRAMETIME_BIN_MS = 1.0
_HOLDTIME_BIN_MS = 4.0


def cursor_path(replay: Replay) -> dict[str, Any]:
    """The full cursor stream as columnar ``t`` (ms) / ``x`` / ``y`` / ``keys``.

    Every frame is included -- lead-in and duplicate-timestamp samples too -- so
    the renderer draws the exact path the client sent. ``keys`` is the raw
    ``ReplayKeys`` bitmask per frame as an int.
    """
    t_ms: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    keys: list[int] = []
    t = 0
    for frame in replay.frames:
        t += frame.time_delta
        t_ms.append(t)
        xs.append(frame.x)
        ys.append(frame.y)
        keys.append(int(frame.keys))
    return {"t": t_ms, "x": xs, "y": ys, "keys": keys}


def velocity_series(replay: Replay) -> dict[str, Any]:
    """Per-step cursor speed (osu!px/ms), stamped at each step's end time.

    ``v[k]`` is the speed over the step ending at ``t[k]``; there is one point
    per inter-frame step, so the arrays are one shorter than the frame count.
    """
    samples = _absolute_samples(replay.frames)
    velocities = _step_velocities(samples)
    t_ms = [samples[i][0] for i in range(1, len(samples))]
    return {"t": t_ms, "v": velocities}


def frametime_series(replay: Replay) -> dict[str, Any]:
    """Positive inter-frame intervals (ms), stamped at the frame they end on.

    Matches the extractor's ``_frametimes`` filter (non-positive deltas dropped),
    so this is the raw sequence behind the D1 timewarp read. Divide ``dt`` by the
    replay's clock rate for a real-time view (the bundle carries the rate).
    """
    t_ms: list[int] = []
    dts: list[float] = []
    t = 0
    for frame in replay.frames:
        t += frame.time_delta
        if frame.time_delta > 0:
            t_ms.append(t)
            dts.append(float(frame.time_delta))
    return {"t": t_ms, "dt": dts}


def _histogram(values: list[float], *, bin_width: float) -> dict[str, Any]:
    """Bin ``values`` into fixed-width buckets aligned to a ``bin_width`` grid.

    Returns ``{bin_width, bin_min, counts}`` where bucket ``i`` covers
    ``[bin_min + i*bin_width, bin_min + (i+1)*bin_width)`` and ``sum(counts) ==
    len(values)``. ``bin_min`` is floored onto the grid so bin edges are stable
    across replays. Empty input yields empty ``counts``.
    """
    if bin_width <= 0.0:
        raise ValueError("bin_width must be positive")
    if not values:
        return {"bin_width": bin_width, "bin_min": 0.0, "counts": []}

    lo = min(values)
    hi = max(values)
    bin_min = math.floor(lo / bin_width) * bin_width
    n_bins = int(math.floor((hi - bin_min) / bin_width)) + 1
    counts = [0] * n_bins
    for value in values:
        idx = int(math.floor((value - bin_min) / bin_width))
        idx = 0 if idx < 0 else n_bins - 1 if idx >= n_bins else idx
        counts[idx] += 1
    return {"bin_width": bin_width, "bin_min": bin_min, "counts": counts}


def frametime_histogram(
    replay: Replay,
    *,
    bin_width_ms: float = _FRAMETIME_BIN_MS,
) -> dict[str, Any]:
    """Histogram of positive frametimes -- the shape a reviewer reads for D1."""
    return _histogram(_frametimes(replay.frames), bin_width=bin_width_ms)


def holdtime_histogram(
    replay: Replay,
    *,
    bin_width_ms: float = _HOLDTIME_BIN_MS,
) -> dict[str, Any]:
    """Histogram of completed key-hold durations -- the shape behind B1."""
    return _histogram(_hold_durations(replay.frames), bin_width=bin_width_ms)


def snap_markers(feats: ReplayFeatures) -> list[dict[str, Any]]:
    """The D2 aim-correction snaps as playback markers, in time order.

    Read straight off the extracted features (no recompute): each marker is the
    snap's absolute time, the A-B-C angle, and the dart distance, so the panel
    can drop a pin on the cursor path exactly where D2 counted a reversal.
    """
    return [
        {"t_ms": snap.t_ms, "angle_deg": snap.angle_deg, "distance": snap.distance}
        for snap in feats.snaps
    ]


def build_analysis_series(
    replay: Replay,
    feats: ReplayFeatures,
    *,
    clock_rate: float = 1.0,
) -> dict[str, Any]:
    """Bundle every series a staff Analyze view needs for one replay.

    ``clock_rate`` (the mod speed multiplier: 1.0 nomod, 1.5 DT, 0.75 HT) is
    carried through so the frontend can render the frametime graphs in real time;
    the series themselves stay in raw replay time. Pure and recomputed on demand.
    """
    return {
        "frame_count": len(replay.frames),
        "duration_ms": feats.duration_ms,
        "clock_rate": clock_rate,
        "cursor": cursor_path(replay),
        "velocity": velocity_series(replay),
        "frametime": {
            "series": frametime_series(replay),
            "histogram": frametime_histogram(replay),
        },
        "holdtime": {
            "histogram": holdtime_histogram(replay),
        },
        "snaps": snap_markers(feats),
    }
