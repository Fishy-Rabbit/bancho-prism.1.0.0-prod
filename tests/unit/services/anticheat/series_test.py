"""Tests for the graph-ready analysis-series builders.

Frames are built directly (as in ``features_test``) so each series is pinned on
a known geometry. The builders are pure and reuse the extractor's own per-step
math, so these assert exact arrays: time alignment, positive-delta filtering,
histogram binning, and that the whole bundle is JSON-serializable.
"""

from __future__ import annotations

import json

import pytest

from app.adapters.osr_replay import Replay
from app.adapters.osr_replay import ReplayFrame
from app.adapters.osr_replay import ReplayKeys
from app.services.anticheat.features import extract_features
from app.services.anticheat.series import _histogram
from app.services.anticheat.series import build_analysis_series
from app.services.anticheat.series import cursor_path
from app.services.anticheat.series import frametime_histogram
from app.services.anticheat.series import frametime_series
from app.services.anticheat.series import holdtime_histogram
from app.services.anticheat.series import snap_markers
from app.services.anticheat.series import velocity_series


def _replay(frames: list[ReplayFrame]) -> Replay:
    return Replay(
        mode=0,
        version=20200207,
        beatmap_md5="a" * 32,
        player_name="tester",
        replay_md5="b" * 32,
        n300=0,
        n100=0,
        n50=0,
        ngeki=0,
        nkatu=0,
        nmiss=0,
        score=0,
        max_combo=0,
        perfect=False,
        mods=0,
        life_bar_graph="",
        timestamp=0,
        frames=tuple(frames),
        rng_seed=None,
        online_score_id=0,
    )


def _frame(dt: int, x: float, y: float, keys: ReplayKeys = ReplayKeys.NONE) -> ReplayFrame:
    return ReplayFrame(time_delta=dt, x=x, y=y, keys=keys)


# --- cursor path ------------------------------------------------------------


def test_cursor_path_includes_every_frame_with_cumulative_time() -> None:
    frames = [
        _frame(0, 1.0, 2.0, ReplayKeys.NONE),
        _frame(10, 3.0, 4.0, ReplayKeys.KEY_1),
        _frame(16, 5.0, 6.0, ReplayKeys.KEY_1 | ReplayKeys.MOUSE_1),
    ]
    path = cursor_path(_replay(frames))
    assert path["t"] == [0, 10, 26]  # cumulative deltas
    assert path["x"] == [1.0, 3.0, 5.0]
    assert path["y"] == [2.0, 4.0, 6.0]
    assert path["keys"] == [0, int(ReplayKeys.KEY_1), int(ReplayKeys.KEY_1 | ReplayKeys.MOUSE_1)]


# --- velocity ---------------------------------------------------------------


def test_velocity_series_is_per_step_speed_stamped_at_step_end() -> None:
    # 10u over 10ms then 20u over 10ms -> speeds 1.0, 2.0 at t=10, t=20.
    frames = [_frame(0, 0.0, 0.0), _frame(10, 10.0, 0.0), _frame(10, 30.0, 0.0)]
    series = velocity_series(_replay(frames))
    assert series["t"] == [10, 20]
    assert series["v"] == [1.0, 2.0]


# --- frametime series -------------------------------------------------------


def test_frametime_series_keeps_only_positive_deltas() -> None:
    frames = [
        _frame(0, 0.0, 0.0),   # lead-in: delta 0 -> dropped
        _frame(16, 1.0, 1.0),  # t=16, dt 16
        _frame(-5, 2.0, 2.0),  # rewind: delta -5 -> dropped (t=11)
        _frame(17, 3.0, 3.0),  # t=28, dt 17
    ]
    series = frametime_series(_replay(frames))
    assert series["t"] == [16, 28]
    assert series["dt"] == [16.0, 17.0]


# --- histograms -------------------------------------------------------------


def test_histogram_bins_align_to_grid_and_count_all_values() -> None:
    values = [16.0, 16.4, 17.0, 18.9, 20.0]
    hist = _histogram(values, bin_width=1.0)
    assert hist["bin_width"] == 1.0
    assert hist["bin_min"] == 16.0
    # [16,17): 16.0,16.4 -> 2 ; [17,18): 17.0 -> 1 ; [18,19): 18.9 -> 1 ;
    # [19,20): none -> 0 ; [20,21): 20.0 -> 1
    assert hist["counts"] == [2, 1, 1, 0, 1]
    assert sum(hist["counts"]) == len(values)


def test_histogram_empty_values_is_empty_counts() -> None:
    assert _histogram([], bin_width=2.0) == {"bin_width": 2.0, "bin_min": 0.0, "counts": []}


def test_histogram_rejects_nonpositive_bin_width() -> None:
    with pytest.raises(ValueError):
        _histogram([1.0], bin_width=0.0)


def test_frametime_histogram_counts_positive_frametimes() -> None:
    frames = [_frame(0, 0.0, 0.0)] + [_frame(16, float(i), 0.0) for i in range(1, 6)]
    hist = frametime_histogram(_replay(frames), bin_width_ms=1.0)
    assert sum(hist["counts"]) == 5  # the leading 0-delta frame is dropped
    assert hist["bin_min"] == 16.0


def test_holdtime_histogram_counts_completed_holds() -> None:
    # K1 held 40ms, K2 held 20ms -> two completed holds.
    frames = [
        _frame(0, 0.0, 0.0, ReplayKeys.KEY_1),
        _frame(40, 0.0, 0.0, ReplayKeys.NONE),
        _frame(10, 1.0, 1.0, ReplayKeys.KEY_2),
        _frame(20, 1.0, 1.0, ReplayKeys.NONE),
    ]
    hist = holdtime_histogram(_replay(frames), bin_width_ms=4.0)
    assert sum(hist["counts"]) == 2


# --- snap markers -----------------------------------------------------------


def test_snap_markers_come_from_extracted_features() -> None:
    # the same hard out-and-back reversal the extractor flags as one snap.
    frames = [_frame(10, 0.0, 0.0), _frame(10, 50.0, 0.0), _frame(10, 0.0, 0.0)]
    feats = extract_features(_replay(frames))
    markers = snap_markers(feats)
    assert len(markers) == 1
    assert markers[0]["t_ms"] == 20
    assert markers[0]["distance"] == 50.0
    assert markers[0]["angle_deg"] < 1e-6


# --- bundle -----------------------------------------------------------------


def test_build_analysis_series_bundles_everything_and_is_json_safe() -> None:
    frames = [_frame(10, 0.0, 0.0), _frame(10, 50.0, 0.0), _frame(10, 0.0, 0.0)]
    replay = _replay(frames)
    feats = extract_features(replay)

    bundle = build_analysis_series(replay, feats, clock_rate=1.5)

    assert set(bundle) == {
        "frame_count",
        "duration_ms",
        "clock_rate",
        "cursor",
        "velocity",
        "frametime",
        "holdtime",
        "snaps",
    }
    assert bundle["frame_count"] == 3
    assert bundle["clock_rate"] == 1.5
    assert set(bundle["frametime"]) == {"series", "histogram"}
    assert bundle["cursor"]["t"] == [10, 20, 30]
    assert len(bundle["snaps"]) == 1
    # the whole bundle must serialize -- catches any ReplayKeys/enum leak.
    json.dumps(bundle)
