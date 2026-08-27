"""Tests for D3 replay-similarity (steal) -- the pairwise cursor-path check.

Two layers are pinned separately, mirroring features/detectors elsewhere:

- **geometry** (``replay_similarity`` + its ``_resample``/``_median`` helpers):
  built on hand-made frame streams so the mean/median distance is exact -- a copy
  reads ~0, a constant offset reads the offset, disjoint windows read as
  not-comparable;
- **flagging** (``detect_replay_similarity``): fed hand-built ``SimilarityResult``
  objects so each severity band and the min-overlap gate are pinned independently
  of the geometry, and the evidence stays JSON-safe (never an ``inf`` leak).
"""

from __future__ import annotations

import json
import math

import pytest

from app.adapters.osr_replay import Replay
from app.adapters.osr_replay import ReplayFrame
from app.adapters.osr_replay import ReplayKeys
from app.services.anticheat.detectors import DEFAULT_CONFIG
from app.services.anticheat.detectors import Severity
from app.services.anticheat.similarity import SimilarityResult
from app.services.anticheat.similarity import _median
from app.services.anticheat.similarity import _resample
from app.services.anticheat.similarity import detect_replay_similarity
from app.services.anticheat.similarity import replay_similarity


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


def _drift(n: int = 600, *, dx: float = 0.0, dy: float = 0.0, step: int = 16) -> list[ReplayFrame]:
    """A smooth diagonal cursor drift of ``n`` frames, optionally offset by (dx, dy).

    ``step`` ms per frame at the default 16 gives ``n`` grid points over the run,
    so ``n >= 500`` clears the detector's overlap gate. The offset shifts the
    whole path rigidly, so a comparison against the un-offset path has a known
    constant distance ``hypot(dx, dy)`` at every aligned point.
    """
    return [_frame(step, i * 1.3 + dx, i * 0.7 + dy) for i in range(n)]


def _result(mean: float, *, n: int = 1000, median: float | None = None, overlap_ms: int = 20_000) -> SimilarityResult:
    return SimilarityResult(
        n=n,
        mean_distance=mean,
        median_distance=mean if median is None else median,
        overlap_ms=overlap_ms,
    )


# --- interpolation primitives ----------------------------------------------


def test_resample_linearly_interpolates_between_samples() -> None:
    samples = [(0, 0.0, 0.0), (10, 10.0, 20.0)]
    assert _resample(samples, [0.0, 5.0, 10.0]) == [(0.0, 0.0), (5.0, 10.0), (10.0, 20.0)]


def test_resample_duplicate_timestamp_takes_earlier_sample() -> None:
    # a paused frame (zero-width bracket) must not divide by zero.
    samples = [(0, 0.0, 0.0), (0, 9.0, 9.0), (10, 10.0, 10.0)]
    assert _resample(samples, [0.0])[0] == (0.0, 0.0)


def test_median_of_even_and_odd_lengths() -> None:
    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5


# --- similarity geometry ----------------------------------------------------


def test_identical_replays_have_zero_distance() -> None:
    replay = _replay(_drift())
    result = replay_similarity(replay, _replay(_drift()))
    assert result.n >= DEFAULT_CONFIG.steal_min_overlap_points
    assert result.mean_distance < 1e-9
    assert result.median_distance < 1e-9


def test_constant_offset_yields_that_offset_as_mean_distance() -> None:
    # same timing, path B shifted +3px in x -> every aligned point is exactly 3 apart.
    result = replay_similarity(_replay(_drift()), _replay(_drift(dx=3.0)))
    assert abs(result.mean_distance - 3.0) < 1e-9
    assert abs(result.median_distance - 3.0) < 1e-9


def test_offset_scales_the_distance() -> None:
    # a 3-4-5 offset resolves to exactly 5px everywhere.
    result = replay_similarity(_replay(_drift()), _replay(_drift(dx=3.0, dy=4.0)))
    assert abs(result.mean_distance - 5.0) < 1e-9


def test_too_short_replay_is_not_comparable() -> None:
    result = replay_similarity(_replay([_frame(16, 0.0, 0.0)]), _replay(_drift()))
    assert result.n == 0
    assert math.isinf(result.mean_distance)


def test_disjoint_time_windows_are_not_comparable() -> None:
    # replay A only starts (huge lead-in) after replay B has already ended.
    late = _replay([_frame(100_000, 0.0, 0.0)] + _drift())
    result = replay_similarity(late, _replay(_drift()))
    assert result.n == 0
    assert math.isinf(result.mean_distance)


def test_replay_similarity_rejects_nonpositive_step() -> None:
    with pytest.raises(ValueError):
        replay_similarity(_replay(_drift()), _replay(_drift()), step_ms=0.0)


# --- flagging bands ---------------------------------------------------------


def test_near_identical_mean_flags_high() -> None:
    signal = detect_replay_similarity(_result(2.0))
    assert signal.severity == Severity.HIGH
    assert signal.flagged is True


def test_close_mean_flags_medium() -> None:
    signal = detect_replay_similarity(_result(12.0))
    assert signal.severity == Severity.MEDIUM
    assert signal.flagged is True


def test_unusually_close_mean_is_low_and_unflagged() -> None:
    signal = detect_replay_similarity(_result(25.0))
    assert signal.severity == Severity.LOW
    assert signal.flagged is False


def test_normal_divergence_is_none() -> None:
    signal = detect_replay_similarity(_result(100.0))
    assert signal.severity == Severity.NONE
    assert signal.flagged is False


def test_band_boundaries_are_inclusive_on_the_stricter_side() -> None:
    cfg = DEFAULT_CONFIG
    assert detect_replay_similarity(_result(cfg.steal_max_sim * 0.5)).severity == Severity.HIGH
    assert detect_replay_similarity(_result(cfg.steal_max_sim)).severity == Severity.MEDIUM
    assert detect_replay_similarity(_result(cfg.steal_near_sim)).severity == Severity.LOW


def test_short_overlap_never_flags_even_when_identical() -> None:
    # a coincidentally-close brief overlap is not evidence of a steal.
    signal = detect_replay_similarity(_result(0.0, n=100))
    assert signal.severity == Severity.NONE
    assert signal.flagged is False
    assert "mean_distance" not in signal.evidence  # not asserted on too little data


def test_evidence_is_json_safe_even_when_not_comparable() -> None:
    # the not-comparable result carries inf distances; the signal must not leak them.
    signal = detect_replay_similarity(SimilarityResult(0, math.inf, math.inf, 0), other_score_id=42)
    assert signal.evidence["other_score_id"] == 42.0
    json.dumps(signal.evidence)  # would raise if an inf reached the evidence dict


def test_evidence_records_the_compared_score() -> None:
    signal = detect_replay_similarity(_result(2.0), other_score_id=777)
    assert signal.evidence["other_score_id"] == 777.0
    assert signal.evidence["mean_distance"] == 2.0


# --- geometry + flagging together -------------------------------------------


def test_copied_replay_end_to_end_flags_high() -> None:
    result = replay_similarity(_replay(_drift()), _replay(_drift()))
    signal = detect_replay_similarity(result, other_score_id=5)
    assert signal.severity == Severity.HIGH
    assert signal.flagged is True
    json.dumps(signal.evidence)
