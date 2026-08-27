"""D3 replay-similarity (replay steal) -- circleguard's ``Steal`` check.

Unlike the detectors in ``detectors.py`` this is **pairwise**: it compares *two*
replays, so it is not a ``(features, config) -> signal`` member of
``run_detectors``. The shape is instead:

    result = replay_similarity(replay_a, replay_b)      # pure geometry
    signal = detect_replay_similarity(result, cfg)      # pure flagging

``replay_similarity`` reduces two cursor streams to one number -- the mean
euclidean distance (osu!px) between the two paths once they are put on a common
time base -- and ``detect_replay_similarity`` grades it. Low distance means the
two replays trace nearly the same line, which independent human play does not
produce: one replay was copied from the other. This mirrors circlecore's
``Comparer``/``Steal`` (mean cursor distance; ``steal_max_sim`` ~18px).

**Time alignment.** The two frame streams are delta-timed and sampled at
different, uneven cadences, so a frame-index comparison would be meaningless. We
resample both onto one fixed-step grid across the window where the plays
*overlap* in time, linearly interpolating each cursor between its bracketing
frames. The grid is shared, so the metric is symmetric in A and B.

**Scope / limitations (documented, deferred).** This compares raw playfield
coordinates. It catches the common case -- an exact or near-exact copy of a
replay under the same mods -- but does not yet normalise for HR's y-flip or
search for a best time-shift alignment (circleguard does both). Mods are score
metadata the worker's bare frame stream lacks; mod-aware normalisation lands with
the score-row-backed service wiring (the analysis endpoint has the mods) and
Phase 3. Until then a HR-vs-nomod pair reads as dissimilar, which is
conservative -- it can miss a cross-mod steal, never manufacture one.

Pure and advisory: like every detector, D3 only ever emits a ``DetectionSignal``
into the review queue. A human actions it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.adapters.osr_replay import Replay
from app.services.anticheat.detectors import DEFAULT_CONFIG
from app.services.anticheat.detectors import DetectionSignal
from app.services.anticheat.detectors import DetectorConfig
from app.services.anticheat.detectors import Severity
from app.services.anticheat.features import _absolute_samples

# resample cadence (ms) for the shared comparison grid. ~60 Hz: fine enough that
# two genuinely identical paths stay near-zero distance and two different plays
# stay far apart, coarse enough that a full-map overlap is a few thousand points.
# The exact value barely moves the metric (a copy is ~0 at any resolution); it
# only sets how much wall-clock overlap ``steal_min_overlap_points`` demands.
_GRID_STEP_MS = 16.0


@dataclass(frozen=True, slots=True)
class SimilarityResult:
    """The outcome of comparing two replays' cursor paths.

    ``n`` is the number of time-aligned points compared; ``mean_distance`` /
    ``median_distance`` are the euclidean cursor distances (osu!px) over those
    points; ``overlap_ms`` is the length of the shared time window. When the two
    replays do not overlap in time (or either is too short to interpolate), ``n``
    is 0 and the distances are ``inf`` -- "not comparable", never a false match.
    """

    n: int
    mean_distance: float
    median_distance: float
    overlap_ms: int


_NO_OVERLAP = SimilarityResult(n=0, mean_distance=math.inf, median_distance=math.inf, overlap_ms=0)


def _resample(samples: list[tuple[int, float, float]], grid: list[float]) -> list[tuple[float, float]]:
    """Linear-interpolate ``(x, y)`` at each ``grid`` time along ``samples``.

    Both ``samples`` (by construction) and ``grid`` (built ascending) are sorted
    by time, and every grid time lies inside ``[samples[0].t, samples[-1].t]``
    (the caller clamps the grid to the overlap window), so one forward pointer
    walks both in a single linear pass. A zero-width bracket (duplicate
    timestamps, e.g. a paused frame) takes the earlier sample's position.
    """
    out: list[tuple[float, float]] = []
    j = 0
    n = len(samples)
    for t in grid:
        while j + 1 < n and samples[j + 1][0] < t:
            j += 1
        t0, x0, y0 = samples[j]
        t1, x1, y1 = samples[j + 1] if j + 1 < n else samples[j]
        span = t1 - t0
        if span <= 0:
            out.append((x0, y0))
        else:
            frac = (t - t0) / span
            out.append((x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)))
    return out


def _median(values: list[float]) -> float:
    """Median of a non-empty list (mean of the two middles when even)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def replay_similarity(
    replay_a: Replay,
    replay_b: Replay,
    *,
    step_ms: float = _GRID_STEP_MS,
) -> SimilarityResult:
    """Mean/median euclidean cursor distance between two replays over their overlap.

    Resamples both cursor paths onto one ``step_ms`` grid across the time window
    where they overlap, then measures the per-point distance. Lower means more
    identical: an exact copy sits near 0, two independent plays sit far above the
    ``steal_max_sim`` line. Returns a not-comparable result (``n == 0``,
    distances ``inf``) when the replays are too short or never overlap in time.

    Pure: same two replays in, same result out.
    """
    if step_ms <= 0.0:
        raise ValueError("step_ms must be positive")

    sa = _absolute_samples(replay_a.frames)
    sb = _absolute_samples(replay_b.frames)
    if len(sa) < 2 or len(sb) < 2:
        return _NO_OVERLAP

    # rewind frames (a negative delta) can make cumulative time briefly
    # non-monotonic; sort so the interpolation walk sees ascending time. Real
    # replays almost never do this, so on the common path this is a no-op.
    sa.sort(key=lambda s: s[0])
    sb.sort(key=lambda s: s[0])

    lo = max(sa[0][0], sb[0][0])
    hi = min(sa[-1][0], sb[-1][0])
    if hi <= lo:
        return _NO_OVERLAP

    n_steps = int(math.floor((hi - lo) / step_ms))
    grid = [lo + k * step_ms for k in range(n_steps + 1)]

    path_a = _resample(sa, grid)
    path_b = _resample(sb, grid)
    distances = [
        math.hypot(ax - bx, ay - by)
        for (ax, ay), (bx, by) in zip(path_a, path_b, strict=True)
    ]

    return SimilarityResult(
        n=len(distances),
        mean_distance=math.fsum(distances) / len(distances),
        median_distance=_median(distances),
        overlap_ms=hi - lo,
    )


def detect_replay_similarity(
    result: SimilarityResult,
    cfg: DetectorConfig = DEFAULT_CONFIG,
    *,
    other_score_id: int = 0,
) -> DetectionSignal:
    """D3 -- grade one replay-similarity result as a steal signal.

    Judges the mean cursor distance from :func:`replay_similarity` against
    ``steal_max_sim``: at/under half of it the paths are near-identical (a
    frame-for-frame copy), at/under it they are too close for independent play,
    and out to ``steal_near_sim`` they are merely worth a reviewer's note. A short
    overlap (fewer than ``steal_min_overlap_points`` aligned points) can be
    coincidentally close, so it never flags. ``other_score_id`` labels which
    replay this was compared against so a reviewer can pull it up.

    Pure and advisory: it only ever flags for review.
    """
    evidence: dict[str, float] = {
        "overlap_points": float(result.n),
        "overlap_ms": float(result.overlap_ms),
        "other_score_id": float(other_score_id),
    }

    if result.n < cfg.steal_min_overlap_points:
        return DetectionSignal(
            code="D3_REPLAY_SIMILARITY",
            title="Replay similarity (steal)",
            severity=Severity.NONE,
            confidence=0.0,
            flagged=False,
            detail=(
                f"only {result.n} time-aligned points; need "
                f"{cfg.steal_min_overlap_points} to judge similarity"
            ),
            evidence=evidence,
        )

    mean = result.mean_distance
    evidence["mean_distance"] = mean
    evidence["median_distance"] = result.median_distance

    if mean <= cfg.steal_max_sim * 0.5:
        return DetectionSignal(
            code="D3_REPLAY_SIMILARITY",
            title="Replay similarity (steal)",
            severity=Severity.HIGH,
            confidence=0.9,
            flagged=True,
            detail=(
                f"cursor paths are near-identical (mean {mean:.1f}px vs score "
                f"{other_score_id}) -- a copied replay, not independent play"
            ),
            evidence=evidence,
        )
    if mean <= cfg.steal_max_sim:
        return DetectionSignal(
            code="D3_REPLAY_SIMILARITY",
            title="Replay similarity (steal)",
            severity=Severity.MEDIUM,
            confidence=0.7,
            flagged=True,
            detail=(
                f"cursor paths are too close for independent play (mean {mean:.1f}px "
                f"<= {cfg.steal_max_sim:.0f} vs score {other_score_id})"
            ),
            evidence=evidence,
        )
    if mean <= cfg.steal_near_sim:
        return DetectionSignal(
            code="D3_REPLAY_SIMILARITY",
            title="Replay similarity (steal)",
            severity=Severity.LOW,
            confidence=0.3,
            flagged=False,
            detail=(
                f"cursor paths are unusually close (mean {mean:.1f}px vs score "
                f"{other_score_id}) but within what two good plays can share"
            ),
            evidence=evidence,
        )
    return DetectionSignal(
        code="D3_REPLAY_SIMILARITY",
        title="Replay similarity (steal)",
        severity=Severity.NONE,
        confidence=0.0,
        flagged=False,
        detail=f"cursor paths diverge normally (mean {mean:.1f}px vs score {other_score_id})",
        evidence=evidence,
    )
