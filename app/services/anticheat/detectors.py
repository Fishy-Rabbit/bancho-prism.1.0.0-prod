"""Behavioural anticheat detectors over extracted replay features.

Each detector is a **pure** function ``(ReplayFeatures, DetectorConfig) ->
DetectionSignal``: same features in, same signal out, no I/O, no globals. They
read the movement statistics produced by ``features.py`` and return a graded
signal; they never decide punishment. Aggregation into a per-score
``DetectionReport`` is likewise pure. The service layer (Track 2.5) consumes the
report to *flag for staff review* -- the standing policy is **flag, never
auto-ban**, until precision is measured against a server-collected baseline.

Why the thresholds here are provisional
---------------------------------------
The default cutoffs in ``DetectorConfig`` are seeded from the *shape and scale*
of a tiny external reference (three verified vanilla players): human jump-segment
aim has arc dispersion ~31 deg, real overshoot (p90 ~2.7 field units), and a
velocity coefficient of variation ~0.38. Aim assistance shows up as movement
that is *too clean* -- it snaps to the note with none of the human correction
tail. But three players is not a calibration set, so these numbers are only a
starting point. They live in a config object precisely so they can be swapped
for values re-derived by running *this* extractor over the server's own verified
replays, without touching detector logic. Until that happens every signal is
advisory and ``confidence`` is a *ranking* aid for the review queue, not a
calibrated probability of cheating.

The detectors, and what each is meant to catch:

- **B1 hold-duration** -- auto-tap / replay injection collapses the spread of
  key-hold widths (to a near-constant width, or to instantaneous clicks) where a
  human's holds vary.
- **B3 aim-controller fit** -- an aim assist produces jump geometry that is too
  clean: negligible overshoot, low arc dispersion, near-constant velocity.
- **B4 robotic timing** -- pixel-identical tap intervals (spectral comb) and
  mechanically straight / constant-velocity cursor runs indicate a timing
  assist or interpolated/injected cursor path.
- **C1 timing/aim independence** -- a human's aim precision and timing precision
  track one skill level together; a *partial* cheat decouples them (machine-clean
  aim with human timing, or robotic timing with human aim). This is the signal
  the single-channel detectors under-weight on their own.
- **D1 timewarp** -- a timewarp/injection tell in the frame cadence itself: a
  frametime distribution that is too regular for real hardware, or (once
  normalised to real time by the replay's clock rate) too fast to be a genuine
  sampling interval. Ported from circleguard's frametime check.
- **D2 aim-correction snaps** -- circleguard's ``Snap`` check: hard single-frame
  cursor reversals across a real distance (extracted in ``features.py``), flagged
  when their density over the cursor stream is implausibly high for human aim.

None of these read the beatmap: they are self-contained over the replay, so they
degrade gracefully on any map and can run before map metadata is resolved. D1
optionally takes the replay's mod clock rate to normalise frametimes (mods are
score metadata, not beatmap geometry); absent it, it assumes nomod and still
runs on its scale-invariant regularity signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum

from app.services.anticheat.features import Distribution
from app.services.anticheat.features import ReplayFeatures
from app.services.anticheat.features import max_robotic_run_taps


class Severity(StrEnum):
    """How strongly a signal argues for review.

    Ordered; use :func:`_severity_rank` to compare. ``NONE`` is a detector that
    ran and found nothing -- it is still reported, so the shadow-mode period can
    measure false-positive rates on clean players, not just hits.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_SEVERITY_ORDER = (Severity.NONE, Severity.LOW, Severity.MEDIUM, Severity.HIGH)


def _severity_rank(severity: Severity) -> int:
    return _SEVERITY_ORDER.index(severity)


def max_severity(severities: tuple[Severity, ...]) -> Severity:
    """The strongest severity in the set; ``NONE`` for an empty set."""
    return max(severities, key=_severity_rank, default=Severity.NONE)


@dataclass(frozen=True, slots=True)
class DetectionSignal:
    """One detector's graded finding on one replay.

    ``flagged`` is a convenience for "severity >= MEDIUM"; ``confidence`` is a
    0..1 ranking aid, not a calibrated probability (see module docstring).
    ``evidence`` carries the raw numbers behind the call so a reviewer -- or a
    later re-derivation of thresholds -- can see exactly what tripped it.
    """

    code: str
    title: str
    severity: Severity
    confidence: float
    flagged: bool
    detail: str
    evidence: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DetectionReport:
    """All detector signals for one replay, plus a rolled-up verdict.

    ``flagged`` is true when any signal reached MEDIUM. It never means "ban":
    it means "surface to a human". ``triggered`` is the flagged subset, ordered
    strongest-first, so the review queue can sort by it.
    """

    signals: tuple[DetectionSignal, ...]
    flagged: bool
    severity: Severity

    @property
    def triggered(self) -> tuple[DetectionSignal, ...]:
        return tuple(
            sorted(
                (s for s in self.signals if s.flagged),
                key=lambda s: (_severity_rank(s.severity), s.confidence),
                reverse=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Provisional detector thresholds.

    Seeded from the external reference's shape/scale (see module docstring);
    meant to be replaced wholesale with values re-derived from the server's own
    verified-replay baseline. Every field is a boundary between "consistent with
    a human" and "too clean / too mechanical to be one".
    """

    # --- B1 hold-duration ---
    # need this many completed holds before the spread means anything.
    hold_min_samples: int = 60
    # human hold widths spread out; a coefficient of variation at or below this
    # is a near-constant machine width (auto-tap).
    hold_cov_floor: float = 0.08
    # holds whose 90th percentile is at/under this many ms are essentially
    # instantaneous injected clicks rather than held presses.
    hold_instant_ms: float = 16.0

    # --- B3 aim-controller fit (jump segment) ---
    # jump legs needed before "too clean" is statistically meaningful.
    aim_min_jump_legs: int = 150
    # human jump overshoot p90 ~2.7 field units; assisted aim ~0. At/under this
    # the reversal-correction tail humans always have is missing.
    aim_overshoot_p90_floor: float = 0.6
    # human jump arc dispersion ~31 deg; assisted aim clusters far tighter.
    aim_arc_std_floor: float = 12.0
    # human jump velocity CoV ~0.38; near-constant speed is interpolation.
    aim_vel_cov_floor: float = 0.15
    # how many of the three "too clean" sub-conditions must hold to flag B3.
    aim_conditions_for_flag: int = 2

    # --- B4 robotic timing ---
    # a run of this many pixel-identical tap intervals is not human motor noise.
    robotic_run_taps_floor: int = 12
    # or this many separate robotic runs across the replay.
    robotic_run_count_floor: int = 3
    # corroborating mechanical cursor runs (straight / constant-velocity).
    mechanical_run_count_floor: int = 6

    # --- C1 timing/aim independence ---
    # flag only when aim- and timing-"machineness" diverge by at least this...
    independence_divergence_floor: float = 0.5
    # ...and the machine-clean side is at least this clean (so two mediocre
    # humans whose two scores happen to differ are not flagged).
    independence_machine_floor: float = 0.6

    # --- D1 timewarp (frametime cadence) ---
    # need this many frames before the frametime distribution means anything.
    timewarp_min_frames: int = 500
    # real hardware frame cadence always jitters (VSync + OS scheduling), so a
    # coefficient of variation at/under this is an unnaturally even cadence --
    # interpolated/injected frames, or a fixed-step timewarp. std/mean is
    # scale-invariant, so this needs no clock-rate normalisation and holds
    # unchanged across rate mods.
    timewarp_cv_floor: float = 0.08
    # a median frametime (ms), normalised to real time by dividing by the
    # replay's clock rate, at/under this is implausibly fast sampling for
    # sustained play. The division is circleguard's cv (clock-converted) vs ucv
    # distinction: a DT (1.5x) replay's frametimes sit on a sped clock, and
    # judging them raw would false-positive every DT play. Provisional -- the
    # ~16.7ms (60 fps) human cluster is the reference; re-derive per-mod against
    # the server's own baseline.
    timewarp_median_floor_ms: float = 11.0

    # --- D2 aim-correction snaps ---
    # need this many cursor frames before a snap density means anything.
    snap_min_frames: int = 500
    # circleguard-style aim-correction snaps per 1000 cursor frames. A human
    # produces the odd snap (a flick overshoot yanked back), but a sustained rate
    # of hard single-frame reversals onto notes is an aim-correction assist.
    # std/mean-free -- a straight rate over the frame gate.
    snap_rate_floor: float = 3.0
    # ...paired with an absolute count so a handful of snaps in a just-long-enough
    # replay can't read as a high rate; both must clear before D2 flags.
    snap_count_floor: int = 20

    # --- D3 replay similarity (steal) ---
    # mean euclidean cursor distance (osu!px) between two time-aligned replays
    # at/under which the paths are too identical to be independent play -- a
    # copied/stolen replay. circleguard's steal_max_sim (~18px) is the reference;
    # provisional, re-derived from the server's own near-miss pairs.
    steal_max_sim: float = 18.0
    # a wider band that is worth a non-flagging note (unusually close, but two
    # good players on an aim-heavy map can legitimately trace similar lines).
    steal_near_sim: float = 30.0
    # need at least this many time-aligned sample points before a mean distance
    # is meaningful; a brief overlap can be coincidentally close.
    steal_min_overlap_points: int = 500


DEFAULT_CONFIG = DetectorConfig()


# --- shared scoring helpers -------------------------------------------------


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _coefficient_of_variation(dist: Distribution) -> float | None:
    """std/mean for a distribution, or ``None`` when mean is non-positive.

    A non-positive mean means there is nothing to normalise against (e.g. every
    hold was zero-width); callers treat that as its own signal rather than a CoV.
    """
    if dist.mean <= 0.0:
        return None
    return dist.std / dist.mean


def _aim_machineness(feats: ReplayFeatures, cfg: DetectorConfig) -> float | None:
    """0..1 how machine-clean the jump-segment aim is; ``None`` if too few legs.

    Averages three sub-scores, each 1.0 when the corresponding human tail is
    fully absent and 0.0 when it is at/above the human floor: missing overshoot,
    collapsed arc dispersion, near-constant velocity. Shared by B3 (which flags
    on it directly) and C1 (which compares it against timing).
    """
    jump = feats.jump
    if jump.n < cfg.aim_min_jump_legs:
        return None

    overshoot_clean = _clamp01(
        (cfg.aim_overshoot_p90_floor - jump.overshoot.p90) / cfg.aim_overshoot_p90_floor,
    )
    arc_clean = _clamp01(
        (cfg.aim_arc_std_floor - jump.arc.std) / cfg.aim_arc_std_floor,
    )
    vel_cov = _coefficient_of_variation(jump.vel)
    if vel_cov is None:
        vel_clean = 1.0  # zero mean velocity over a jump segment is itself odd.
    else:
        vel_clean = _clamp01((cfg.aim_vel_cov_floor - vel_cov) / cfg.aim_vel_cov_floor)

    return (overshoot_clean + arc_clean + vel_clean) / 3.0


def _timing_machineness(feats: ReplayFeatures, cfg: DetectorConfig) -> float | None:
    """0..1 how machine-clean the tap *timing* is; ``None`` if too few taps.

    Combines the longest robotic (pixel-identical-interval) tap run as a
    fraction of all taps with the collapse of hold-width spread. Shared by C1.
    """
    if feats.tap_count < 2:
        return None

    robotic_fraction = _clamp01(
        max_robotic_run_taps(feats) / feats.tap_count,
    )

    hold = feats.hold
    if hold.n < cfg.hold_min_samples:
        hold_machineness = 0.0  # not enough holds to judge; lean human.
    elif hold.mean <= 0.0:
        hold_machineness = 1.0  # every press released the same frame -> injected.
    else:
        cov = hold.std / hold.mean
        hold_machineness = _clamp01((cfg.hold_cov_floor - cov) / cfg.hold_cov_floor)

    return max(robotic_fraction, hold_machineness)


# --- B1: hold-duration ------------------------------------------------------


def detect_hold_uniformity(
    feats: ReplayFeatures,
    cfg: DetectorConfig = DEFAULT_CONFIG,
) -> DetectionSignal:
    """B1 -- flag auto-tap / injected clicks by collapsed hold-width spread."""
    hold = feats.hold
    evidence: dict[str, float] = {
        "hold_n": float(hold.n),
        "hold_mean_ms": hold.mean,
        "hold_std_ms": hold.std,
        "hold_p90_ms": hold.p90,
    }

    if hold.n < cfg.hold_min_samples:
        return DetectionSignal(
            code="B1_HOLD_DURATION",
            title="Hold-duration uniformity",
            severity=Severity.NONE,
            confidence=0.0,
            flagged=False,
            detail=f"only {hold.n} holds; need {cfg.hold_min_samples} to judge spread",
            evidence=evidence,
        )

    # every press released on its own frame -> instantaneous injected clicks.
    if hold.mean <= 0.0:
        return DetectionSignal(
            code="B1_HOLD_DURATION",
            title="Hold-duration uniformity",
            severity=Severity.HIGH,
            confidence=0.95,
            flagged=True,
            detail="every key press has zero hold time -- inputs are injected, not held",
            evidence=evidence,
        )

    cov = hold.std / hold.mean
    evidence["hold_cov"] = cov
    instant = hold.p90 <= cfg.hold_instant_ms
    uniform = cov <= cfg.hold_cov_floor

    if uniform and instant:
        return DetectionSignal(
            code="B1_HOLD_DURATION",
            title="Hold-duration uniformity",
            severity=Severity.HIGH,
            confidence=0.9,
            flagged=True,
            detail=(
                f"holds are both near-constant (cov {cov:.3f}) and near-instant "
                f"(p90 {hold.p90:.1f}ms) -- consistent with an auto-clicker"
            ),
            evidence=evidence,
        )
    if uniform:
        return DetectionSignal(
            code="B1_HOLD_DURATION",
            title="Hold-duration uniformity",
            severity=Severity.MEDIUM,
            confidence=0.7,
            flagged=True,
            detail=f"hold widths barely vary (cov {cov:.3f} <= {cfg.hold_cov_floor})",
            evidence=evidence,
        )
    if instant:
        return DetectionSignal(
            code="B1_HOLD_DURATION",
            title="Hold-duration uniformity",
            severity=Severity.LOW,
            confidence=0.4,
            flagged=False,
            detail=f"holds are very short (p90 {hold.p90:.1f}ms) but do vary",
            evidence=evidence,
        )

    return DetectionSignal(
        code="B1_HOLD_DURATION",
        title="Hold-duration uniformity",
        severity=Severity.NONE,
        confidence=0.0,
        flagged=False,
        detail=f"hold widths vary humanly (cov {cov:.3f})",
        evidence=evidence,
    )


# --- B3: aim-controller fit --------------------------------------------------


def detect_aim_controller(
    feats: ReplayFeatures,
    cfg: DetectorConfig = DEFAULT_CONFIG,
) -> DetectionSignal:
    """B3 -- flag aim assistance by jump geometry that is too clean."""
    jump = feats.jump
    evidence: dict[str, float] = {
        "jump_legs": float(jump.n),
        "jump_overshoot_p90": jump.overshoot.p90,
        "jump_arc_std": jump.arc.std,
        "jump_vel_mean": jump.vel.mean,
        "jump_vel_std": jump.vel.std,
    }

    if jump.n < cfg.aim_min_jump_legs:
        return DetectionSignal(
            code="B3_AIM_CONTROLLER",
            title="Aim-controller cleanliness",
            severity=Severity.NONE,
            confidence=0.0,
            flagged=False,
            detail=f"only {jump.n} jump legs; need {cfg.aim_min_jump_legs}",
            evidence=evidence,
        )

    reasons: list[str] = []
    if jump.overshoot.p90 <= cfg.aim_overshoot_p90_floor:
        reasons.append(f"no overshoot tail (p90 {jump.overshoot.p90:.2f})")
    if jump.arc.std <= cfg.aim_arc_std_floor:
        reasons.append(f"low arc dispersion (std {jump.arc.std:.1f} deg)")
    vel_cov = _coefficient_of_variation(jump.vel)
    if vel_cov is not None:
        evidence["jump_vel_cov"] = vel_cov
        if vel_cov <= cfg.aim_vel_cov_floor:
            reasons.append(f"near-constant velocity (cov {vel_cov:.3f})")

    met = len(reasons)
    machineness = _aim_machineness(feats, cfg) or 0.0
    evidence["aim_machineness"] = machineness

    if met >= cfg.aim_conditions_for_flag:
        severity = Severity.HIGH if met >= 3 else Severity.MEDIUM
        return DetectionSignal(
            code="B3_AIM_CONTROLLER",
            title="Aim-controller cleanliness",
            severity=severity,
            confidence=_clamp01(0.5 + 0.2 * met),
            flagged=True,
            detail="jump aim is too clean for a human: " + "; ".join(reasons),
            evidence=evidence,
        )
    if met == 1:
        return DetectionSignal(
            code="B3_AIM_CONTROLLER",
            title="Aim-controller cleanliness",
            severity=Severity.LOW,
            confidence=0.35,
            flagged=False,
            detail="one clean-aim indicator: " + reasons[0],
            evidence=evidence,
        )
    return DetectionSignal(
        code="B3_AIM_CONTROLLER",
        title="Aim-controller cleanliness",
        severity=Severity.NONE,
        confidence=0.0,
        flagged=False,
        detail="jump aim carries a human correction tail",
        evidence=evidence,
    )


# --- B4: robotic timing / mechanical runs -----------------------------------


def detect_robotic_timing(
    feats: ReplayFeatures,
    cfg: DetectorConfig = DEFAULT_CONFIG,
) -> DetectionSignal:
    """B4 -- flag pixel-identical tap intervals and mechanical cursor runs."""
    longest = max_robotic_run_taps(feats)
    run_count = len(feats.robotic_tap_runs)
    mechanical = len(feats.straight_runs) + len(feats.constant_velocity_runs)
    evidence: dict[str, float] = {
        "max_robotic_run_taps": float(longest),
        "robotic_run_count": float(run_count),
        "straight_run_count": float(len(feats.straight_runs)),
        "constant_velocity_run_count": float(len(feats.constant_velocity_runs)),
    }

    strong_comb = (
        longest >= cfg.robotic_run_taps_floor
        or run_count >= cfg.robotic_run_count_floor
    )
    mechanical_cursor = mechanical >= cfg.mechanical_run_count_floor

    if strong_comb and mechanical_cursor:
        return DetectionSignal(
            code="B4_ROBOTIC_TIMING",
            title="Robotic timing / mechanical motion",
            severity=Severity.HIGH,
            confidence=0.85,
            flagged=True,
            detail=(
                f"pixel-identical tap comb (longest run {longest} taps) alongside "
                f"{mechanical} mechanical cursor runs"
            ),
            evidence=evidence,
        )
    if strong_comb:
        return DetectionSignal(
            code="B4_ROBOTIC_TIMING",
            title="Robotic timing / mechanical motion",
            severity=Severity.MEDIUM,
            confidence=0.7,
            flagged=True,
            detail=(
                f"pixel-identical tap interval sustained {longest} taps over "
                f"{run_count} run(s) -- not human motor noise"
            ),
            evidence=evidence,
        )
    if mechanical_cursor:
        return DetectionSignal(
            code="B4_ROBOTIC_TIMING",
            title="Robotic timing / mechanical motion",
            severity=Severity.LOW,
            confidence=0.4,
            flagged=False,
            detail=f"{mechanical} straight/constant-velocity cursor runs",
            evidence=evidence,
        )
    return DetectionSignal(
        code="B4_ROBOTIC_TIMING",
        title="Robotic timing / mechanical motion",
        severity=Severity.NONE,
        confidence=0.0,
        flagged=False,
        detail="tap intervals and cursor motion carry human variation",
        evidence=evidence,
    )


# --- C1: timing/aim independence --------------------------------------------


def detect_timing_aim_independence(
    feats: ReplayFeatures,
    cfg: DetectorConfig = DEFAULT_CONFIG,
) -> DetectionSignal:
    """C1 -- flag a machine-clean channel paired with a human one.

    A human's aim precision and timing precision reflect one skill level, so
    they move together. A partial cheat -- aim assist with untouched tapping, or
    a timing assist with untouched aim -- drives the two apart. This compares the
    aim- and timing-"machineness" scores and flags a large divergence where the
    clean side is genuinely clean, which is exactly the case B3/B1/B4 each
    under-weight when looking at only their own channel.
    """
    aim = _aim_machineness(feats, cfg)
    timing = _timing_machineness(feats, cfg)
    evidence: dict[str, float] = {}
    if aim is not None:
        evidence["aim_machineness"] = aim
    if timing is not None:
        evidence["timing_machineness"] = timing

    if aim is None or timing is None:
        return DetectionSignal(
            code="C1_TIMING_AIM_INDEPENDENCE",
            title="Timing/aim independence",
            severity=Severity.NONE,
            confidence=0.0,
            flagged=False,
            detail="not enough jump legs and/or taps to compare the two channels",
            evidence=evidence,
        )

    divergence = abs(aim - timing)
    cleanest = max(aim, timing)
    evidence["divergence"] = divergence

    if (
        divergence >= cfg.independence_divergence_floor
        and cleanest >= cfg.independence_machine_floor
    ):
        channel = "aim" if aim > timing else "timing"
        other = "timing" if aim > timing else "aim"
        return DetectionSignal(
            code="C1_TIMING_AIM_INDEPENDENCE",
            title="Timing/aim independence",
            severity=Severity.MEDIUM,
            confidence=_clamp01(divergence),
            flagged=True,
            detail=(
                f"{channel} is machine-clean ({cleanest:.2f}) while {other} looks "
                f"human -- the two are decoupled (divergence {divergence:.2f})"
            ),
            evidence=evidence,
        )
    return DetectionSignal(
        code="C1_TIMING_AIM_INDEPENDENCE",
        title="Timing/aim independence",
        severity=Severity.NONE,
        confidence=0.0,
        flagged=False,
        detail=f"aim and timing are consistent (divergence {divergence:.2f})",
        evidence=evidence,
    )


# --- D1: timewarp / frametime cadence ---------------------------------------


def detect_timewarp(
    feats: ReplayFeatures,
    cfg: DetectorConfig = DEFAULT_CONFIG,
    *,
    clock_rate: float = 1.0,
) -> DetectionSignal:
    """D1 -- flag a timewarped / injected frame cadence from the frametimes.

    Ported from circleguard's frametime check. Two independent tells over the
    inter-frame-interval distribution (``feats.frametime``):

    - **too regular** -- real hardware frame cadence always jitters, so a
      coefficient of variation at/under ``timewarp_cv_floor`` is an unnaturally
      even cadence (interpolated/injected frames, or a fixed-step timewarp).
      ``std/mean`` is scale-invariant, so this holds regardless of rate mods.
    - **too fast** -- a median frametime, *normalised to real time by the clock
      rate*, at/under ``timewarp_median_floor_ms`` is implausibly fast sampling
      for sustained play. Dividing by ``clock_rate`` is the cv/ucv conversion
      circleguard exposes: a DT (1.5x) replay's recorded frametimes are on a
      sped clock, and judging them raw would false-positive every DT play.

    ``clock_rate`` defaults to 1.0 (nomod). The analysis worker analyses the
    bare stored frame stream, which carries no mods, so it runs at 1.0 -- correct
    for the nomod majority and never harsher for rate mods; the staff analysis
    endpoint, which has the score row, passes the true rate for a mod-accurate
    read. The full frametime histogram a reviewer eyeballs is served separately
    (``series.py``); this detector keys on the robust scalar summary.

    Pure and advisory: like every detector here it only ever flags for review.
    """
    ft = feats.frametime
    real_median = ft.p50 / clock_rate if clock_rate > 0.0 else ft.p50
    evidence: dict[str, float] = {
        "frametime_n": float(ft.n),
        "frametime_median_ms": ft.p50,
        "frametime_real_median_ms": real_median,
        "clock_rate": clock_rate,
    }

    if ft.n < cfg.timewarp_min_frames:
        return DetectionSignal(
            code="D1_TIMEWARP",
            title="Timewarp / frametime cadence",
            severity=Severity.NONE,
            confidence=0.0,
            flagged=False,
            detail=f"only {ft.n} frames; need {cfg.timewarp_min_frames} to judge cadence",
            evidence=evidence,
        )

    cv = _coefficient_of_variation(ft)
    if cv is not None:
        evidence["frametime_cv"] = cv
    cov_str = f"{cv:.3f}" if cv is not None else "n/a"

    too_regular = cv is not None and cv <= cfg.timewarp_cv_floor
    too_fast = 0.0 < real_median <= cfg.timewarp_median_floor_ms

    reasons: list[str] = []
    if too_fast:
        reasons.append(
            f"real-time median {real_median:.1f}ms <= {cfg.timewarp_median_floor_ms:.0f}ms",
        )
    if too_regular:
        reasons.append(f"near-constant cadence (cov {cov_str})")

    if too_fast and too_regular:
        return DetectionSignal(
            code="D1_TIMEWARP",
            title="Timewarp / frametime cadence",
            severity=Severity.HIGH,
            confidence=0.85,
            flagged=True,
            detail="frame cadence is both too fast and too regular for hardware: "
            + "; ".join(reasons),
            evidence=evidence,
        )
    if too_fast or too_regular:
        return DetectionSignal(
            code="D1_TIMEWARP",
            title="Timewarp / frametime cadence",
            severity=Severity.MEDIUM,
            confidence=0.6,
            flagged=True,
            detail=reasons[0],
            evidence=evidence,
        )

    # merely on the fast / regular side: worth noting for a reviewer, not a flag.
    near_fast = 0.0 < real_median <= cfg.timewarp_median_floor_ms * 1.25
    near_regular = cv is not None and cv <= cfg.timewarp_cv_floor * 1.5
    if near_fast or near_regular:
        return DetectionSignal(
            code="D1_TIMEWARP",
            title="Timewarp / frametime cadence",
            severity=Severity.LOW,
            confidence=0.3,
            flagged=False,
            detail=f"frame cadence is on the fast/regular side (median {real_median:.1f}ms, cov {cov_str})",
            evidence=evidence,
        )

    return DetectionSignal(
        code="D1_TIMEWARP",
        title="Timewarp / frametime cadence",
        severity=Severity.NONE,
        confidence=0.0,
        flagged=False,
        detail=f"frame cadence looks human (median {real_median:.1f}ms, cov {cov_str})",
        evidence=evidence,
    )


# --- D2: aim-correction snaps -----------------------------------------------


def detect_aim_snaps(
    feats: ReplayFeatures,
    cfg: DetectorConfig = DEFAULT_CONFIG,
) -> DetectionSignal:
    """D2 -- flag aim-correction snaps: hard cursor reversals onto the note.

    Ported from circleguard's ``Snap`` / aim-correction check. ``features.py``
    extracts every snap (a cursor sample where the path darts out and back --
    a sharp A-B-C angle across a real distance); this detector judges their
    *density* against the cursor frame count. A human leaves the occasional snap
    (a flick overshoot yanked back), but a sustained rate of hard single-frame
    reversals is an aim assist snapping onto each note. Both an absolute count
    and a per-frame rate must clear their floors, so a brief burst on a short
    replay doesn't read as a high rate. Beatmap-free -- pure cursor geometry.

    Pure and advisory: it only ever flags for review.
    """
    snap_count = len(feats.snaps)
    frames = feats.frame_count
    rate = (snap_count / frames * 1000.0) if frames > 0 else 0.0
    evidence: dict[str, float] = {
        "snap_count": float(snap_count),
        "frame_count": float(frames),
        "snaps_per_1k_frames": rate,
    }

    if frames < cfg.snap_min_frames:
        return DetectionSignal(
            code="D2_AIM_SNAPS",
            title="Aim-correction snaps",
            severity=Severity.NONE,
            confidence=0.0,
            flagged=False,
            detail=f"only {frames} frames; need {cfg.snap_min_frames} to judge snap density",
            evidence=evidence,
        )

    enough = snap_count >= cfg.snap_count_floor
    high_rate = rate >= cfg.snap_rate_floor
    very_dense = rate >= cfg.snap_rate_floor * 2.0

    if enough and high_rate and very_dense:
        return DetectionSignal(
            code="D2_AIM_SNAPS",
            title="Aim-correction snaps",
            severity=Severity.HIGH,
            confidence=0.85,
            flagged=True,
            detail=(
                f"{snap_count} aim-correction snaps at {rate:.1f}/1k frames -- a dense "
                f"pattern of hard reversals onto notes"
            ),
            evidence=evidence,
        )
    if enough and high_rate:
        return DetectionSignal(
            code="D2_AIM_SNAPS",
            title="Aim-correction snaps",
            severity=Severity.MEDIUM,
            confidence=0.6,
            flagged=True,
            detail=(
                f"{snap_count} aim-correction snaps at {rate:.1f}/1k frames "
                f"(>= {cfg.snap_rate_floor}) -- more reversals than human aim carries"
            ),
            evidence=evidence,
        )
    if enough or high_rate:
        return DetectionSignal(
            code="D2_AIM_SNAPS",
            title="Aim-correction snaps",
            severity=Severity.LOW,
            confidence=0.3,
            flagged=False,
            detail=(
                f"{snap_count} snaps at {rate:.1f}/1k frames -- on the high side but "
                f"below the flag line"
            ),
            evidence=evidence,
        )
    return DetectionSignal(
        code="D2_AIM_SNAPS",
        title="Aim-correction snaps",
        severity=Severity.NONE,
        confidence=0.0,
        flagged=False,
        detail=f"{snap_count} snaps at {rate:.1f}/1k frames -- within human range",
        evidence=evidence,
    )


# --- aggregation ------------------------------------------------------------


# the uniform ``(feats, cfg)`` detectors. D1 timewarp is run separately because
# it additionally takes the replay's clock rate (see ``run_detectors``).
_MOVEMENT_DETECTORS = (
    detect_hold_uniformity,
    detect_aim_controller,
    detect_robotic_timing,
    detect_timing_aim_independence,
    detect_aim_snaps,
)


def run_detectors(
    feats: ReplayFeatures,
    cfg: DetectorConfig = DEFAULT_CONFIG,
    *,
    clock_rate: float = 1.0,
) -> DetectionReport:
    """Run every detector and roll the signals into one report.

    ``clock_rate`` is the replay's mod speed multiplier (1.0 nomod, 1.5 DT,
    0.75 HT), forwarded to D1 timewarp so its frametime read is normalised to
    real time; every other detector is scale-invariant and ignores it. It
    defaults to 1.0, which is correct for nomod and conservative otherwise, so
    the worker (which has no mods on the bare frame stream) can call this as-is.

    Pure. The report is advisory input to the staff review queue, never an
    auto-ban: ``flagged`` means "a human should look", set when any detector
    reached MEDIUM.
    """
    signals = tuple(
        [detector(feats, cfg) for detector in _MOVEMENT_DETECTORS]
        + [detect_timewarp(feats, cfg, clock_rate=clock_rate)],
    )
    severity = max_severity(tuple(s.severity for s in signals))
    flagged = _severity_rank(severity) >= _severity_rank(Severity.MEDIUM)
    return DetectionReport(signals=signals, flagged=flagged, severity=severity)
