"""On-demand replay analysis for the staff review panel.

The worker (``replay_analysis.py``) analyses each score *once* and stores scalar
signals. This service answers the complementary, interactive questions a
reviewer asks while looking at one score, recomputed live from the retained
``.osr`` and never stored:

- **analysis series** -- the graph-ready bundle from ``series.py`` (cursor path,
  velocity, frametime/holdtime histograms, snap markers) for the Analyze view and
  the Phase 6 renderer;
- **similarity** -- D3 run pairwise against the other leaderboard replays on the
  same beatmap+mode, surfacing the nearest ones so a reviewer can spot a steal.

Why this is separate from the worker's ``ReplayAnalysisService``: the worker sees
only the bare stored frame stream (``parse_replay_frame_data`` forces ``mods=0``),
so it analyses in raw replay time. This service is backed by the ``scores`` row,
so it alone knows the score's real mods -- hence it, not the worker, computes the
``clock_rate`` the frametime graphs need, and it owns the cross-replay candidate
lookup D3 requires.

Impure edges are injected callables (read the ``.osr``, read the score row, list
candidates), so the whole thing is unit-testable with no filesystem and no
database, in the frozen-dataclass style the rest of the codebase uses. And like
everything in the anticheat, it is **advisory**: it computes numbers for a human
to read, it never actions a player.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any
from typing import Protocol

from app.adapters.osr_replay import Replay
from app.adapters.osr_replay import ReplayParseError
from app.adapters.osr_replay import parse_replay_frame_data
from app.constants.mods import Mods
from app.services.anticheat.detectors import DEFAULT_CONFIG
from app.services.anticheat.detectors import DetectionSignal
from app.services.anticheat.detectors import DetectorConfig
from app.services.anticheat.features import extract_features
from app.services.anticheat.series import build_analysis_series
from app.services.anticheat.similarity import detect_replay_similarity
from app.services.anticheat.similarity import replay_similarity

# DT and NC are the same 1.5x speed-up; the flag differs only in pitch.
_SPEED_UP_MODS = Mods.DOUBLETIME | Mods.NIGHTCORE


def _clock_rate(mods: int) -> float:
    """The mod speed multiplier: 1.5 under DT/NC, 0.75 under HT, else 1.0.

    The worker analyses the bare stored frame stream (mods live on the score
    row, not in the ``.osr``), so only this score-row-backed service can supply
    the real rate to the frametime graphs. DT+HT is rejected as an invalid combo
    at submission, so the two branches are mutually exclusive in practice; the
    speed-up is checked first regardless.
    """
    if mods & _SPEED_UP_MODS:
        return 1.5
    if mods & Mods.HALFTIME:
        return 0.75
    return 1.0


class ScoreRow(Protocol):
    """The score-row fields this service reads (structural; ``Score`` satisfies it)."""

    map_md5: str
    mode: int
    mods: int
    userid: int


class ScoreReader(Protocol):
    """Read one score row by id, or ``None`` if it is gone."""

    def __call__(self, score_id: int, /) -> Awaitable[ScoreRow | None]: ...


class CandidateRow(Protocol):
    """A comparison candidate (structural; ``ReplayCandidate`` satisfies it)."""

    score_id: int
    user_id: int
    mods: int


class CandidateFinder(Protocol):
    """List same-map/mode scores to compare one replay against."""

    def __call__(
        self,
        *,
        map_md5: str,
        mode: int,
        exclude_score_id: int,
        exclude_user_id: int | None,
        limit: int,
    ) -> Awaitable[list[CandidateRow]]: ...


class ReplayBytesReader(Protocol):
    """Read a stored replay's raw frame bytes, or ``None`` if absent."""

    def __call__(self, score_id: int, /) -> Awaitable[bytes | None]: ...


class ReplayUnavailableError(Exception):
    """The score exists but its replay cannot be analysed.

    Raised when the ``.osr`` is missing on disk or present but undecodable --
    distinct from the score row itself not existing (which the query methods
    signal with ``None``), so the endpoint can tell "wrong id" from "no replay".
    """


@dataclass(frozen=True, slots=True)
class SimilarityMatch:
    """One candidate replay compared against the score under analysis.

    ``mean_distance`` / ``median_distance`` are the D3 cursor distances (osu!px;
    lower is more identical) and ``signal`` is the graded steal verdict for this
    pair. Only comparable pairs (a shared timeline) become matches, so the
    distances here are always finite.
    """

    score_id: int
    user_id: int
    mean_distance: float
    median_distance: float
    overlap_points: int
    signal: DetectionSignal


@dataclass(frozen=True, slots=True)
class SimilaritySummary:
    """The D3 outcome for one score: its nearest same-map replays, closest first.

    ``compared`` is how many candidates were actually comparable (had a loadable,
    decodable, time-overlapping replay); ``matches`` is the closest ``top_k`` of
    them, most-similar first, so a reviewer sees the likeliest steal at the top.
    """

    score_id: int
    compared: int
    matches: list[SimilarityMatch]


@dataclass(frozen=True)
class ReplayAnalysisQueryService:
    read_replay_bytes: ReplayBytesReader
    fetch_score: ScoreReader
    fetch_candidates: CandidateFinder
    config: DetectorConfig = DEFAULT_CONFIG
    # how many same-map replays to load and compare against at most -- bounds the
    # `.osr` reads one similarity request does. Candidates are the leaderboard
    # top (score desc), so the truncation keeps the plays most worth comparing.
    max_candidates: int = 50
    # how many of the closest matches to return; a steal shows up as one or two
    # near-zero distances, so the long tail of unrelated plays is not returned.
    top_k: int = 10

    async def _load_replay(self, score_id: int, *, mode: int) -> Replay:
        """Read and decode one stored replay, or raise ``ReplayUnavailableError``."""
        raw = await self.read_replay_bytes(score_id)
        if raw is None:
            raise ReplayUnavailableError(f"no replay on disk for score {score_id}")
        try:
            return parse_replay_frame_data(raw, mode=mode)
        except ReplayParseError as exc:
            raise ReplayUnavailableError(
                f"replay for score {score_id} could not be decoded: {exc}",
            ) from exc

    async def analysis_series(self, score_id: int) -> dict[str, Any] | None:
        """The graph-ready series bundle for one score, or ``None`` if unknown.

        Recomputed live from the retained ``.osr``; the ``clock_rate`` is derived
        from the score's real mods so the frontend can show frametimes in real
        time. Raises ``ReplayUnavailableError`` if the score exists but its
        replay is missing or undecodable.
        """
        score = await self.fetch_score(score_id)
        if score is None:
            return None
        replay = await self._load_replay(score_id, mode=score.mode)
        feats = extract_features(replay)
        return build_analysis_series(
            replay,
            feats,
            clock_rate=_clock_rate(score.mods),
        )

    async def similarity(self, score_id: int) -> SimilaritySummary | None:
        """Nearest same-map replays to one score by D3 cursor distance.

        Loads the score's replay, then every candidate the finder returns, runs
        the pairwise similarity check, and returns the closest ``top_k`` as
        graded matches. A candidate whose replay is missing, undecodable, or
        shares no time window with this one is simply skipped (not comparable),
        never fatal. Returns ``None`` only if the score row itself is unknown;
        raises ``ReplayUnavailableError`` if *this* score's replay is unusable.
        """
        score = await self.fetch_score(score_id)
        if score is None:
            return None
        replay_a = await self._load_replay(score_id, mode=score.mode)

        candidates = await self.fetch_candidates(
            map_md5=score.map_md5,
            mode=score.mode,
            exclude_score_id=score_id,
            exclude_user_id=score.userid,
            limit=self.max_candidates,
        )

        matches: list[SimilarityMatch] = []
        for candidate in candidates:
            try:
                replay_b = await self._load_replay(candidate.score_id, mode=score.mode)
            except ReplayUnavailableError:
                # a candidate whose replay is gone or corrupt just isn't
                # comparable; skip it rather than fail the whole report.
                continue

            result = replay_similarity(replay_a, replay_b)
            if result.n == 0:
                # no shared timeline -> not comparable, and its distance is inf
                # (not JSON-safe). Drop it instead of surfacing a non-match.
                continue

            signal = detect_replay_similarity(
                result,
                self.config,
                other_score_id=candidate.score_id,
            )
            matches.append(
                SimilarityMatch(
                    score_id=candidate.score_id,
                    user_id=candidate.user_id,
                    mean_distance=result.mean_distance,
                    median_distance=result.median_distance,
                    overlap_points=result.n,
                    signal=signal,
                ),
            )

        # closest first: the likeliest steal sits at the top.
        matches.sort(key=lambda match: match.mean_distance)
        return SimilaritySummary(
            score_id=score_id,
            compared=len(matches),
            matches=matches[: self.top_k],
        )
