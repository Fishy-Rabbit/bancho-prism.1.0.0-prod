"""Guards for the on-demand staff replay-analysis service.

This service is the interactive, score-row-backed counterpart to the worker's
one-shot ``ReplayAnalysisService``. Two behaviours are pinned:

- **analysis_series** recomputes the graph bundle live from the retained
  ``.osr`` and stamps the *real* ``clock_rate`` from the score's mods -- the
  thing the worker (which sees ``mods=0`` on the bare frame stream) cannot do;
- **similarity** runs D3 pairwise against the same-map candidates the finder
  returns, skips any that are unreadable/undecodable/non-overlapping, and returns
  the closest ``top_k`` most-similar first.

Both impure edges are injected, so there is no filesystem and no database: a dict
of ``.osr`` bytes stands in for the replay volume, a canned row for the score,
and a recording fake for the candidate lookup. The pipeline in between runs for
real against genuinely LZMA-compressed frames.
"""

from __future__ import annotations

import json
import lzma
from dataclasses import dataclass
from typing import Any

import pytest

from app.constants.mods import Mods
from app.services.anticheat.analysis_query import ReplayAnalysisQueryService
from app.services.anticheat.analysis_query import ReplayUnavailableError
from app.services.anticheat.analysis_query import _clock_rate
from app.services.anticheat.detectors import Severity


def _compress_frames(frame_tokens: list[str]) -> bytes:
    """LZMA1 alone-format frame stream -- exactly what lands on disk."""
    raw = ",".join(frame_tokens).encode("ascii")
    filters = [{"id": lzma.FILTER_LZMA1, "preset": 6}]
    return lzma.compress(raw, format=lzma.FORMAT_ALONE, filters=filters)


def _drift_bytes(*, n: int = 600, dx: float = 0.0, dy: float = 0.0, lead_in: int = 0) -> bytes:
    """A smooth diagonal cursor drift as a compressed ``.osr`` frame stream.

    ``n`` 16 ms frames give ``n`` aligned grid points against an identically
    timed replay, so ``n >= 500`` clears D3's overlap gate. ``dx``/``dy`` shift
    the whole path rigidly (a known constant distance per point); ``lead_in``
    prepends a single large-delta frame to push the whole window later in time
    (used to build a replay that never overlaps another).
    """
    tokens: list[str] = []
    if lead_in:
        tokens.append(f"{lead_in}|0.0|0.0|0")
    tokens += [f"16|{i * 1.3 + dx}|{i * 0.7 + dy}|0" for i in range(n)]
    return _compress_frames(tokens)


@dataclass(frozen=True)
class _FakeScore:
    map_md5: str = "a" * 32
    mode: int = 0
    mods: int = 0
    userid: int = 1


@dataclass(frozen=True)
class _FakeCandidate:
    score_id: int
    user_id: int
    mods: int = 0


class _CandidateFinder:
    """Records the lookup kwargs and returns a canned candidate list."""

    def __init__(self, candidates: list[_FakeCandidate]) -> None:
        self._candidates = candidates
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        map_md5: str,
        mode: int,
        exclude_score_id: int,
        exclude_user_id: int | None,
        limit: int,
    ) -> list[_FakeCandidate]:
        self.calls.append(
            {
                "map_md5": map_md5,
                "mode": mode,
                "exclude_score_id": exclude_score_id,
                "exclude_user_id": exclude_user_id,
                "limit": limit,
            },
        )
        return list(self._candidates)


def _svc(
    *,
    replays: dict[int, bytes],
    score: _FakeScore | None,
    finder: _CandidateFinder | None = None,
    **kwargs: Any,
) -> ReplayAnalysisQueryService:
    async def read_replay_bytes(score_id: int) -> bytes | None:
        return replays.get(score_id)

    async def fetch_score(score_id: int) -> _FakeScore | None:
        return score

    return ReplayAnalysisQueryService(
        read_replay_bytes=read_replay_bytes,
        fetch_score=fetch_score,
        fetch_candidates=finder or _CandidateFinder([]),
        **kwargs,
    )


# --- clock rate -------------------------------------------------------------


def test_clock_rate_maps_mods_to_speed_multiplier() -> None:
    assert _clock_rate(0) == 1.0
    assert _clock_rate(int(Mods.DOUBLETIME)) == 1.5
    assert _clock_rate(int(Mods.NIGHTCORE)) == 1.5
    assert _clock_rate(int(Mods.HALFTIME)) == 0.75
    # a speed-up combined with other mods still reads 1.5 (HD|DT).
    assert _clock_rate(int(Mods.HIDDEN | Mods.DOUBLETIME)) == 1.5


# --- analysis series --------------------------------------------------------


async def test_analysis_series_bundles_and_stamps_real_clock_rate() -> None:
    # the score carries DT, which the worker's bare frame stream cannot know;
    # this service reads it off the row and stamps 1.5 onto the bundle.
    score = _FakeScore(mods=int(Mods.DOUBLETIME))
    service = _svc(replays={100: _drift_bytes()}, score=score)

    bundle = await service.analysis_series(100)

    assert bundle is not None
    assert bundle["clock_rate"] == 1.5
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
    json.dumps(bundle)  # the whole bundle must serialize for the API envelope.


async def test_analysis_series_is_none_when_score_unknown() -> None:
    service = _svc(replays={}, score=None)
    assert await service.analysis_series(404) is None


async def test_analysis_series_raises_when_replay_missing() -> None:
    # score row exists, but its `.osr` never landed on disk.
    service = _svc(replays={}, score=_FakeScore())
    with pytest.raises(ReplayUnavailableError):
        await service.analysis_series(100)


async def test_analysis_series_raises_when_replay_undecodable() -> None:
    service = _svc(replays={100: b"\xff\xff not lzma"}, score=_FakeScore())
    with pytest.raises(ReplayUnavailableError):
        await service.analysis_series(100)


# --- similarity -------------------------------------------------------------


async def test_similarity_flags_an_identical_candidate_replay() -> None:
    finder = _CandidateFinder([_FakeCandidate(score_id=10, user_id=2)])
    service = _svc(
        replays={100: _drift_bytes(), 10: _drift_bytes()},  # identical paths
        score=_FakeScore(),
        finder=finder,
    )

    summary = await service.similarity(100)

    assert summary is not None
    assert summary.compared == 1
    assert len(summary.matches) == 1
    match = summary.matches[0]
    assert match.score_id == 10
    assert match.mean_distance < 1e-9
    assert match.signal.severity == Severity.HIGH
    assert match.signal.flagged is True
    json.dumps(match.signal.evidence)  # evidence stays JSON-safe.


async def test_similarity_does_not_flag_a_divergent_candidate() -> None:
    finder = _CandidateFinder([_FakeCandidate(score_id=10, user_id=2)])
    service = _svc(
        replays={100: _drift_bytes(), 10: _drift_bytes(dx=200.0)},  # 200px apart
        score=_FakeScore(),
        finder=finder,
    )

    summary = await service.similarity(100)

    assert summary is not None
    assert summary.compared == 1
    assert summary.matches[0].signal.severity == Severity.NONE
    assert summary.matches[0].signal.flagged is False


async def test_similarity_skips_a_candidate_with_no_replay_on_disk() -> None:
    finder = _CandidateFinder([_FakeCandidate(score_id=10, user_id=2)])
    # candidate 10 has no bytes -> not comparable, skipped (not fatal).
    service = _svc(replays={100: _drift_bytes()}, score=_FakeScore(), finder=finder)

    summary = await service.similarity(100)

    assert summary is not None
    assert summary.compared == 0
    assert summary.matches == []


async def test_similarity_skips_an_undecodable_candidate() -> None:
    finder = _CandidateFinder([_FakeCandidate(score_id=10, user_id=2)])
    service = _svc(
        replays={100: _drift_bytes(), 10: b"\xff\xff garbage"},
        score=_FakeScore(),
        finder=finder,
    )

    summary = await service.similarity(100)

    assert summary is not None
    assert summary.compared == 0


async def test_similarity_skips_a_non_overlapping_candidate() -> None:
    finder = _CandidateFinder([_FakeCandidate(score_id=10, user_id=2)])
    # candidate's window starts long after this replay's has ended -> n == 0.
    service = _svc(
        replays={100: _drift_bytes(), 10: _drift_bytes(lead_in=500_000)},
        score=_FakeScore(),
        finder=finder,
    )

    summary = await service.similarity(100)

    assert summary is not None
    assert summary.compared == 0


async def test_similarity_orders_matches_closest_first() -> None:
    finder = _CandidateFinder(
        [
            _FakeCandidate(score_id=11, user_id=3),  # far
            _FakeCandidate(score_id=10, user_id=2),  # identical
        ],
    )
    service = _svc(
        replays={
            100: _drift_bytes(),
            10: _drift_bytes(),  # identical -> ~0
            11: _drift_bytes(dx=50.0),  # 50px away
        },
        score=_FakeScore(),
        finder=finder,
    )

    summary = await service.similarity(100)

    assert summary is not None
    assert summary.compared == 2
    assert [m.score_id for m in summary.matches] == [10, 11]
    assert summary.matches[0].mean_distance < summary.matches[1].mean_distance


async def test_similarity_caps_matches_to_top_k_but_counts_all_compared() -> None:
    finder = _CandidateFinder(
        [_FakeCandidate(score_id=10, user_id=2), _FakeCandidate(score_id=11, user_id=3)],
    )
    service = _svc(
        replays={100: _drift_bytes(), 10: _drift_bytes(), 11: _drift_bytes(dx=5.0)},
        score=_FakeScore(),
        finder=finder,
        top_k=1,
    )

    summary = await service.similarity(100)

    assert summary is not None
    assert summary.compared == 2  # both were compared
    assert len(summary.matches) == 1  # but only the closest is returned
    assert summary.matches[0].score_id == 10


async def test_similarity_excludes_the_score_and_its_owner_from_candidates() -> None:
    finder = _CandidateFinder([])
    service = _svc(
        replays={100: _drift_bytes()},
        score=_FakeScore(userid=77),
        finder=finder,
        max_candidates=25,
    )

    await service.similarity(100)

    assert finder.calls == [
        {
            "map_md5": "a" * 32,
            "mode": 0,
            "exclude_score_id": 100,
            "exclude_user_id": 77,
            "limit": 25,
        },
    ]


async def test_similarity_is_none_when_score_unknown() -> None:
    service = _svc(replays={}, score=None)
    assert await service.similarity(404) is None


async def test_similarity_raises_when_the_scores_own_replay_is_missing() -> None:
    # the score row exists but its replay is gone: nothing to compare from.
    service = _svc(replays={}, score=_FakeScore())
    with pytest.raises(ReplayUnavailableError):
        await service.similarity(100)
