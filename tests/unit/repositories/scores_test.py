"""Guards for the scores repository's anticheat-facing query.

Only ``fetch_replay_candidates`` is covered here -- the D3 (replay-steal)
comparison pool. The properties that make the pool safe to compare against:

- it deserializes rows into ``ReplayCandidate`` faithfully;
- it restricts to ``BEST`` rows on the requested beatmap+mode and excludes the
  score under analysis, ordered by raw score so a stolen top play stays in a
  truncated pool;
- the owner-exclusion is applied only when asked (comparing a player against
  their own earlier play is not a steal).

No DB is involved: a fake compiles the statement the repository builds and serves
canned rows, so the test exercises the repository's own SQL, not MySQL.
"""

from __future__ import annotations

from typing import Any

from app.constants.score_statuses import SubmissionStatus
from app.repositories.scores import ReplayCandidate
from app.repositories.scores import ScoresRepository


class _FakeDatabase:
    """Compiles and records the read statement, then serves canned rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.sql = ""
        self.params: dict[str, Any] = {}

    async def fetch_all(self, statement: Any) -> list[dict[str, Any]]:
        compiled = statement.compile()
        self.sql = str(compiled)
        self.params = dict(compiled.params)
        return self.rows


def _repo(rows: list[dict[str, Any]]) -> tuple[ScoresRepository, _FakeDatabase]:
    database = _FakeDatabase(rows)
    return ScoresRepository(database), database  # type: ignore[arg-type]


async def test_fetch_replay_candidates_deserializes_rows() -> None:
    repository, _ = _repo(
        [
            {"score_id": 10, "user_id": 2, "mods": 8},
            {"score_id": 11, "user_id": 3, "mods": 0},
        ],
    )

    candidates = await repository.fetch_replay_candidates(
        map_md5="a" * 32,
        mode=0,
        exclude_score_id=100,
    )

    assert candidates == [
        ReplayCandidate(score_id=10, user_id=2, mods=8),
        ReplayCandidate(score_id=11, user_id=3, mods=0),
    ]


async def test_fetch_replay_candidates_filters_orders_and_limits() -> None:
    repository, database = _repo([])

    await repository.fetch_replay_candidates(
        map_md5="f" * 32,
        mode=3,
        exclude_score_id=1234,
    )

    values = set(database.params.values())
    # the beatmap, mode, BEST status, and the excluded score are all bound.
    assert "f" * 32 in values
    assert 3 in values
    assert SubmissionStatus.BEST.value in values
    assert 1234 in values
    # top-of-leaderboard first, and bounded.
    assert "ORDER BY scores.score DESC" in database.sql
    assert "LIMIT" in database.sql.upper()


async def test_fetch_replay_candidates_applies_owner_exclusion_when_given() -> None:
    repository, database = _repo([])

    await repository.fetch_replay_candidates(
        map_md5="a" * 32,
        mode=0,
        exclude_score_id=1,
        exclude_user_id=777,
    )

    # SELECT references scores.userid once (the label); the exclusion adds a
    # second reference in the WHERE.
    assert database.sql.count("scores.userid") == 2
    assert 777 in set(database.params.values())


async def test_fetch_replay_candidates_omits_owner_exclusion_by_default() -> None:
    repository, database = _repo([])

    await repository.fetch_replay_candidates(
        map_md5="a" * 32,
        mode=0,
        exclude_score_id=1,
    )

    # only the SELECT label references scores.userid -- no WHERE inequality.
    assert database.sql.count("scores.userid") == 1
