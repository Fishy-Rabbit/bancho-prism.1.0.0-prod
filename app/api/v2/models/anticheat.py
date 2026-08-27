from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import Literal

from . import BaseModel

# output models


class AnticheatFlagModel(BaseModel):
    """A staff-facing view of one flagged score.

    Mirrors the repository's ``AnticheatFlag`` one-for-one. ``status`` and
    ``severity`` are surfaced as plain strings (the repository's ``severity`` is
    already an opaque string, and ``AnticheatFlagStatus`` is a ``StrEnum`` that
    validates cleanly against ``str``), so a reviewer's client is not coupled to
    the detector layer's enums. ``evidence`` is the raw numbers behind the top
    signal, or null if none were recorded.
    """

    score_id: int
    user_id: int
    mode: int
    status: str
    severity: str
    top_signal_code: str
    top_signal_title: str
    confidence: float
    triggered_count: int
    detail: str
    evidence: dict[str, Any] | None
    first_flagged_at: datetime
    last_flagged_at: datetime
    resolved_by: int | None
    resolved_at: datetime | None
    resolution_note: str | None


class ReplaySimilarityMatchModel(BaseModel):
    """One same-map replay compared against a score, with its D3 (steal) verdict.

    Mirrors ``analysis_query.SimilarityMatch``: the compared score's id + owner,
    the cursor distances (osu!px; lower is more identical), the count of aligned
    points behind them, and the graded signal flattened to the same
    ``severity``/``flagged``/``detail``/``evidence`` shape as ``AnticheatFlagModel``
    so a reviewer's client reads one signal representation everywhere.
    """

    score_id: int
    user_id: int
    mean_distance: float
    median_distance: float
    overlap_points: int
    severity: str
    flagged: bool
    detail: str
    evidence: dict[str, float]


class ReplaySimilarityModel(BaseModel):
    """The nearest same-map replays to one score, most-similar first.

    ``compared`` is how many candidates were actually comparable; ``matches`` is
    the closest handful, so a reviewer opens on the likeliest steal.
    """

    score_id: int
    compared: int
    matches: list[ReplaySimilarityMatchModel]


# input models


class ResolveFlagRequest(BaseModel):
    """A staff member's decision on a flag.

    ``status`` deliberately excludes ``"open"``: reopening is not a decision a
    reviewer records here (a fresh flag is already ``OPEN``), so the only
    transitions offered are claiming it (``reviewing``) or a terminal verdict
    (``actioned`` / ``dismissed``). Rejecting ``"open"`` at parse time keeps the
    invalid transition out of the service entirely.
    """

    status: Literal["reviewing", "actioned", "dismissed"]
    note: str | None = None
