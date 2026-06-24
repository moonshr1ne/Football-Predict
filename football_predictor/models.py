from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MatchRecord:
    date: str
    home_team: str
    away_team: str
    home_goals: int | None = None
    away_goals: int | None = None
    home_corners: float | None = None
    away_corners: float | None = None
    competition: str = ""
    stage: str = ""
    neutral: bool = True
    source: str = "manual"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchRecord":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "home_corners": self.home_corners,
            "away_corners": self.away_corners,
            "competition": self.competition,
            "stage": self.stage,
            "neutral": self.neutral,
            "source": self.source,
        }

    def involves(self, team: str) -> bool:
        return self.home_team == team or self.away_team == team

    def is_finished(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    def goals_for(self, team: str) -> int | None:
        if not self.is_finished():
            return None
        return self.home_goals if self.home_team == team else self.away_goals

    def goals_against(self, team: str) -> int | None:
        if not self.is_finished():
            return None
        return self.away_goals if self.home_team == team else self.home_goals

    def corners_for(self, team: str) -> float | None:
        if self.home_team == team:
            return self.home_corners
        if self.away_team == team:
            return self.away_corners
        return None

    def corners_against(self, team: str) -> float | None:
        if self.home_team == team:
            return self.away_corners
        if self.away_team == team:
            return self.home_corners
        return None


@dataclass
class TeamStats:
    team: str
    sample_size: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    corners_for: float = 0.0
    corners_against: float = 0.0
    corner_samples: int = 0
    clean_sheets: int = 0
    failed_to_score: int = 0
    recent: list[MatchRecord] = field(default_factory=list)

    @property
    def avg_goals_for(self) -> float:
        return self.goals_for / self.sample_size if self.sample_size else 1.15

    @property
    def avg_goals_against(self) -> float:
        return self.goals_against / self.sample_size if self.sample_size else 1.15

    @property
    def points_per_match(self) -> float:
        return (self.wins * 3 + self.draws) / self.sample_size if self.sample_size else 1.35

    @property
    def avg_corners_for(self) -> float | None:
        return self.corners_for / self.corner_samples if self.corner_samples else None

    @property
    def avg_corners_against(self) -> float | None:
        return self.corners_against / self.corner_samples if self.corner_samples else None

    @property
    def avg_total_corners(self) -> float | None:
        if not self.corner_samples:
            return None
        return (self.corners_for + self.corners_against) / self.corner_samples

    def as_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "sample_size": self.sample_size,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "avg_goals_for": round(self.avg_goals_for, 2),
            "avg_goals_against": round(self.avg_goals_against, 2),
            "points_per_match": round(self.points_per_match, 2),
            "avg_corners_for": None if self.avg_corners_for is None else round(self.avg_corners_for, 2),
            "avg_corners_against": None if self.avg_corners_against is None else round(self.avg_corners_against, 2),
            "avg_total_corners": None if self.avg_total_corners is None else round(self.avg_total_corners, 2),
            "clean_sheets": self.clean_sheets,
            "failed_to_score": self.failed_to_score,
            "recent": [match.to_dict() for match in self.recent],
        }
