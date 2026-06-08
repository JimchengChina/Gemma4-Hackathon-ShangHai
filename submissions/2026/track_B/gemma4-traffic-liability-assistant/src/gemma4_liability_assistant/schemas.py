from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceFrame:
    image_path: str
    timestamp_sec: float
    description: str = ""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiabilityResult:
    case_id: str
    accident_detected: bool
    parties: list[str]
    liability_ratio: dict[str, float]
    liability_bucket: str
    evidence_chain: list[dict[str, Any]]
    supporting_articles: list[dict[str, Any]]
    uncertainties: list[str]
    human_review_required: bool
    model_opinion: str
