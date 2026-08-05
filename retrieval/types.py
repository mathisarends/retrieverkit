from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

type Embedding = list[float]


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    parent_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    document: Document
    score: float
