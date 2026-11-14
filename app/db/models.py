from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LeadRecord:
    id: str | None = None
    name: str = ""
    email: str = ""
    phone: str = ""
    status: str = "new"
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InsuranceRecord:
    id: str | None = None
    lead_id: str | None = None
    policy_type: str = ""
    coverage_amount: float = 0.0
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
