from __future__ import annotations

from pydantic import BaseModel


class InsuranceQuoteRequest(BaseModel):
    lead_id: str
    policy_type: str
    coverage_amount: float


class InsuranceQuoteResponse(BaseModel):
    id: str
    lead_id: str
    policy_type: str
    coverage_amount: float
    status: str = "pending"
