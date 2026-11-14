from __future__ import annotations

from app.db.models import InsuranceRecord
from app.repositories.insurance_repository import InsuranceRepository


class InsuranceService:
    def __init__(self, repository: InsuranceRepository | None = None) -> None:
        self.repository = repository or InsuranceRepository()

    async def create_policy(
        self,
        lead_id: str,
        policy_type: str,
        coverage_amount: float,
        **kwargs: str,
    ) -> InsuranceRecord:
        records = await self.repository.list()
        record = InsuranceRecord(
            id=kwargs.get("id", f"policy-{len(records) + 1}"),
            lead_id=lead_id,
            policy_type=policy_type,
            coverage_amount=coverage_amount,
            status=kwargs.get("status", "pending"),
            metadata=kwargs.get("metadata", {}),
        )
        return await self.repository.create(record)

    async def get_policies_for_lead(self, lead_id: str) -> list[InsuranceRecord]:
        return await self.repository.get_by_lead_id(lead_id)
