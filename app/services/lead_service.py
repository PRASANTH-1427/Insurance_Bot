from __future__ import annotations

from app.db.models import LeadRecord
from app.repositories.lead_repository import LeadRepository


class LeadService:
    def __init__(self, repository: LeadRepository | None = None) -> None:
        self.repository = repository or LeadRepository()

    async def create_lead(self, name: str, email: str, phone: str, **kwargs: str) -> LeadRecord:
        records = await self.repository.list()
        lead = LeadRecord(
            id=kwargs.get("id", f"lead-{len(records) + 1}"),
            name=name,
            email=email,
            phone=phone,
            status=kwargs.get("status", "new"),
            metadata=kwargs.get("metadata", {}),
        )
        return await self.repository.create(lead)

    async def get_lead(self, lead_id: str) -> LeadRecord | None:
        return await self.repository.get_by_id(lead_id)

    async def list_leads(self) -> list[LeadRecord]:
        return await self.repository.list()
