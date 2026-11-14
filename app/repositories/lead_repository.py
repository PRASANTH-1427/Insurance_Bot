from __future__ import annotations

from app.db.models import LeadRecord


class LeadRepository:
    def __init__(self) -> None:
        self._records: list[LeadRecord] = []

    async def create(self, lead: LeadRecord) -> LeadRecord:
        self._records.append(lead)
        return lead

    async def get_by_id(self, lead_id: str) -> LeadRecord | None:
        for lead in self._records:
            if lead.id == lead_id:
                return lead
        return None

    async def list(self) -> list[LeadRecord]:
        return list(self._records)
