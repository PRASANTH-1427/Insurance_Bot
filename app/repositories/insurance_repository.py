from __future__ import annotations

from app.db.models import InsuranceRecord


class InsuranceRepository:
    def __init__(self) -> None:
        self._records: list[InsuranceRecord] = []

    async def create(self, record: InsuranceRecord) -> InsuranceRecord:
        self._records.append(record)
        return record

    async def get_by_id(self, record_id: str) -> InsuranceRecord | None:
        return next((r for r in self._records if r.id == record_id), None)

    async def get_by_lead_id(self, lead_id: str) -> list[InsuranceRecord]:
        return [item for item in self._records if item.lead_id == lead_id]

    async def update(self, record_id: str, **fields) -> InsuranceRecord | None:
        record = await self.get_by_id(record_id)
        if record is None:
            return None
        for key, value in fields.items():
            if hasattr(record, key):
                setattr(record, key, value)
        return record

    async def list(self) -> list[InsuranceRecord]:
        return list(self._records)
