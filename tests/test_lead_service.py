import asyncio
from app.services.lead_service import LeadService


def test_create_lead() -> None:
    service = LeadService()
    lead = asyncio.run(service.create_lead("Jane Doe", "jane@example.com", "555-1234"))

    assert lead.name == "Jane Doe"
    assert lead.email == "jane@example.com"
    assert lead.phone == "555-1234"
    assert lead.status == "new"
