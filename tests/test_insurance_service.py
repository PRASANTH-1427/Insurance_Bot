import asyncio
from app.services.insurance_service import InsuranceService


def test_create_policy() -> None:
    service = InsuranceService()
    policy = asyncio.run(service.create_policy("lead-1", "auto", 250000.0))

    assert policy.lead_id == "lead-1"
    assert policy.policy_type == "auto"
    assert policy.coverage_amount == 250000.0
    assert policy.status == "pending"
