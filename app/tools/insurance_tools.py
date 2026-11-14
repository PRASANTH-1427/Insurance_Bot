"""
Insurance Tools
===============
LangChain @tool-decorated functions used by the InsuranceAgent.

Module-level repository singleton keeps data consistent within a process.
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.repositories.insurance_repository import InsuranceRepository

# Module-level singleton so all tool calls share the same in-memory store
_insurance_repo = InsuranceRepository()

# Allowed fields that can be updated on a policy
_UPDATABLE_FIELDS = {"contact_email", "contact_phone", "coverage_amount", "beneficiary", "status"}


@tool
def list_policy_types() -> dict:
    """Return all available insurance policy types offered by InsureAssist.

    Returns:
        A dict with ``policy_types`` (list of strings) and ``descriptions``.
    """
    return {
        "policy_types": ["auto", "health", "life", "home"],
        "descriptions": {
            "auto": "Covers vehicle damage and third-party liability.",
            "health": "Covers medical expenses and hospitalisation.",
            "life": "Provides a payout to beneficiaries on the policyholder's death.",
            "home": "Covers home structure and contents against damage or theft.",
        },
    }


@tool
async def lookup_policy(policy_number: str) -> dict:
    """Look up an existing insurance policy by its policy number.

    Args:
        policy_number: The unique policy identifier (e.g. LEAD-XXXXXXXX or policy-N).

    Returns:
        A dict with ``found`` (bool) and ``policy`` details if found.
    """
    record = await _insurance_repo.get_by_id(policy_number.strip())
    if record is None:
        return {
            "found": False,
            "policy_number": policy_number,
            "message": f"No policy found with number '{policy_number}'. Please double-check and try again.",
        }
    return {
        "found": True,
        "policy_number": record.id,
        "lead_id": record.lead_id,
        "policy_type": record.policy_type,
        "coverage_amount": record.coverage_amount,
        "status": record.status,
        "metadata": record.metadata,
    }


@tool
async def update_policy_field(policy_number: str, field: str, new_value: str) -> dict:
    """Update a specific field on an existing insurance policy.

    Allowed fields: contact_email, contact_phone, coverage_amount, beneficiary, status.

    Args:
        policy_number: The unique policy identifier.
        field: The name of the field to update (must be in the allowed list).
        new_value: The new value to set for the field.

    Returns:
        A dict with ``success`` (bool), ``field``, ``new_value``, and ``message``.
    """
    if field not in _UPDATABLE_FIELDS:
        return {
            "success": False,
            "field": field,
            "message": f"'{field}' is not an updatable field. Allowed fields: {', '.join(sorted(_UPDATABLE_FIELDS))}.",
        }

    record = await _insurance_repo.get_by_id(policy_number.strip())
    if record is None:
        return {
            "success": False,
            "field": field,
            "message": f"Policy '{policy_number}' not found.",
        }

    # Store custom fields in metadata dict; coverage_amount updates the actual field
    if field == "coverage_amount":
        try:
            record.coverage_amount = float(new_value)
        except ValueError:
            return {"success": False, "field": field, "message": "coverage_amount must be a numeric value."}
    else:
        if record.metadata is None:
            record.metadata = {}
        record.metadata[field] = new_value

    return {
        "success": True,
        "policy_number": policy_number,
        "field": field,
        "new_value": new_value,
        "message": f"Policy {policy_number} updated: {field} set to '{new_value}'.",
    }


@tool
def estimate_premium(policy_type: str, coverage_amount: float) -> dict:
    """Estimate the annual insurance premium for a given policy type and coverage amount.

    Args:
        policy_type: One of auto, home, life, health.
        coverage_amount: The total coverage amount in local currency.

    Returns:
        A dict with ``policy_type``, ``coverage_amount``, ``estimated_annual_premium``, and ``rate_used``.
    """
    base_rates: dict[str, float] = {
        "auto": 0.04,
        "home": 0.03,
        "life": 0.02,
        "health": 0.05,
    }
    rate = base_rates.get(policy_type.lower(), 0.05)
    premium = round(coverage_amount * rate, 2)
    return {
        "policy_type": policy_type,
        "coverage_amount": coverage_amount,
        "rate_used": rate,
        "estimated_annual_premium": premium,
        "message": f"Estimated annual premium for {policy_type} insurance on {coverage_amount} coverage is {premium}.",
    }
