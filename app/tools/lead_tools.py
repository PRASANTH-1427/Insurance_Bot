"""
Lead Tools
==========
LangChain @tool-decorated functions used by the LeadAgent.

Each tool is a function that the LLM can call via tool-calling.
Module-level service singletons are used so data persists within a process.
"""

from __future__ import annotations

import re
import uuid

from langchain_core.tools import tool

from app.db.models import LeadRecord
from app.repositories.lead_repository import LeadRepository

# Module-level singleton so all tool calls share the same in-memory store
_lead_repo = LeadRepository()


@tool
def validate_email(email: str) -> dict:
    """Validate whether an email address has the correct format.

    Args:
        email: The email address string to validate.

    Returns:
        A dict with keys ``is_valid`` (bool) and ``message`` (str).
    """
    pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
    is_valid = bool(re.match(pattern, email.strip()))
    return {
        "email": email,
        "is_valid": is_valid,
        "message": "Email format is valid." if is_valid else "Invalid email format. Please provide a valid email (e.g. name@example.com).",
    }


@tool
def validate_phone(phone: str) -> dict:
    """Validate whether a phone number has the correct format.

    Accepts digits, spaces, hyphens, parentheses, and a leading +.
    Requires 7–15 digits total.

    Args:
        phone: The phone number string to validate.

    Returns:
        A dict with keys ``is_valid`` (bool) and ``message`` (str).
    """
    cleaned = re.sub(r"[\s\-\(\)\+]", "", phone.strip())
    is_valid = cleaned.isdigit() and 7 <= len(cleaned) <= 15
    return {
        "phone": phone,
        "is_valid": is_valid,
        "message": "Phone number format is valid." if is_valid else "Invalid phone number. Please provide a valid number (e.g. +91-9876543210).",
    }


@tool
def qualify_lead(name: str, email: str) -> dict:
    """Check whether a lead is qualified based on name and email validity.

    A lead is 'qualified' if both name is non-empty and email contains '@'.
    Otherwise it is marked 'needs_review'.

    Args:
        name: The lead's full name.
        email: The lead's email address.

    Returns:
        A dict with keys ``status`` and ``message``.
    """
    if name.strip() and "@" in email:
        return {"status": "qualified", "message": f"{name} is a qualified lead."}
    return {"status": "needs_review", "message": "Lead needs review — name or email may be incomplete."}


@tool
async def save_lead(name: str, email: str, phone: str, interest: str) -> dict:
    """Save a new insurance lead to the database.

    Call this only after all four fields have been collected and validated.

    Args:
        name: Full name of the lead.
        email: Email address (must be valid format).
        phone: Phone number (must be valid format).
        interest: Type of insurance interested in (e.g. auto, health, life, home).

    Returns:
        A dict with ``lead_id``, ``status``, and ``message``.
    """
    lead_id = f"LEAD-{uuid.uuid4().hex[:8].upper()}"
    record = LeadRecord(
        id=lead_id,
        name=name.strip(),
        email=email.strip().lower(),
        phone=phone.strip(),
        status="new",
        metadata={"interest": interest.strip().lower()},
    )
    await _lead_repo.create(record)
    return {
        "lead_id": lead_id,
        "status": "saved",
        "message": f"Lead successfully created for {name}. Our team will contact you at {email} shortly.",
    }
