from __future__ import annotations

from pydantic import BaseModel, EmailStr


class LeadCreate(BaseModel):
    name: str
    email: EmailStr
    phone: str
    source: str = "website"


class LeadResponse(BaseModel):
    id: str
    name: str
    email: EmailStr
    phone: str
    status: str = "new"
