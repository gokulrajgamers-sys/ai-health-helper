from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field

from safety_privacy import get_disclaimer


# --- Auth ---

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class UserOut(BaseModel):
    id: str
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# --- Symptom checking pipeline ---

class SymptomCheckRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Free-text description of symptoms")


class ConditionMatchOut(BaseModel):
    condition_id: str
    name: str
    confidence: float
    matched_symptoms: list[str]
    severity: str
    description: str
    advice: str


class SymptomCheckResponse(BaseModel):
    id: Optional[str] = None
    extracted_symptoms: list[str]
    risk_level: str = Field(description="'low' | 'moderate' | 'urgent' -- from the Risk Assessment stage")
    red_flags: list[str] = Field(default_factory=list)
    is_emergency: bool
    matches: list[ConditionMatchOut]
    recommendation: str = Field(description="Headline guidance from the Recommendation Engine")
    disclaimer: str = Field(default_factory=get_disclaimer)
