"""
AI Health Symptom Assistant - FastAPI backend.

Orchestrates the pipeline exactly as laid out in the architecture diagram:

    User Interface
        -> NLP & Symptom Extraction   (nlp_extraction.py)
        -> AI Symptom Analysis        (symptom_analysis.py + knowledge_base.json)
        -> Risk Assessment            (risk_assessment.py)
        -> Recommendation Engine      (recommendation_engine.py)
        -> User (personalized, safety-focused response)

Safety & Privacy (safety_privacy.py) is cross-cutting -- it's not a pipeline
step, its helpers are called from within /symptoms/check and from auth.py /
models.py for the disclaimer, consent notice, and log redaction.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload

Docs:
    http://localhost:8000/docs
"""

import json
import logging
from datetime import timedelta

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

import auth
import nlp_extraction
import risk_assessment
import recommendation_engine
import safety_privacy
from symptom_analysis import analyzer
from models import SymptomCheck, User, get_db, init_db
from schemas import (
    ConditionMatchOut,
    SymptomCheckRequest,
    SymptomCheckResponse,
    Token,
    UserCreate,
    UserOut,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("health-assistant")

app = FastAPI(
    title="AI Health Symptom Assistant API",
    description="Symptom-checking API. Informational use only -- not a substitute for professional medical advice.",
    version="2.0.0",
)

# CORS - lock this down to your real frontend origin(s) in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production, e.g. ["https://yourapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Database initialized.")


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------------- Auth ----------------

@app.post("/auth/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=payload.email, hashed_password=auth.hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth.create_access_token(
        data={"sub": user.id}, expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return Token(access_token=token)


@app.get("/auth/me", response_model=UserOut)
def read_me(current_user: User = Depends(auth.require_user)):
    return current_user


# ---------------- Symptom checking pipeline ----------------

def _run_pipeline(raw_text: str):
    """
    The full diagram pipeline in one place, reused by both /symptoms/check
    (live text) and /symptoms/history (replaying stored symptoms).
    """
    normalized = nlp_extraction.normalize(raw_text)

    # Stage: NLP & Symptom Extraction
    extracted = nlp_extraction.extract_symptoms(raw_text)

    # Stage: AI Symptom Analysis (against Medical Knowledge Base)
    matches = analyzer.analyze(extracted)

    # Stage: Risk Assessment
    risk = risk_assessment.assess_risk(matches, normalized)

    # Stage: Recommendation Engine
    recommendation = recommendation_engine.build_recommendation(risk["risk_level"], matches)

    return extracted, matches, risk, recommendation


def _to_match_out(matches) -> list:
    return [
        ConditionMatchOut(
            condition_id=m.condition_id,
            name=m.name,
            confidence=m.confidence,
            matched_symptoms=m.matched_symptoms,
            severity=m.severity,
            description=m.description,
            advice=m.advice,
        )
        for m in matches
    ]


@app.post("/symptoms/check", response_model=SymptomCheckResponse)
def check_symptoms(
    payload: SymptomCheckRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_user),  # optional (anonymous allowed)
):
    extracted, matches, risk, recommendation = _run_pipeline(payload.text)

    logger.info("symptom check processed: %s", safety_privacy.redact_for_logging(payload.text))

    # Persist the check (linked to user if authenticated, anonymous otherwise)
    record = SymptomCheck(
        user_id=current_user.id if current_user else None,
        raw_input=payload.text,
        extracted_symptoms=json.dumps(extracted),
        top_match_id=matches[0].condition_id if matches else None,
        top_match_confidence=matches[0].confidence if matches else None,
        is_emergency_flag=str(risk["risk_level"] == "urgent").lower(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return SymptomCheckResponse(
        id=record.id,
        extracted_symptoms=extracted,
        risk_level=risk["risk_level"],
        red_flags=risk["red_flags"],
        is_emergency=(risk["risk_level"] == "urgent"),
        matches=_to_match_out(matches),
        recommendation=recommendation["headline"],
    )


@app.get("/symptoms/history", response_model=list[SymptomCheckResponse])
def get_history(db: Session = Depends(get_db), current_user: User = Depends(auth.require_user)):
    records = (
        db.query(SymptomCheck)
        .filter(SymptomCheck.user_id == current_user.id)
        .order_by(SymptomCheck.created_at.desc())
        .limit(50)
        .all()
    )
    results = []
    for r in records:
        extracted = json.loads(r.extracted_symptoms) if r.extracted_symptoms else []
        matches = analyzer.analyze(extracted)
        risk = risk_assessment.assess_risk(matches, nlp_extraction.normalize(r.raw_input))
        recommendation = recommendation_engine.build_recommendation(risk["risk_level"], matches)

        results.append(
            SymptomCheckResponse(
                id=r.id,
                extracted_symptoms=extracted,
                risk_level=risk["risk_level"],
                red_flags=risk["red_flags"],
                is_emergency=(risk["risk_level"] == "urgent"),
                matches=_to_match_out(matches),
                recommendation=recommendation["headline"],
            )
        )
    return results


@app.get("/conditions")
def list_conditions():
    """Browse the full curated knowledge base."""
    return analyzer.kb


@app.get("/privacy-notice")
def privacy_notice():
    """Safety & Privacy module, surfaced as an endpoint the frontend can display."""
    return {
        "disclaimer": safety_privacy.get_disclaimer(),
        "consent_notice": safety_privacy.get_consent_notice(),
    }
