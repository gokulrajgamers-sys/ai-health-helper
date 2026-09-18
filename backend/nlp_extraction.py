"""
NLP & Symptom Extraction Module
--------------------------------
Corresponds to the "NLP & SYMPTOM EXTRACTION" stage in the architecture
diagram: takes raw user text and extracts structured symptom mentions.

Includes basic context understanding via negation detection -- "no fever" or
"denies chest pain" should NOT be extracted as the symptom, since including it
would corrupt everything downstream (analysis, risk assessment).
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

KB_PATH = Path(__file__).parent / "knowledge_base.json"

NEGATION_WORDS = {"no", "not", "without", "never", "none", "denies", "denied", "isn't", "wasn't"}


def _load_vocab() -> list:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)["conditions"]
    # Longest phrases first so "chest pain" matches before bare "pain"
    return sorted({s.lower() for c in kb for s in c["symptoms"]}, key=len, reverse=True)


VOCAB = _load_vocab()


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _fuzzy_contains(haystack: str, needle: str, threshold: float = 0.82) -> bool:
    """Substring match, or fuzzy sliding-window match for typo tolerance."""
    if needle in haystack:
        return True
    words = haystack.split()
    n = len(needle.split())
    for i in range(len(words) - n + 1):
        window = " ".join(words[i : i + n])
        if SequenceMatcher(None, window, needle).ratio() >= threshold:
            return True
    return False


def _is_negated(normalized_text: str, phrase: str, window: int = 3) -> bool:
    """Context understanding: look a few words back from the phrase for negation cues."""
    idx = normalized_text.find(phrase)
    if idx == -1:
        return False
    preceding = normalized_text[:idx].split()[-window:]
    return any(w in NEGATION_WORDS for w in preceding)


def extract_symptoms(free_text: str) -> list:
    """Extract known symptom phrases from raw text, filtering out negated mentions."""
    normalized = normalize(free_text)
    found = []
    for phrase in VOCAB:
        if _fuzzy_contains(normalized, phrase) and not _is_negated(normalized, phrase):
            found.append(phrase)
    return found
