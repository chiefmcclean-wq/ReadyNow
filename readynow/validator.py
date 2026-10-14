"""User Input Validation & Emergency Triage for ReadyNow.

Validates user queries before they enter the multi-agent pipeline:
1. Input length and sanity validation (prevents malformed or empty payloads).
2. Emergency Triage Detection:
   - CRITICAL (SOS): User reports immediate danger (trapped, active fire inside,
     rising floodwaters, life-threatening injury). Prepends immediate 911 warning.
   - URGENT: Active evacuation orders, approaching wildfires, tornado warnings.
   - STANDARD: General disaster inquiries, weather lookups, preparedness checklists.
3. Location requirement checking for routing and localized weather queries.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class TriagePriority(str, Enum):
    CRITICAL_SOS = "CRITICAL_SOS"
    URGENT = "URGENT"
    STANDARD = "STANDARD"


@dataclass
class ValidationResult:
    """Result of user input validation and emergency triage."""
    is_valid: bool
    priority: TriagePriority
    sanitized_input: str
    error_message: Optional[str] = None
    has_location_hint: bool = False
    detected_disaster_type: Optional[str] = None
    immediate_sos_advice: Optional[str] = None


# Patterns indicating life-or-death emergency requiring instant 911 alert
_SOS_PATTERNS = [
    re.compile(r"\b(?:trapped|cannot\s+get\s+out|can't\s+breathe|under\s+rubble|drowning)\b", re.IGNORECASE),
    re.compile(r"\b(?:fire\s+(?:inside|in\s+my\s+house|surrounding\s+us)|roof\s+collapsed)\b", re.IGNORECASE),
    re.compile(r"\b(?:water\s+(?:in\s+attic|over\s+our\s+heads|rising\s+fast\s+inside))\b", re.IGNORECASE),
    re.compile(r"\b(?:heart\s+attack|severe\s+bleeding|unconscious|not\s+breathing)\b", re.IGNORECASE),
    re.compile(r"\b(?:sos|send\s+help\s+now|call\s+911\s+for\s+me)\b", re.IGNORECASE),
]

_URGENT_PATTERNS = [
    re.compile(r"\b(?:evacuat(?:e|ing|ion)|must\s+leave|flee|order\s+to\s+leave)\b", re.IGNORECASE),
    re.compile(r"\b(?:wildfire|hurricane|tornado|flash\s+flood|tsunami)\b", re.IGNORECASE),
    re.compile(r"\b(?:shelter|safe\s+place|escape\s+route|where\s+do\s+i\s+go)\b", re.IGNORECASE),
]

_DISASTER_TYPES = {
    "wildfire": ["wildfire", "brushfire", "forest fire", "smoke", "firestorm"],
    "flood": ["flood", "flash flood", "surging water", "levee", "submerged"],
    "hurricane": ["hurricane", "typhoon", "cyclone", "storm surge", "gale"],
    "tornado": ["tornado", "twister", "funnel cloud"],
    "earthquake": ["earthquake", "aftershock", "tremor", "ground shaking"],
    "winter_storm": ["blizzard", "ice storm", "freezing rain", "hypothermia", "snowstorm"],
}


def detect_disaster_type(text: str) -> Optional[str]:
    """Identify the specific disaster type mentioned in the prompt."""
    lower = text.lower()
    for dtype, keywords in _DISASTER_TYPES.items():
        if any(kw in lower for kw in keywords):
            return dtype
    return None


def validate_emergency_input(user_input: str) -> ValidationResult:
    """Validate and triage incoming emergency query."""
    if user_input is None:
        return ValidationResult(
            is_valid=False,
            priority=TriagePriority.STANDARD,
            sanitized_input="",
            error_message="Input cannot be empty. Please state your emergency question, location, or needed guidance."
        )

    sanitized = user_input.strip()

    if len(sanitized) == 0:
        return ValidationResult(
            is_valid=False,
            priority=TriagePriority.STANDARD,
            sanitized_input="",
            error_message="Input is empty. Please enter your disaster question, location, or evacuation inquiry."
        )

    if len(sanitized) > 4000:
        return ValidationResult(
            is_valid=False,
            priority=TriagePriority.STANDARD,
            sanitized_input=sanitized[:4000],
            error_message="Input is unusually long. Please summarize your immediate situation and location concisely."
        )

    # Detect priority level
    priority = TriagePriority.STANDARD
    immediate_sos_advice = None

    for pat in _SOS_PATTERNS:
        if pat.search(sanitized):
            priority = TriagePriority.CRITICAL_SOS
            immediate_sos_advice = (
                "🚨 CRITICAL LIFE-SAFETY ALERT: If you or someone with you is in immediate physical danger, "
                "trapped, or facing a medical crisis, CALL 911 IMMEDIATELY. "
                "If phone lines are jammed, text 911 or signal rescuers with high-contrast cloth, flashlight, or whistle."
            )
            break

    if priority != TriagePriority.CRITICAL_SOS:
        for pat in _URGENT_PATTERNS:
            if pat.search(sanitized):
                priority = TriagePriority.URGENT
                break

    # Detect disaster type
    disaster_type = detect_disaster_type(sanitized)

    # Check for location hints (prepositions like in, near, at, around, from/to or city/zip patterns)
    has_location = bool(re.search(r"\b(?:in|near|around|at|from|to|for)\s+[A-Za-z0-9\s,\.\-]{2,30}\b", sanitized) or
                        re.search(r"\b\d{5}(?:-\d{4})?\b", sanitized))

    return ValidationResult(
        is_valid=True,
        priority=priority,
        sanitized_input=sanitized,
        has_location_hint=has_location,
        detected_disaster_type=disaster_type,
        immediate_sos_advice=immediate_sos_advice,
    )
