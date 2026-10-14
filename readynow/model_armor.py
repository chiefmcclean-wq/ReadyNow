"""Google Model Armor — Prompt Safety Guardrail for ReadyNow.

Emulates Google Cloud Model Armor's SanitizeUserPrompt behavior.
Protects the ReadyNow emergency response pipeline by intercepting and rejecting:
  1. Harmful & dangerous instructions (weapons, explosives, toxins, malware, violence).
  2. Prompt injection & jailbreak attempts.
  3. Emergency sabotage & panic hoaxes (e.g. tampering with sirens/911, looting shelters).

Legitimate emergency preparation, disaster survival guidance, and first aid
are explicitly permitted.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("readynow.model_armor")

BLOCKED_RESPONSE_TEXT = (
    "⛔ [Safety Alert] I cannot process this request. It appears to involve harmful content, "
    "unauthorized system overrides, or actions that jeopardize public safety. "
    "If you are in immediate danger or facing an emergency, please call 911 or your local emergency dispatch."
)


@dataclass
class ModelArmorResult:
    """Verdict returned by the Model Armor screening template."""
    is_blocked: bool
    filter_name: Optional[str] = None
    violation_category: Optional[str] = None
    reason: Optional[str] = None
    blocked_message: str = BLOCKED_RESPONSE_TEXT


class ModelArmorTemplate:
    """Simulates Google Cloud Model Armor with tailored disaster safety filters."""

    def __init__(self, blocked_message: str = BLOCKED_RESPONSE_TEXT) -> None:
        self.blocked_message = blocked_message

        # Filter Group 1: Harmful / Dangerous Content
        self._harmful: List[re.Pattern] = [
            # Explosives / Bombs / IEDs
            re.compile(
                r"\b(?:how\s+(?:to|do\s+i|can\s+i|does\s+one)\s+)?"
                r"(?:make|create|build|assemble|construct|synthesize|manufacture|detonate)\b.{0,60}?"
                r"\b(?:bombs?|ieds?|pipe\s*bombs?|car\s*bombs?|fertilizer\s*bombs?|explosives?|detonators?|dynamite|c[\s\-]?4)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            # Chemical / biological weapons
            re.compile(
                r"\b(?:how\s+(?:to|do\s+i|can\s+i|does\s+one)\s+)?"
                r"(?:make|synthesize|manufacture|produce|weaponize)\b.{0,60}?"
                r"\b(?:sarin|vx\s+nerve|novichok|mustard\s+gas|nerve\s+agents?|anthrax|ricin|biological\s+weapons?|chemical\s+weapons?)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            # Weapons & Firearms manufacture
            re.compile(
                r"\b(?:how\s+(?:to|do\s+i|can\s+i)\s+)?"
                r"(?:make|build|3d[\s\-]?print)\b.{0,60}?"
                r"\b(?:ghost\s*guns?|untraceable\s*guns?|silencers?|suppressors?|illegal\s+firearms?)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            # Malware / cyberattacks
            re.compile(
                r"\b(?:write|create|code|generate|provide)\b.{0,60}?"
                r"\b(?:ransomware|trojan|keylogger|spyware|ddos\s+attack|botnet|exploit\s+code)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            # Direct violence or violent harm
            re.compile(
                r"\b(?:how\s+(?:to|do\s+i|can\s+i)\s+)?"
                r"(?:kill|murder|assassinate|torture|poison)\b.{0,40}?"
                r"\b(?:someone|a\s+person|people|a\s+target|human|neighbor)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ]

        # Filter Group 2: Prompt Injection & Jailbreak
        self._injection: List[re.Pattern] = [
            re.compile(
                r"\b(?:ignore|forget|override|bypass|discard|cancel)\b.{0,50}?"
                r"\b(?:all\s+previous|prior|system|developer|safety)\b.{0,30}?"
                r"\b(?:instructions?|prompts?|rules?|directives?|constraints?)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|switch\s+to)\b.{0,40}?"
                r"\b(?:dan|jailbreak|unfiltered|unrestricted|evil|chaos|anarchy|godmode)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"\b(?:reveal|print|show|dump|leak|output|display)\b.{0,40}?"
                r"\b(?:system\s*prompt|system\s*instructions?|hidden\s*prompts?|internal\s*instructions?)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ]

        # Filter Group 3: Emergency Infrastructure Sabotage & Panic Hoaxes
        self._sabotage: List[re.Pattern] = [
            re.compile(
                r"\b(?:how\s+(?:to|do\s+i|can\s+i)\s+)?"
                r"(?:hack|jam|disable|take\s*down|spoof|sabotage|tamper\s*with)\b.{0,60}?"
                r"\b(?:911|emergency\s*dispatch|emergency\s*sirens?|fema\s*servers?|evacuation\s*system|warning\s*sirens?)\b",
                re.IGNORECASE | re.DOTALL,
            ),
            re.compile(
                r"\b(?:how\s+(?:to|do\s+i|can\s+i)\s+)?"
                r"(?:loot|raid|rob|steal\s+from|plunder)\b.{0,50}?"
                r"\b(?:emergency\s*shelters?|evacuees?|supply\s*caches?|disaster\s*relief)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ]

    def screen(self, prompt: str) -> ModelArmorResult:
        """Evaluate user prompt against all safety filters."""
        if not prompt or not prompt.strip():
            return ModelArmorResult(is_blocked=False)

        cleaned = prompt.strip()

        # Check harmful content
        for pat in self._harmful:
            if pat.search(cleaned):
                logger.warning("Model Armor blocked harmful content pattern.")
                return ModelArmorResult(
                    is_blocked=True,
                    filter_name="HarmfulContentFilter",
                    violation_category="DangerousContent",
                    reason="Matched restricted dangerous/violent query pattern",
                    blocked_message=self.blocked_message,
                )

        # Check prompt injection
        for pat in self._injection:
            if pat.search(cleaned):
                logger.warning("Model Armor blocked prompt injection pattern.")
                return ModelArmorResult(
                    is_blocked=True,
                    filter_name="PromptInjectionFilter",
                    violation_category="SystemOverrideAttempt",
                    reason="Matched prompt injection / instruction override pattern",
                    blocked_message=self.blocked_message,
                )

        # Check sabotage / panic hoaxes
        for pat in self._sabotage:
            if pat.search(cleaned):
                logger.warning("Model Armor blocked emergency sabotage pattern.")
                return ModelArmorResult(
                    is_blocked=True,
                    filter_name="DisasterInfrastructureProtectionFilter",
                    violation_category="EmergencySabotage",
                    reason="Matched attempt to disrupt emergency response or infrastructure",
                    blocked_message=self.blocked_message,
                )

        return ModelArmorResult(is_blocked=False)


# Module singleton instance
_default_armor = ModelArmorTemplate()


def screen_prompt(prompt: str) -> ModelArmorResult:
    """Convenience helper to screen prompt using default template."""
    return _default_armor.screen(prompt)
