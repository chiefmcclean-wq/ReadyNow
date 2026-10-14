"""Lifecycle Callbacks and Interaction Logging for ReadyNow.

Implements Google ADK agent lifecycle hooks:
1. Persistent conversation and audit logging to 'readynow.log'.
2. Safety screening powered by Google Model Armor.
3. Emergency input validation and SOS priority triage.
4. Chained before_model_callback and after_model_callback for all ADK agents.
"""

from typing import Optional, Any
import logging
import os
from datetime import datetime

from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from readynow.model_armor import screen_prompt, BLOCKED_RESPONSE_TEXT
from readynow.validator import validate_emergency_input, TriagePriority

logger = logging.getLogger("readynow.callbacks")

LOG_FILE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "readynow.log")
)


class ConversationLogger:
    """Manages persistent logging of all user-agent interactions during disaster response."""

    @staticmethod
    def _append(entry: str) -> None:
        """Appends a log line to readynow.log."""
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception as e:
            logger.error("Failed to write to '%s': %s", LOG_FILE_PATH, e)

    @classmethod
    def log_user(cls, agent_name: str, session_id: str, message: str, priority: str = "STANDARD") -> None:
        """Logs user query with emergency priority tag."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [USER]    (Agent: {agent_name} | Session: {session_id} | Priority: {priority}) -> {message}"
        logger.info(entry)
        cls._append(entry)

    @classmethod
    def log_model(cls, agent_name: str, session_id: str, response: str) -> None:
        """Logs model output from an agent."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [MODEL]   (Agent: {agent_name} | Session: {session_id}) -> {response}"
        logger.info(entry)
        cls._append(entry)

    @classmethod
    def log_blocked(cls, agent_name: str, session_id: str, user_prompt: str, category: str, reason: str, blocked_response: str) -> None:
        """Logs security / Model Armor guardrail rejections."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"[{timestamp}] [BLOCKED] (Agent: {agent_name} | Session: {session_id})\n"
            f"           Category: {category} | Reason: {reason}\n"
            f"           INPUT:    {user_prompt}\n"
            f"           ACTION:   {blocked_response}"
        )
        logger.warning(entry)
        cls._append(entry)

    @classmethod
    def log_validation_error(cls, agent_name: str, session_id: str, error_msg: str, raw_input: str) -> None:
        """Logs input validation errors."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"[{timestamp}] [VALIDATION_ERROR] (Agent: {agent_name} | Session: {session_id})\n"
            f"           Error: {error_msg} | Input: {raw_input}"
        )
        logger.warning(entry)
        cls._append(entry)

    @classmethod
    def log_tool(cls, agent_name: str, session_id: str, tool_name: str, args: Any) -> None:
        """Logs tool invocation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [TOOL]    (Agent: {agent_name} | Session: {session_id}) Invoked '{tool_name}' with {args}"
        logger.info(entry)
        cls._append(entry)


def _extract_last_user_text(llm_request: LlmRequest) -> Optional[str]:
    """Helper to extract the most recent user turn text from an LlmRequest."""
    if not llm_request.contents:
        return None
    for content in reversed(llm_request.contents):
        if content.role == "user" and content.parts:
            text_parts = [p.text for p in content.parts if hasattr(p, "text") and p.text]
            if text_parts:
                return " ".join(text_parts).strip()
    return None


def _extract_model_text(llm_response: LlmResponse) -> Optional[str]:
    """Helper to extract text from an LlmResponse."""
    if llm_response.content and llm_response.content.parts:
        text_parts = [p.text for p in llm_response.content.parts if hasattr(p, "text") and p.text]
        if text_parts:
            return " ".join(text_parts).strip()
    return None


def chained_before_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """Pre-execution callback: validates input, applies Model Armor, and logs query.

    Returns an LlmResponse to short-circuit if blocked, or None to proceed.
    """
    user_text = _extract_last_user_text(llm_request)
    if not user_text:
        return None

    agent_name = getattr(callback_context, "agent_name", "readynow_agent")
    session_id = getattr(callback_context, "session_id", "session_unknown")

    # 1. Screen with Model Armor
    verdict = screen_prompt(user_text)
    if verdict.is_blocked:
        ConversationLogger.log_blocked(
            agent_name=agent_name,
            session_id=str(session_id),
            user_prompt=user_text,
            category=verdict.violation_category or "PolicyViolation",
            reason=verdict.reason or "Prohibited dangerous content",
            blocked_response=verdict.blocked_message,
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=verdict.blocked_message)],
            ),
            custom_metadata={
                "model_armor_blocked": True,
                "violation_category": verdict.violation_category,
                "reason": verdict.reason,
            },
        )

    # 2. Validate input and detect priority
    val_res = validate_emergency_input(user_text)
    if not val_res.is_valid:
        ConversationLogger.log_validation_error(
            agent_name=agent_name,
            session_id=str(session_id),
            error_msg=val_res.error_message or "Invalid input",
            raw_input=user_text,
        )
        msg = f"⚠️ Input validation error: {val_res.error_message}"
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=msg)],
            ),
            custom_metadata={"validation_failed": True},
        )

    # 3. Log approved query with priority classification
    ConversationLogger.log_user(
        agent_name=agent_name,
        session_id=str(session_id),
        message=user_text,
        priority=val_res.priority.value,
    )

    return None


def log_model_response(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """Post-execution callback: logs model response to file and audit log."""
    model_text = _extract_model_text(llm_response)
    if model_text:
        agent_name = getattr(callback_context, "agent_name", "readynow_agent")
        session_id = getattr(callback_context, "session_id", "session_unknown")
        ConversationLogger.log_model(
            agent_name=agent_name,
            session_id=str(session_id),
            response=model_text,
        )
    return None
