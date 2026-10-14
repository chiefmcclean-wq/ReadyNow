"""Multi-Agent Workflow & Refinement Protocol for ReadyNow.

Defines:
1. System workflow metadata (hierarchical coordinator + sequential refinement pipeline).
2. Session state helpers for cross-agent data sharing.
3. Verification routines for response refinement and safety compliance.
"""

from typing import Any, Dict, List
import logging

logger = logging.getLogger("readynow.workflow")


def get_workflow_metadata() -> Dict[str, Any]:
    """Returns architecture metadata for ReadyNow."""
    return {
        "system_name": "ReadyNow Disaster Response & Safety Agent",
        "architecture": "Hierarchical Coordinator with Sequential Safety Validation & Refinement",
        "root_agent": "root_agent",
        "specialized_sub_agents": [
            "weather_agent",
            "search_agent",
            "routes_agent",
            "safety_agent",
        ],
        "refinement_pipeline": {
            "name": "validation_refinement_pipeline",
            "pattern": "SequentialAgent",
            "stages": [
                {"order": 1, "agent": "intake_agent", "output_key": "initial_guidance"},
                {"order": 2, "agent": "safety_validator_agent", "output_key": "critique_notes"},
                {"order": 3, "agent": "refinement_agent", "output_key": "final_answer"},
            ],
        },
        "integrations": [
            "National Weather Service (NWS) & Open-Meteo",
            "Google Search Grounding (google_search)",
            "Google Maps Directions & Geocoding API",
            "Google Model Armor Safety Guardrails",
            "Vertex AI Agent Engine / Agent Platform",
        ],
        "safety_guardrails": [
            "Model Armor dangerous content & prompt injection screening",
            "Emergency triage input validator (SOS priority detection)",
            "Mandatory 911 / emergency dispatch advice for life threats",
            "Persistent audit and conversation logging to 'readynow.log'",
        ],
    }


def update_session_state(tool_context: Any, key: str, value: Any) -> Dict[str, str]:
    """Appends or updates values in session state across agent transitions."""
    if hasattr(tool_context, "state") and tool_context.state is not None:
        tool_context.state[key] = value
        return {"status": "success", "key": key}
    return {"status": "error", "message": "No session state available in context"}
