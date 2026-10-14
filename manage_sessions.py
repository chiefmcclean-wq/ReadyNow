"""Vertex AI Agent Engine / Agent Platform — Session & Request Manager for ReadyNow.

Provides CLI commands to interact with an already-deployed ReadyNow Agent
on Vertex AI Agent Engine:
  query       — Send an emergency inquiry and stream the multi-agent response.
  sessions    — List all remote sessions for a user.
  get-session — Inspect state for a specific session.
  delete      — Delete a remote session.

Usage examples:
    py manage_sessions.py query "Hurricane approaching Tampa, FL. What evacuation route should I take?"
    py manage_sessions.py sessions
    py manage_sessions.py get-session <session_id>
    py manage_sessions.py delete <session_id>
"""

import os
import sys
import asyncio
import warnings

# Suppress FutureWarnings from google-api-core
warnings.filterwarnings("ignore", category=FutureWarning)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

PROJECT_ID   = os.getenv("GOOGLE_CLOUD_PROJECT", "")
LOCATION     = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
DEFAULT_USER = os.getenv("VERTEX_USER_ID", "readynow_user")

_RESOURCE_FILE = os.path.join(os.path.dirname(__file__), "deployed_agent_resource.txt")


def get_resource_name() -> str:
    """Retrieve deployed agent resource identifier."""
    name = os.getenv("VERTEX_AGENT_RESOURCE", "")
    if not name and os.path.isfile(_RESOURCE_FILE):
        with open(_RESOURCE_FILE, "r", encoding="utf-8") as f:
            name = f.read().strip()
    return name


SEP = "═" * 70


def _header(t: str) -> None:
    print(f"\n{SEP}\n  {t}\n{SEP}")


def _ok(m: str) -> None:
    print(f"  ✅  {m}")


def _info(m: str) -> None:
    print(f"  ℹ️   {m}")


def _fail(m: str) -> None:
    print(f"  ❌  {m}")


def load_remote_agent():
    """Initializes Vertex AI SDK and loads remote deployed agent engine."""
    import vertexai
    from vertexai import agent_engines

    resource = get_resource_name()
    if not resource:
        _fail(
            "No deployed Vertex AI agent resource name found.\n"
            "  • Set VERTEX_AGENT_RESOURCE in .env, or\n"
            "  • Run 'py deploy_to_vertex.py' to deploy the agent first."
        )
        sys.exit(1)

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    _ok(f"Vertex AI initialized — project={PROJECT_ID}, location={LOCATION}")

    agent = agent_engines.get(resource)
    _ok(f"Connected to remote agent: {resource}")
    return agent


async def cmd_query(agent, query_text: str, user_id: str = DEFAULT_USER) -> None:
    """Creates a fresh remote session and streams the multi-agent response."""
    import uuid
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    _header(f"Querying ReadyNow on Vertex AI Agent Engine ({session_id})")
    _info(f"User: {user_id} | Emergency Inquiry: {query_text}")

    session = await agent.async_create_session(user_id=user_id, session_id=session_id)
    _ok(f"Remote session started: {session.get('id', session_id)}")

    print("\n  ── ReadyNow Disaster Briefing ─────────────────────────────────")
    async for event in agent.async_stream_query(
        user_id=user_id,
        session_id=session_id,
        message=query_text,
    ):
        if isinstance(event, dict):
            parts = event.get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part and part["text"]:
                    print(part["text"], end="", flush=True)
    print("\n  " + "─" * 65 + "\n")


async def cmd_list_sessions(agent, user_id: str = DEFAULT_USER) -> None:
    """Lists remote sessions."""
    _header(f"Listing Sessions for User '{user_id}'")
    sessions = await agent.async_list_sessions(user_id=user_id)
    if not sessions:
        _info("No active sessions found.")
        return
    for s in sessions:
        if isinstance(s, dict):
            print(f"  • Session ID: {s.get('id')} (Created: {s.get('create_time', 'N/A')})")
        else:
            print(f"  • Session ID: {s}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("ReadyNow Vertex AI Session Manager")
        print("Usage:")
        print("  py manage_sessions.py query \"<emergency question>\"")
        print("  py manage_sessions.py sessions")
        sys.exit(0)

    cmd = args[0].lower()
    agent = load_remote_agent()

    if cmd == "query":
        query_text = args[1] if len(args) > 1 else "What is the hurricane evacuation procedure?"
        asyncio.run(cmd_query(agent, query_text))
    elif cmd == "sessions":
        asyncio.run(cmd_list_sessions(agent))
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
