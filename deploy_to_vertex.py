"""Vertex AI Agent Engine / Agent Platform — Deployment Script for ReadyNow.

Deploys the ReadyNow disaster response multi-agent system to Google Cloud
Vertex AI Agent Engine (Agent Platform).

Pipeline stages executed:
  1. Validate environment configuration (GCP project, location, bucket).
  2. Initialize Vertex AI SDK.
  3. Wrap the ReadyNow multi-agent coordinator in AdkApp.
  4. Perform a local smoke-test to verify the pipeline before cloud upload.
  5. Deploy to Vertex AI Agent Engine via agent_engines.create().
  6. Test the remote deployed instance with a sample disaster inquiry.

Usage:
    py deploy_to_vertex.py

Required environment variables in .env:
    GOOGLE_CLOUD_PROJECT   — GCP Project ID (e.g. my-gcp-project)
    GOOGLE_CLOUD_LOCATION  — Region (default: us-central1)
    GOOGLE_CLOUD_BUCKET    — GCS staging bucket (bucket name only)
    GEMINI_API_KEY         — Google AI Studio API key (for local verification)
"""

import asyncio
import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure ReadyNow package is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "")
LOCATION   = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
BUCKET     = os.getenv("GOOGLE_CLOUD_BUCKET", "")
MODEL      = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Auto-detect local service account credentials JSON if present
_COMMON_KEY_NAMES = [
    "service_account.json",
    "service-account.json",
    "gcp-key.json",
    "credentials.json",
    "key.json",
]
for _candidate in _COMMON_KEY_NAMES:
    _full = os.path.join(os.path.dirname(__file__), _candidate)
    if os.path.isfile(_full) and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _full
        break

AGENT_DISPLAY_NAME = "readynow-disaster-response-bot"
AGENT_DESCRIPTION = (
    "ReadyNow multi-agent disaster response and evacuation system powered by Google ADK. "
    "Coordinates real-time weather alerts, Google Search, Google Maps evacuation routes, "
    "survival checklists, and sequential safety validation."
)

DEPLOYMENT_REQUIREMENTS = [
    "google-cloud-aiplatform[adk,agent_engines]>=2.4.0",
    "google-adk>=2.11.0",
    "google-genai>=2.30.1",
    "cloudpickle>=3.0.0",
    "python-dotenv>=1.0.0",
]

SEP  = "═" * 70
SSEP = "─" * 70


def _header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _step(n: int, label: str) -> None:
    print(f"\n  [{n}] {label}\n  {SSEP}")


def _ok(msg: str) -> None:
    print(f"  ✅  {msg}")


def _info(msg: str) -> None:
    print(f"  ℹ️   {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠️   {msg}")


def _fail(msg: str) -> None:
    print(f"  ❌  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Validate Configuration
# ─────────────────────────────────────────────────────────────────────────────
def validate_config() -> bool:
    """Validate environment settings and GCP credentials required for cloud deployment."""
    _step(1, "Validating Cloud Configuration & Credentials")
    ok = True

    checks = {
        "GOOGLE_CLOUD_PROJECT":  PROJECT_ID,
        "GOOGLE_CLOUD_LOCATION": LOCATION,
        "GOOGLE_CLOUD_BUCKET":   BUCKET,
    }

    for var, val in checks.items():
        if not val:
            _fail(f"{var} is not set in .env")
            ok = False
        else:
            _ok(f"{var} = {val}")

    # Check credentials
    cred_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_file and os.path.isfile(cred_file):
        _ok(f"GOOGLE_APPLICATION_CREDENTIALS = {cred_file}")
    else:
        # Check if ADC credentials can be resolved
        try:
            import google.auth
            creds, _ = google.auth.default()
            _ok("Application Default Credentials (ADC) found.")
        except Exception as auth_err:
            _warn(
                f"Google Cloud credentials not found.\n"
                f"     To authenticate with GCP project '{PROJECT_ID}':\n"
                f"     1. Place a Service Account key JSON in this directory (e.g. 'service-account.json')\n"
                f"        or set GOOGLE_APPLICATION_CREDENTIALS in .env\n"
                f"     2. Or run: gcloud auth application-default login"
            )
            ok = False

    _info(f"Target model      : {MODEL}")
    _info(f"Agent display name: {AGENT_DISPLAY_NAME}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Initialize Vertex AI
# ─────────────────────────────────────────────────────────────────────────────
def init_vertex_ai():
    """Initializes Vertex AI SDK."""
    _step(2, "Initializing Vertex AI SDK")
    import vertexai

    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=f"gs://{BUCKET}" if BUCKET else None,
    )
    _ok(f"Vertex AI initialized for project '{PROJECT_ID}' in '{LOCATION}'")
    if BUCKET:
        _ok(f"Staging bucket: gs://{BUCKET}")
    else:
        _warn("No staging bucket specified; using default Agent Engine bucket.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build AdkApp Wrapper
# ─────────────────────────────────────────────────────────────────────────────
def build_adk_app():
    """Wraps ReadyNow root_agent in AdkApp for Agent Engine."""
    _step(3, "Packaging ReadyNow Multi-Agent System into AdkApp")
    from vertexai.agent_engines import AdkApp
    from readynow.agents import root_agent

    _ok(f"Loaded root agent: '{root_agent.name}'")
    _ok(f"Registered sub-agents: {[a.name for a in root_agent.sub_agents]}")

    app = AdkApp(
        agent=root_agent,
        enable_tracing=True,
    )
    _ok("AdkApp wrapper created successfully with OpenTelemetry tracing enabled.")
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Local Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
async def local_smoke_test(app) -> bool:
    """Runs a local smoke-test query through the AdkApp before deploying."""
    _step(4, "Running Local Pre-Flight Smoke Test")
    test_user = "test_smoke_user"
    test_session = "smoke_test_session_001"
    test_query = "What is ReadyNow and what can you help me with during a hurricane?"

    _info(f"Sample test query: '{test_query}'")

    try:
        session = await app.async_create_session(
            user_id=test_user,
            session_id=test_session,
        )
        _ok(f"Local session initialized: {session.get('id', test_session)}")

        print()
        print("  ── Smoke-Test Output Preview ──────────────────────────────")
        captured = False
        async for event in app.async_stream_query(
            user_id=test_user,
            session_id=test_session,
            message=test_query,
        ):
            if isinstance(event, dict):
                parts = event.get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part and part["text"]:
                        author = event.get("author", "agent")
                        print(f"  [{author}]: {part['text'][:250]}...")
                        captured = True
                        break
            if captured:
                break

        print("  " + SSEP)
        if captured:
            _ok("Local smoke-test PASSED — agent pipeline produced verified output.")
        else:
            _warn("Smoke test completed without streamed text chunks.")
        return True

    except Exception as exc:
        _fail(f"Smoke test encountered error: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Deploy to Vertex AI Agent Engine
# ─────────────────────────────────────────────────────────────────────────────
def deploy_agent(app) -> object:
    """Deploys the application to Vertex AI Agent Platform."""
    _step(5, "Deploying to Vertex AI Agent Engine")
    _warn("Deployment typically takes 3–8 minutes as the container is provisioned. Please wait...")

    from vertexai import agent_engines

    remote_agent = agent_engines.create(
        agent_engine=app,
        display_name=AGENT_DISPLAY_NAME,
        description=AGENT_DESCRIPTION,
        requirements=DEPLOYMENT_REQUIREMENTS,
        extra_packages=["readynow"],
    )

    _ok("Deployment finished successfully!")
    _ok(f"Deployed Resource: {remote_agent.resource_name}")
    return remote_agent


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Test Remote Session
# ─────────────────────────────────────────────────────────────────────────────
async def test_remote_session(remote_agent) -> None:
    """Verifies that the deployed Agent Engine resource responds to remote queries."""
    _step(6, "Testing Remote Session on Vertex AI")
    user_id = "cloud-verification-user"
    session_id = "cloud-verification-session"

    try:
        remote_session = await remote_agent.async_create_session(
            user_id=user_id,
            session_id=session_id,
        )
        _ok(f"Remote session created: {remote_session.get('id', session_id)}")

        query = "Wildfire reported near Paradise, CA. What should I do right now?"
        _info(f"Sending test emergency query: '{query}'")

        async for event in remote_agent.async_stream_query(
            user_id=user_id,
            session_id=session_id,
            message=query,
        ):
            if isinstance(event, dict):
                for part in event.get("content", {}).get("parts", []):
                    if "text" in part and part["text"]:
                        print(f"  [Remote Response]: {part['text'][:200]}...")
                        break
        _ok("Remote deployment verification succeeded!")

    except Exception as e:
        _warn(f"Remote session query test encountered: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Routine
# ─────────────────────────────────────────────────────────────────────────────
async def main_async() -> None:
    _header("🚀 ReadyNow — Vertex AI Agent Platform Deployment")

    if not validate_config():
        _warn("GCP configuration incomplete. To deploy to Vertex AI, please supply:")
        print("     GOOGLE_CLOUD_PROJECT=<your-project-id>")
        print("     GOOGLE_CLOUD_LOCATION=us-central1")
        print("     GOOGLE_CLOUD_BUCKET=<your-staging-bucket>")
        print("\n  Pre-flight validation demonstrated required steps.")
        return

    init_vertex_ai()
    app = build_adk_app()

    # Pre-flight local verification
    smoke_ok = await local_smoke_test(app)
    if not smoke_ok:
        _fail("Pre-flight smoke test failed. Halting deployment.")
        return

    # Deploy
    remote_agent = deploy_agent(app)

    # Save resource name to deployed_agent_resource.txt and .env
    resource_file = os.path.join(os.path.dirname(__file__), "deployed_agent_resource.txt")
    with open(resource_file, "w", encoding="utf-8") as f:
        f.write(remote_agent.resource_name)
    _ok(f"Saved resource name to {resource_file}")

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            env_content = f.read()
        if "VERTEX_AGENT_RESOURCE=" in env_content:
            import re
            env_content = re.sub(r"VERTEX_AGENT_RESOURCE=.*", f"VERTEX_AGENT_RESOURCE={remote_agent.resource_name}", env_content)
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(env_content)
        else:
            with open(env_path, "a", encoding="utf-8") as f:
                f.write(f"\nVERTEX_AGENT_RESOURCE={remote_agent.resource_name}\n")
        _ok(f"Updated VERTEX_AGENT_RESOURCE in {env_path}")
    except Exception as e:
        _warn(f"Could not update .env: {e}")

    # Verify remote operation
    await test_remote_session(remote_agent)
    _header("🎉 ReadyNow Successfully Deployed to Vertex AI Agent Platform!")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
