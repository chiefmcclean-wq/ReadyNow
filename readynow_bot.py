"""ReadyNow Bot — Main Interactive CLI Application.

Coordinates a multi-agent disaster response system built with Google ADK:
1. 'root_agent': Coordinates tasks and explains capabilities.
2. 'weather_agent': Real-time forecasts and storm alerts.
3. 'search_agent': Live web search for breaking disaster news and official updates.
4. 'routes_agent': Evacuation routes and shelters via Google Maps API.
5. 'safety_agent': Survival checklists, disaster Q&A, and emergency hotlines.
6. 'validation_refinement_pipeline': Sequential pipeline validating and refining emergency briefings.

Environment variables (set in .env):
  GEMINI_API_KEY       — Required for full LLM multi-agent reasoning.
  GOOGLE_MAPS_API_KEY  — Optional. For live Google Maps API directions (fallback available).
  GEMINI_MODEL         — Optional. Default: 'gemini-2.5-flash'.
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

# Add current workspace to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from google.adk.runners import InMemoryRunner
from google.genai import types

from readynow.agents import root_agent
from readynow.model_armor import screen_prompt
from readynow.validator import validate_emergency_input, TriagePriority
from readynow.callbacks import ConversationLogger
from readynow.tools import (
    get_current_weather_and_hazards,
    get_evacuation_route,
    find_emergency_shelters,
    get_emergency_safety_checklist,
    get_emergency_contacts,
)

# ── ANSI Styling ─────────────────────────────────────────────────────────────
_ANSI = os.getenv("TERM", "").lower() not in ("", "dumb") or sys.platform == "win32"


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _ANSI else text


BOLD    = lambda t: _c("1", t)
DIM     = lambda t: _c("2", t)
RED     = lambda t: _c("31", t)
GREEN   = lambda t: _c("32", t)
YELLOW  = lambda t: _c("33", t)
BLUE    = lambda t: _c("34", t)
MAGENTA = lambda t: _c("35", t)
CYAN    = lambda t: _c("36", t)

AGENT_BADGES = {
    "root_agent":                     (MAGENTA, "🚨 Coordinator"),
    "weather_agent":                  (BLUE,    "🌤️ Weather"),
    "search_agent":                   (CYAN,    "🌐 Web Search"),
    "routes_agent":                   (GREEN,   "🗺️ Routes (Google Maps)"),
    "safety_agent":                   (YELLOW,  "🛡️ Safety & Survival"),
    "intake_agent":                   (CYAN,    "📥 Intake"),
    "safety_validator_agent":         (RED,     "🧐 Safety Validator"),
    "refinement_agent":               (GREEN,   "✨ Refine"),
    "validation_refinement_pipeline": (MAGENTA, "🔄 Sequential Pipeline"),
}

SESSION_COUNTER = 0


def print_banner() -> None:
    """Print the ReadyNow disaster response banner."""
    sep = "═" * 70
    print()
    print(BOLD(sep))
    print(BOLD("  🚨  ReadyNow  —  Real-Time Disaster Response & Evacuation Assistant"))
    print(BOLD("                 Powered by Google Agent Development Kit (ADK)"))
    print(BOLD(sep))
    print("  Active Capabilities & Specialized Agents:")
    print(f"    {MAGENTA('• Root Coordinator')}        — Assesses situation, explains features & routes tasks")
    print(f"    {BLUE('• Weather Specialist')}      — Real-time forecasts, storm tracks & NWS hazard alerts")
    print(f"    {CYAN('• Google Search')}           — Breaking emergency declarations & road closures")
    print(f"    {GREEN('• Google Maps Routing')}     — Evacuation corridors, traffic delays & safe shelters")
    print(f"    {YELLOW('• Safety & Survival')}       — Life-safety checklists (wildfire, flood, tornado, etc.)")
    print(f"    {RED('• Sequential Refinement')}   — 3-stage validation ensuring life-safety before delivery")
    print()
    print(f"  {DIM('Security: Google Model Armor safety guardrails + Input Triage active')}")
    print(f"  {DIM('Audit: All user-agent interactions logged to readynow.log')}")
    print(f"  {DIM('Type  exit  or  quit  to end the session.')}")
    print(BOLD(sep))
    print()


def _format_badge(name: str) -> str:
    if name in AGENT_BADGES:
        fn, label = AGENT_BADGES[name]
        return fn(f"[{label}]")
    return f"[{name}]"


async def run_pipeline(
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    user_query: str,
) -> None:
    """Streams the multi-agent execution pipeline for a user query."""
    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_query)],
    )

    current_author: str | None = None

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    ):
        # 1. Tool execution notifications
        func_calls = event.get_function_calls()
        if func_calls:
            for fc in func_calls:
                args_str = ", ".join(f"{k}='{v}'" for k, v in fc.args.items()) if fc.args else ""
                print(f"\n  {DIM(f'[Tool Invoked] {event.author} → {fc.name}({args_str})')}")
                ConversationLogger.log_tool(
                    agent_name=event.author or "agent",
                    session_id=session_id,
                    tool_name=fc.name,
                    args=fc.args,
                )

        # 2. Agent transfer notifications
        if event.actions and event.actions.transfer_to_agent:
            target = event.actions.transfer_to_agent
            print(f"\n  {DIM(f'[→ Transferring to {target}]')}")

        # 3. Stream model text response
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    author = event.author or "root_agent"
                    if current_author != author:
                        print()
                        print(_format_badge(author) + "  ", end="", flush=True)
                        current_author = author
                    print(part.text, end="", flush=True)

        # 4. Error reporting
        if event.error_code:
            print(f"\n  {RED(f'[ADK Error {event.error_code}] {event.error_message}')}")

    print("\n")


def run_direct_tool_demo() -> None:
    """Direct interactive fallback mode when GEMINI_API_KEY is not configured."""
    print(YELLOW("\n  ℹ️  No GEMINI_API_KEY found in .env."))
    print("      Starting in Direct Tool Interactive Demo Mode with Model Armor,")
    print("      emergency input validation, and full conversation logging enabled.\n")
    print("      Available quick commands:")
    print("        weather <city>           — Real-time weather & hazard alerts")
    print("        route <from> to <to>     — Evacuation route via Google Maps engine")
    print("        shelters <city>          — Nearby emergency shelters & safe havens")
    print("        checklist <disaster>     — Wildfire, flood, hurricane, tornado, winter storm")
    print("        contacts                 — Key emergency hotlines (911, FEMA, Red Cross)")
    print("        exit                     — Quit\n")

    session_id = "direct_demo_session"

    while True:
        try:
            raw = input(BOLD("❓ Emergency Inquiry: ")).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{GREEN('  👋  Stay safe! Goodbye.')}\n")
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "q", "bye"}:
            print(f"\n{GREEN('  👋  Stay safe! Goodbye.')}\n")
            break

        # 1. Screen with Model Armor
        armor = screen_prompt(raw)
        if armor.is_blocked:
            ConversationLogger.log_blocked(
                agent_name="direct_demo",
                session_id=session_id,
                user_prompt=raw,
                category=armor.violation_category or "Security",
                reason=armor.reason or "Blocked query",
                blocked_response=armor.blocked_message,
            )
            print(f"\n{RED(armor.blocked_message)}\n")
            continue

        # 2. Validate input and detect priority
        val = validate_emergency_input(raw)
        if not val.is_valid:
            print(f"\n{YELLOW(f'⚠️ {val.error_message}')}\n")
            continue

        if val.priority == TriagePriority.CRITICAL_SOS and val.immediate_sos_advice:
            print(f"\n{RED(BOLD(val.immediate_sos_advice))}\n")

        ConversationLogger.log_user("direct_demo", session_id, raw, val.priority.value)

        # 3. Direct dispatch
        lower = raw.lower()
        if lower.startswith("weather "):
            loc = raw[8:].strip()
            print(f"\n  {BLUE('[Fetching weather & hazards for ' + loc + '...]')}")
            res = get_current_weather_and_hazards(loc)
            if res.get("status") == "success":
                f = res.get("forecast", {})
                a = res.get("hazard_alerts", {})
                print(f"\n  📍 Location: {res.get('location')}")
                print(f"  🌡️ Temperature: {f.get('temperature', 'N/A')}")
                print(f"  💨 Wind: {f.get('wind', 'N/A')}")
                print(f"  📝 Conditions: {f.get('short_forecast', 'N/A')}")
                print(f"  ⚠️ Active Hazard Alerts: {a.get('alert_count', 0)}")
                for alert in a.get("alerts", []):
                    print(f"     • {alert.get('event')}: {alert.get('headline')}")
            else:
                print(f"  ❌ {res.get('message')}")

        elif " to " in lower and (lower.startswith("route ") or "evacuate" in lower):
            cleaned_cmd = raw.replace("route ", "")
            parts = cleaned_cmd.split(" to ")
            orig, dest = parts[0].strip(), parts[1].strip()
            print(f"\n  {GREEN('[Calculating Google Maps evacuation corridor from ' + orig + ' to ' + dest + '...]')}")
            route = get_evacuation_route(orig, dest)
            print(f"\n  🗺️ Route: {route.get('origin')} ➔ {route.get('destination')}")
            print(f"  📏 Distance: {route.get('distance')}")
            print(f"  ⏱️ Evacuation Est. Time: {route.get('evacuation_estimated_duration')}")
            print(f"  🛣️ Corridor: {route.get('summary_highway')}")
            print("  📋 Key Steps:")
            for s in route.get("key_steps", []):
                print(f"     • {s.get('instruction')} ({s.get('distance', '')})")
            print("  ⚠️ Advisories:")
            for adv in route.get("safety_route_advisories", []):
                print(f"     {adv}")

        elif lower.startswith("shelters "):
            loc = raw[9:].strip()
            print(f"\n  {GREEN('[Searching emergency shelters for ' + loc + '...]')}")
            sh = find_emergency_shelters(loc)
            print(f"\n  📍 Designated Shelters for {sh.get('location')}:")
            for s in sh.get("designated_shelters", []):
                print(f"     🏛️ {s.get('name')} ({s.get('type')}) - Status: {s.get('status')}")
                print(f"        Amenities: {s.get('amenities')} | Contact: {s.get('contact')}")
            print(f"  ℹ️ {sh.get('shelter_dispatch_note')}")

        elif lower.startswith("checklist") or "prepare" in lower:
            dtype = "general"
            for candidate in ["wildfire", "flood", "hurricane", "tornado", "earthquake", "winter_storm"]:
                if candidate in lower:
                    dtype = candidate
                    break
            res = get_emergency_safety_checklist(dtype)
            chk = res.get("checklist", {})
            print(f"\n  🛡️ {BOLD(chk.get('title', 'Emergency Checklist'))}")
            print("  🚨 Immediate Life-Safety Actions:")
            for act in chk.get("immediate_actions", []):
                print(f"     1. {act}")
            if chk.get("hazards_to_avoid"):
                print("  ⚠️ Critical Hazards To Avoid:")
                for haz in chk.get("hazards_to_avoid", []):
                    print(f"     • {haz}")

        elif "contact" in lower or "phone" in lower or "hotline" in lower:
            c = get_emergency_contacts()
            print(f"\n  📞 {BOLD('Emergency Numbers & Hotlines:')}")
            for item in c.get("contacts", []):
                print(f"     • {item['service']:<40} : {BOLD(item['number'])} ({item.get('notes', item.get('website', ''))})")

        else:
            print(f"\n  🤖 [ReadyNow]: To get real-time disaster guidance, please specify:")
            print(f"     - A location and situation (e.g. 'Hurricane approaching Tampa, where should I evacuate?')")
            print(f"     - Or use: weather <city>, route <from> to <to>, shelters <city>, checklist <disaster>, contacts")

        print()


async def main_async() -> None:
    """Main async interactive loop for ReadyNow."""
    global SESSION_COUNTER
    print_banner()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        run_direct_tool_demo()
        return

    runner = InMemoryRunner(agent=root_agent)
    user_id = "readynow_user"

    while True:
        SESSION_COUNTER += 1
        session_id = f"readynow_session_{SESSION_COUNTER:04d}"

        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )

        try:
            raw = input(BOLD("❓ Emergency Situation or Inquiry: ")).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{GREEN('  👋  Stay safe! Goodbye.')}\n")
            break

        if not raw:
            print(f"  {DIM('(Please enter your situation, location, or type exit to quit.)')}")
            continue

        if raw.lower() in {"exit", "quit", "q", "bye"}:
            print(f"\n{GREEN('  👋  Stay safe! Goodbye.')}\n")
            break

        # 1. Model Armor Safety Screen
        armor = screen_prompt(raw)
        if armor.is_blocked:
            ConversationLogger.log_blocked(
                agent_name="root_agent",
                session_id=session_id,
                user_prompt=raw,
                category=armor.violation_category or "PolicyViolation",
                reason=armor.reason or "Restricted query",
                blocked_response=armor.blocked_message,
            )
            print(f"\n{RED(armor.blocked_message)}\n")
            continue

        # 2. Input validation & emergency priority triage
        val = validate_emergency_input(raw)
        if not val.is_valid:
            print(f"\n{YELLOW(f'⚠️ {val.error_message}')}\n")
            continue

        if val.priority == TriagePriority.CRITICAL_SOS and val.immediate_sos_advice:
            print(f"\n{RED(BOLD(val.immediate_sos_advice))}\n")

        print()
        print(BOLD(f"  🚀  Coordinating response for session {session_id} …"))
        print(DIM("  " + "─" * 66))

        try:
            await run_pipeline(runner, user_id, session_id, raw)
        except Exception as exc:
            print(f"\n  {RED(f'[System error] {exc}')}\n")

        try:
            again = input(DIM("  Ask another question or request additional routes? [Y/n]: ")).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{GREEN('  👋  Stay safe! Goodbye.')}\n")
            break

        if again in {"n", "no", "exit", "quit", "q"}:
            print(f"\n{GREEN('  👋  Stay safe! Goodbye.')}\n")
            break

        print()


def main() -> None:
    """Synchronous entrypoint."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nSession terminated by user. Stay safe!")


if __name__ == "__main__":
    main()
