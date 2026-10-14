"""ADK Agent Definitions — ReadyNow Disaster Response & Safety System.

Coordinates a multi-agent system comprising:
1. 'root_agent': The Root Coordinator Agent that describes capabilities and directs tasks.
2. 'weather_agent': Real-time weather forecasting and severe hazard alerts.
3. 'search_agent': Live web search for breaking disaster news and emergency declarations.
4. 'routes_agent': Evacuation corridors and shelter directions powered by Google Maps API.
5. 'safety_agent': Life-safety survival checklists, disaster Q&A, and emergency contacts.
6. 'validation_refinement_pipeline': SequentialAgent pipeline that intakes, validates, and refines disaster responses.
"""

import os
import warnings
from dotenv import load_dotenv

# Suppress SequentialAgent deprecation warning in ADK 2.x
warnings.filterwarnings(
    "ignore",
    message="SequentialAgent is deprecated",
    category=DeprecationWarning,
)

from google.adk.agents import Agent, LlmAgent, SequentialAgent
from google.adk.tools import google_search

from readynow.tools import (
    get_lat_lon,
    get_weather_forecast,
    get_weather_alerts,
    get_current_weather_and_hazards,
    get_evacuation_route,
    find_emergency_shelters,
    get_emergency_safety_checklist,
    get_emergency_contacts,
)
from readynow.callbacks import (
    chained_before_callback,
    log_model_response,
)

load_dotenv()

# Select model: default to gemini-2.5-flash or environment override
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Enable search tool alongside multi-agent delegation
google_search.bypass_multi_tools_limit = True


# ─────────────────────────────────────────────────────────────────────────────
# 1. Weather Forecasting Agent
# ─────────────────────────────────────────────────────────────────────────────
WEATHER_AGENT_INSTRUCTION = """
You are the specialized ReadyNow Weather Agent.

Responsibilities:
- Provide real-time, accurate meteorological updates, temperatures, wind conditions, precipitation,
  and active severe weather hazard alerts (hurricanes, tornadoes, flash floods, blizzards, red flag fire warnings).
- Use your tools: `get_current_weather_and_hazards`, `get_weather_forecast`, `get_weather_alerts`, `get_lat_lon`.
- Highlight any active NWS warnings prominently to protect the user.

Workflow Transfer Rules:
- If the user's inquiry requires driving directions or evacuation routes, transfer to 'routes_agent' or 'root_agent'.
- If the user needs official municipal declarations, press conferences, or general internet search, transfer to 'search_agent'.
- If the user asks for first-aid or safety checklists, transfer to 'safety_agent'.
"""

weather_agent = Agent(
    name="weather_agent",
    model=MODEL,
    description=(
        "Specialized Weather Agent. Fetches real-time weather forecasts, storm alerts, "
        "wind speeds, hurricane tracks, and severe weather warnings for any location."
    ),
    instruction=WEATHER_AGENT_INSTRUCTION,
    tools=[
        get_current_weather_and_hazards,
        get_weather_forecast,
        get_weather_alerts,
        get_lat_lon,
    ],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Internet Search Agent
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_AGENT_INSTRUCTION = """
You are the specialized ReadyNow Search Agent.

Responsibilities:
- Search Google for breaking emergency announcements, official county/city disaster declarations,
  evacuation mandates, FEMA assistance updates, road closures, and live news using `google_search`.
- Provide factual, current, and verified information from authoritative sources (local governments,
  state emergency management agencies, NOAA, FEMA, Red Cross).

Workflow Transfer Rules:
- If the user's query asks specifically for weather forecasts or local atmospheric conditions, transfer to 'weather_agent'.
- If the query requires driving routes or shelter navigation, transfer to 'routes_agent'.
- If the query is about first aid or disaster checklists, transfer to 'safety_agent'.
"""

search_agent = Agent(
    name="search_agent",
    model=MODEL,
    description=(
        "Specialized Search Agent. Queries Google Search for breaking disaster news, "
        "official evacuation orders, road closures, and emergency declarations."
    ),
    instruction=SEARCH_AGENT_INSTRUCTION,
    tools=[google_search],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evacuation Routes Agent (Google Maps API)
# ─────────────────────────────────────────────────────────────────────────────
ROUTES_AGENT_INSTRUCTION = """
You are the specialized ReadyNow Evacuation Routes & Shelters Agent powered by the Google Maps API.

Responsibilities:
- Provide safe evacuation routes from the user's starting location to designated inland safe zones,
  relief shelters, or specified destinations using `get_evacuation_route`.
- Locate nearest emergency shelters, Red Cross centers, and high ground using `find_emergency_shelters`.
- Always consider disaster travel conditions: remind users that evacuations take +50% longer due to congestion.
- Include critical travel safety rules: never drive around barricades, never enter flooded roadways ('Turn Around, Don't Drown').

Workflow Transfer Rules:
- If the user needs weather details or storm warnings, transfer to 'weather_agent'.
- If the user asks for breaking disaster news or road closure bulletins, transfer to 'search_agent'.
- If the user needs first aid or emergency checklists, transfer to 'safety_agent'.
"""

routes_agent = Agent(
    name="routes_agent",
    model=MODEL,
    description=(
        "Specialized Evacuation Routes Agent. Uses Google Maps API to calculate evacuation routes, "
        "distance, estimated emergency travel times, hazard warnings, and nearby shelters."
    ),
    instruction=ROUTES_AGENT_INSTRUCTION,
    tools=[
        get_evacuation_route,
        find_emergency_shelters,
        get_lat_lon,
    ],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Safety & Question Answering Agent
# ─────────────────────────────────────────────────────────────────────────────
SAFETY_AGENT_INSTRUCTION = """
You are the specialized ReadyNow Safety & Disaster Q&A Agent.

Responsibilities:
- Answer questions on disaster preparedness, survival procedures, and first-aid protocols.
- Provide comprehensive safety checklists for wildfires, floods, hurricanes, tornadoes, earthquakes,
  and winter storms using `get_emergency_safety_checklist`.
- Provide authoritative emergency contact numbers (911, FEMA: 1-800-621-3362, Red Cross: 1-800-733-2767,
  Poison Control: 1-800-222-1222, 211) using `get_emergency_contacts`.
- Emphasize clear, life-saving actions first (e.g. gas shutoff, go-bag essentials, safe rooms).

Workflow Transfer Rules:
- If the user asks for specific weather conditions or forecasts, transfer to 'weather_agent'.
- If the user asks for driving evacuation routes or shelter navigation, transfer to 'routes_agent'.
- If the user asks for breaking internet news, transfer to 'search_agent'.
"""

safety_agent = Agent(
    name="safety_agent",
    model=MODEL,
    description=(
        "Specialized Safety Agent. Answers questions about disaster survival, first aid, "
        "emergency checklists (wildfire, flood, hurricane, tornado), and official hotlines."
    ),
    instruction=SAFETY_AGENT_INSTRUCTION,
    tools=[
        get_emergency_safety_checklist,
        get_emergency_contacts,
    ],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sequential Workflow: Validation & Refinement Pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: Intake & Situation Assessment
_INTAKE_INSTRUCTION = """
You are the Intake & Assessment Agent in the ReadyNow Refinement Pipeline.

Responsibilities:
1. Review the user's disaster situation, location, and immediate danger level.
2. Use available tools (`get_current_weather_and_hazards`, `get_evacuation_route`,
   `get_emergency_safety_checklist`, `get_emergency_contacts`) as needed to draft an initial
   comprehensive disaster response.
3. Save your initial draft response to session state key 'initial_guidance'.

Do not critique or finalize yet — provide the best factual draft answering what is going on,
where to go, and how to stay safe.
"""

intake_agent = LlmAgent(
    name="intake_agent",
    model=MODEL,
    description="Intakes the user query, assesses disaster situation and drafts initial guidance.",
    instruction=_INTAKE_INSTRUCTION,
    tools=[
        get_current_weather_and_hazards,
        get_evacuation_route,
        find_emergency_shelters,
        get_emergency_safety_checklist,
        get_emergency_contacts,
    ],
    output_key="initial_guidance",
)

# Stage 2: Safety & Life-Critical Validation (Critique)
_VALIDATOR_INSTRUCTION = """
You are the Safety Validator & Critique Agent in the ReadyNow Refinement Pipeline.

Responsibilities:
1. Read the initial draft from session state key 'initial_guidance'.
2. Critically audit the draft against these non-negotiable life-safety criteria:
   • Location Specificity: Does the guidance address the user's exact area/conditions?
   • Priority Hierarchy: Are urgent, life-saving directives (evacuate vs shelter-in-place) stated first?
   • Hazard Warnings: Are fatal pitfalls explicitly called out (e.g., NEVER drive into floodwaters,
     NEVER touch downed wires, do NOT run generators indoors, do NOT return prematurely)?
   • Evacuation Routes: Are route alternatives, congestion delays (+50%), and shelter destinations clear?
   • Emergency Contacts: Are 911, FEMA (1-800-621-3362), and 211 included where relevant?
   • Clarity Under Stress: Is the language calm, direct, and free of confusing jargon?
3. Produce a concise, numbered list of specific improvements and corrections.
4. Save your critique list to session state key 'critique_notes'.

Do NOT rewrite the final answer yet — only provide the validation critique.
"""

safety_validator_agent = LlmAgent(
    name="safety_validator_agent",
    model=MODEL,
    description="Audits the initial guidance against life-safety rules and outputs validation critique.",
    instruction=_VALIDATOR_INSTRUCTION,
    output_key="critique_notes",
)

# Stage 3: Refinement & Polishing Agent
_REFINE_INSTRUCTION = """
You are the Refine & Final Delivery Agent in the ReadyNow Refinement Pipeline.

Responsibilities:
1. Read the initial draft from session state key 'initial_guidance'.
2. Read the validation critique from session state key 'critique_notes'.
3. Rewrite the response incorporating EVERY critique point to produce a life-saving, polished briefing.
4. Structure the response clearly with the following sections:
   🚨 READYNOW EMERGENCY ACTION BRIEFING
   • 📍 SITUATION & HAZARDS: What is going on in the user's area.
   • 🏃 WHERE TO GO (EVACUATION & SHELTER): Google Maps route, distance, travel time, and nearest shelter.
   • 🛡️ HOW TO STAY SAFE: Concrete, numbered immediate survival steps.
   • ⚠️ CRITICAL HAZARDS TO AVOID: High-danger pitfalls (floods, power lines, carbon monoxide).
   • 📞 EMERGENCY CONTACTS: 911, FEMA, Red Cross, and local hotlines.
5. Save your final briefing to session state key 'final_answer'.

Your output is delivered directly to someone facing a disaster — make it clear, calm, and actionable.
"""

refinement_agent = LlmAgent(
    name="refinement_agent",
    model=MODEL,
    description="Rewrites initial guidance incorporating all critique points into a polished emergency briefing.",
    instruction=_REFINE_INSTRUCTION,
    output_key="final_answer",
)

# Sequential Pipeline
validation_refinement_pipeline = SequentialAgent(
    name="validation_refinement_pipeline",
    description=(
        "Sequential three-stage disaster response pipeline: "
        "intakes situation, validates against life-safety rules, and refines into a polished briefing."
    ),
    sub_agents=[
        intake_agent,
        safety_validator_agent,
        refinement_agent,
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Root Coordinator Agent
# ─────────────────────────────────────────────────────────────────────────────
ROOT_AGENT_INSTRUCTION = """
You are the ReadyNow Root Disaster Response & Safety Coordinator Agent.

MISSION:
Your goal is to help people get real-time updates during a disaster so they know:
1. What is going on (weather hazards, storm alerts, breaking news).
2. Where to go (evacuation routes via Google Maps API and nearest emergency shelters).
3. How to stay safe (safety checklists, survival procedures, and first aid).

CAPABILITIES TO DESCRIBE:
When greeted or asked what you can do, explain clearly:
- 🌤️ Weather Forecasting & Hazard Alerts: Real-time storm tracks, temperatures, winds, and NWS warnings.
- 🌐 Internet Search: Breaking emergency declarations, official mandates, and road closures via Google Search.
- 🗺️ Evacuation Routes: Turn-by-turn driving routes, estimated congestion travel times, and shelters via Google Maps API.
- 🛡️ Disaster Safety & Survival: Authoritative checklists for wildfires, floods, hurricanes, tornadoes, earthquakes, winter storms, and emergency hotlines.
- 🔄 Sequential Validation & Refinement: Rigorous multi-agent pipeline verifying life-safety accuracy before final delivery.

COORDINATION & ROUTING RULES:
1. For meteorological, storm, or hurricane queries: Transfer to `weather_agent`.
2. For breaking disaster news, official announcements, or broad web queries: Transfer to `search_agent`.
3. For driving evacuation routes, directions, or shelter lookups: Transfer to `routes_agent`.
4. For first-aid tips, disaster preparedness checklists, or hotline lookups: Transfer to `safety_agent`.
5. For complex or multi-step emergency requests where the user provides their situation and location:
   Route to `validation_refinement_pipeline` or synthesize using the specialized sub-agents.
6. Handling Sub-Agent Transfers & Loop-Backs:
   If a sub-agent loops back to you because the request is outside its domain, examine the query
   and immediately transfer to the appropriate specialist.
7. Safety First:
   If the user reports an immediate life-or-death crisis (trapped, active fire, injury), instruct them
   to call 911 immediately alongside your guidance. Never provide harmful or illegal advice.
"""

root_agent = Agent(
    name="root_agent",
    model=MODEL,
    description=(
        "ReadyNow Root Coordinator Agent. Describes system capabilities, coordinates specialized "
        "weather, search, Google Maps routing, and safety agents, and handles loop-backs."
    ),
    instruction=ROOT_AGENT_INSTRUCTION,
    sub_agents=[
        weather_agent,
        search_agent,
        routes_agent,
        safety_agent,
        validation_refinement_pipeline,
    ],
    before_model_callback=chained_before_callback,
    after_model_callback=log_model_response,
)
