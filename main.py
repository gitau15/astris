# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

# --- MOCK Councillors ---
async def the_empath(query: str) -> str:
    await asyncio.sleep(0.2)
    return "I sense feelings of burnout and uncertainty. Transitioning careers is a big step — your well-being matters most."

async def the_chronicler(query: str) -> str:
    await asyncio.sleep(0.2)
    return "As of late 2025, AI-powered design tools (like Figma's AI plugins) are reshaping UI/UX. Remote roles remain strong, but portfolios now must showcase AI collaboration fluency."

async def the_analyst(query: str) -> str:
    await asyncio.sleep(0.2)
    return "Here’s a 5-step plan: (1) Audit your transferable dev skills, (2) Build 2–3 UI case studies, (3) Learn Figma + AI design plugins, (4) Network with designers on LinkedIn, (5) Apply to hybrid dev/design roles first."

async def lead_councillor(empath_res: str, chronicler_res: str, analyst_res: str, user_query: str) -> str:
    await asyncio.sleep(0.1)
    return (
        f"Thank you for sharing your experience. {empath_res}\n\n"
        f"Regarding your interest in UI/UX: {chronicler_res}\n\n"
        f"To move forward: {analyst_res}\n\n"
        "You’re not alone in this shift — many developers are evolving into design-minded roles. Trust your instincts and build gradually."
    )

@app.post("/council/debate")
async def council_debate(request: QueryRequest):
    empath_res, chronicler_res, analyst_res = await asyncio.gather(
        the_empath(request.user_query),
        the_chronicler(request.user_query),
        the_analyst(request.user_query)
    )
    final_response = await lead_councillor(empath_res, chronicler_res, analyst_res, request.user_query)
    return {"response": final_response}