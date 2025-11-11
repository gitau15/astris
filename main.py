# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

# --- Load all keys ---
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GLM_API_KEY = os.getenv("GLM_API_KEY")

if not HF_API_KEY or not TAVILY_API_KEY or not GLM_API_KEY:
    raise RuntimeError("Missing one or more API keys")

HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
GLM_HEADERS = {
    "Authorization": f"Bearer {GLM_API_KEY}",
    "Content-Type": "application/json"
}

# --- Hugging Face Inference (for specialists) ---
async def hf_infer(model: str, prompt: str, max_tokens: int = 384) -> str:
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.post(API_URL, headers=HF_HEADERS, json=payload)
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "").strip() or "[No response]"
            return "[HF format error]"
        except Exception as e:
            logging.error(f"HF error: {e}")
            return "[Specialist unavailable]"

# --- Tavily ---
async def tavily_search(query: str) -> str:
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query + " 2025 trends",
                    "search_depth": "advanced",
                    "include_answer": True,
                    "max_results": 3
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("answer", "No recent context found.")
        except Exception as e:
            logging.error(f"Tavily error: {e}")
            return "[Web context unavailable]"

# --- GLM-4 Inference (Lead Synthesizer) ---
async def glm_synthesize(prompt: str) -> str:
    API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    payload = {
        "model": "glm-4-0520",  # or "glm-4" — 0520 is often cheapest
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 600
    }
    async with httpx.AsyncClient(timeout=25.0) as client:
        try:
            resp = await client.post(API_URL, headers=GLM_HEADERS, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logging.error(f"GLM error: {e}")
            # Fallback to HF Mixtral if GLM fails
            fallback = await hf_infer("mistralai/Mixtral-8x7B-Instruct-v0.1", prompt, 600)
            return fallback

# --- Councillors ---
async def the_empath(q): 
    p = f"You are The Empath. Respond to: '{q}' with emotional validation in 1–2 sentences."
    return await hf_infer("mistralai/Mistral-7B-Instruct-v0.3", p, 200)

async def the_strategist(q):
    p = f"You are The Strategist. Give 3–5 actionable steps for: '{q}'. Numbered list only."
    return await hf_infer("HuggingFaceH4/zephyr-7b-beta", p, 300)

async def the_sentinel(q):
    p = f"You are The Sentinel. Identify risks or ethical concerns in: '{q}'. 1–2 sentences."
    return await hf_infer("cognitivecomputations/dolphin-mixtral-8x7b", p, 200)

async def the_chronicler(q):
    res = await tavily_search(q)
    return f"Latest context: {res}"

async def the_lead(e, c, s, t, q):
    prompt = f"""You are the Lead Councillor of Astris. Synthesize:

Empath: "{e}"
Chronicler: "{c}"
Strategist: "{s}"
Sentinel: "{t}"

User query: "{q}"

Write a wise, cohesive 4–6 sentence response: validate emotions, give facts, offer plan, note caution, end with hope."""
    return await glm_synthesize(prompt)

# --- Endpoint ---
@app.post("/council/debate")
async def council_debate(request: QueryRequest):
    if not request.user_query.strip():
        raise HTTPException(400, "Query cannot be empty")
    
    try:
        e, c, s, t = await asyncio.gather(
            the_empath(request.user_query),
            the_chronicler(request.user_query),
            the_strategist(request.user_query),
            the_sentinel(request.user_query)
        )
        final = await the_lead(e, c, s, t, request.user_query)
        return {"response": final}
    except Exception as e:
        logging.error(f"Debate failed: {e}")
        raise HTTPException(500, "Council unavailable")