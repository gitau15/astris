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

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not HF_API_KEY or not TAVILY_API_KEY:
    raise RuntimeError("Missing HUGGINGFACE_API_KEY or TAVILY_API_KEY")

HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

async def hf_infer(model: str, prompt: str, max_tokens: int = 512) -> str:
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "return_full_text": False,
            "do_sample": True,
        }
    }
    async with httpx.AsyncClient(timeout=25.0) as client:
        try:
            response = await client.post(API_URL, headers=HF_HEADERS, json=payload)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated = result[0].get("generated_text", "").strip()
                return generated if generated else "No response."
            else:
                return "Unexpected response format."
        except Exception as e:
            logging.error(f"HF failed for {model}: {e}")
            return f"[Model unavailable: {model.split('/')[-1]}]"

async def tavily_search(query: str) -> str:
    API_URL = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": 3
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "No summary.")
            return f"{answer}"
        except Exception as e:
            logging.error(f"Tavily failed: {e}")
            return "[Web search unavailable]"

# --- Councillors with reliable models ---
async def the_empath(query: str) -> str:
    prompt = f"""You are The Empath. Identify emotional state (burnout, hope, anxiety). Validate feelings. Be warm, concise.

User: "{query}"
Response (1–2 sentences):"""
    return await hf_infer("mistralai/Mistral-7B-Instruct-v0.3", prompt, 200)

async def the_strategist(query: str) -> str:
    prompt = f"""You are The Strategist. Create 3–5 actionable steps for: "{query}". Be specific and practical. Numbered list only."""
    return await hf_infer("HuggingFaceH4/zephyr-7b-beta", prompt, 300)

async def the_sentinel(query: str) -> str:
    prompt = f"""You are The Sentinel. Identify ethical risks, biases, or long-term regrets in: "{query}". 1–2 sentences."""
    return await hf_infer("cognitivecomputations/dolphin-mixtral-8x7b", prompt, 200)

async def the_chronicler(query: str) -> str:
    result = await tavily_search(query + " 2025 trends")
    return f"Latest context: {result}"

async def the_lead(empath, chronicler, strategist, sentinel, query):
    prompt = f"""You are the Lead Councillor. Synthesize:

Empath: "{empath}"
Chronicler: "{chronicler}"
Strategist: "{strategist}"
Sentinel: "{sentinel}"

User: "{query}"

Write 4–6 sentences: start with empathy, add facts, give plan, note caution, end with hope. Mentor tone."""
    # Use Mixtral-8x7B — more reliable than Llama-3 on HF free tier
    return await hf_infer("mistralai/Mixtral-8x7B-Instruct-v0.1", prompt, 500)

@app.post("/council/debate")
async def council_debate(request: QueryRequest):
    if not request.user_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        tasks = [
            the_empath(request.user_query),
            the_chronicler(request.user_query),
            the_strategist(request.user_query),
            the_sentinel(request.user_query)
        ]
        e, c, s, t = await asyncio.gather(*tasks)
        final = await the_lead(e, c, s, t, request.user_query)
        return {"response": final}
    except Exception as e:
        logging.error(f"Debate failed: {e}")
        raise HTTPException(status_code=500, detail="Council unavailable")