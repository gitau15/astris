# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
import asyncio
import logging

# Optional: enable logs in Vercel
logging.basicConfig(level=logging.INFO)

app = FastAPI()

class QueryRequest(BaseModel):
    user_query: str

# --- API Keys (from Vercel env) ---
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not HF_API_KEY or not TAVILY_API_KEY:
    raise RuntimeError("Missing HUGGINGFACE_API_KEY or TAVILY_API_KEY")

HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
TAVILY_HEADERS = {"Content-Type": "application/json"}

# --- Hugging Face Inference Helper ---
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
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(API_URL, headers=HF_HEADERS, json=payload)
            response.raise_for_status()
            result = response.json()
            # Handle HF's output format
            if isinstance(result, list) and len(result) > 0:
                generated = result[0].get("generated_text", "").strip()
                return generated if generated else "No response generated."
            else:
                return "Unexpected response format."
        except Exception as e:
            logging.error(f"HF call failed for {model}: {e}")
            return f"[Error: {model} unavailable]"

# --- Tavily Search Helper ---
async def tavily_search(query: str) -> str:
    API_URL = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": 5
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("answer", "No summary available.") + "\n\n" + "\n".join(
                [f"- {r['title']}: {r['url']}" for r in data.get("results", [])[:3]]
            )
        except Exception as e:
            logging.error(f"Tavily failed: {e}")
            return "[Error: Web search unavailable]"

# --- Councillors ---
async def the_empath(query: str) -> str:
    prompt = f"""You are The Empath, a wise and compassionate advisor who specializes in emotional intelligence. 
Your role is to:
- Identify the user’s emotional state (e.g., burnout, hope, confusion, fear)
- Acknowledge their feelings with warmth and validation
- Highlight psychological risks or opportunities
- Never give advice—only reflect and support

User query: "{query}"
Respond in 1–2 sentences. Be concise, kind, and human."""
    return await hf_infer("mistralai/Mistral-7B-Instruct-v0.3", prompt, 256)

async def the_strategist(query: str) -> str:
    prompt = f"""You are The Strategist, a pragmatic problem-solver who creates actionable plans. 
Break the user’s goal into 3–5 concrete, ordered steps.
Prioritize low-effort, high-impact actions first.
Consider time, resources, and beginner-friendliness.

User query: "{query}"

Respond as a numbered list. Be specific."""
    return await hf_infer("meta-llama/Meta-Llama-3-8B-Instruct", prompt, 384)

async def the_sentinel(query: str) -> str:
    prompt = f"""You are The Sentinel, an ethical advisor who identifies blind spots, biases, and long-term risks.
Analyze the user's situation for:
- Hidden assumptions
- Ethical dilemmas
- Potential regrets or unintended consequences

User query: "{query}"

Respond in 1–2 clear sentences."""
    return await hf_infer("cognitivecomputations/dolphin-mixtral-8x7b", prompt, 256)

async def the_chronicler(query: str) -> str:
    search_result = await tavily_search(query + " latest trends 2025")
    return f"Recent context: {search_result}"

async def the_lead(
    empath: str, chronicler: str, strategist: str, sentinel: str, query: str
) -> str:
    synthesis_prompt = f"""You are the Lead Councillor of Astris, a wise and integrative mind.
Synthesize these insights into one empathetic, coherent response:

- The Empath: "{empath}"
- The Chronicler: "{chronicler}"
- The Strategist: "{strategist}"
- The Sentinel: "{sentinel}"

User query: "{query}"

Write in a calm, mentor-like tone. Start with emotional validation, then weave in facts, plan, and ethical insight. End with encouragement. 4–6 sentences. No markdown."""
    
    # Try Llama-3-70B first, fallback to 8B if too slow/expensive
    result = await hf_infer("meta-llama/Meta-Llama-3-70B-Instruct", synthesis_prompt, 512)
    if "[Error" in result or "unavailable" in result:
        result = await hf_infer("meta-llama/Meta-Llama-3-8B-Instruct", synthesis_prompt, 512)
    return result

# --- Endpoint ---
@app.post("/council/debate")
async def council_debate(request: QueryRequest):
    if not request.user_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Run all specialists in parallel
        tasks = [
            the_empath(request.user_query),
            the_chronicler(request.user_query),
            the_strategist(request.user_query),
            the_sentinel(request.user_query)
        ]
        empath_res, chronicler_res, strategist_res, sentinel_res = await asyncio.gather(*tasks)

        final_response = await the_lead(
            empath_res, chronicler_res, strategist_res, sentinel_res, request.user_query
        )

        return {"response": final_response}

    except Exception as e:
        logging.error(f"Debate failed: {e}")
        raise HTTPException(status_code=500, detail="Council deliberation failed")