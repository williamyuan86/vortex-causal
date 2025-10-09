# src/ensemble/ensemble_cc_agent_openai.py
import os
import json
import asyncio
import logging
import aiohttp
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ============ CONFIG ============ #
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", 60))
# ================================= #

def build_prompt(var_i, var_j, evidence_text):
    return f"""
You are a causal reasoning assistant.
Task: Decide whether variable "{var_i}" causes "{var_j}".
Evidence: {evidence_text}

Respond strictly in JSON format like:
{{"edge": "{var_i}-{var_j}", "direction": "i->j" or "j->i" or "no_link", "confidence": 0.0-1.0, "comment": "brief reason"}}
"""

async def call_openai_api(session, model, prompt):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 300
    }
    try:
        async with session.post(f"{OPENAI_BASE_URL}/chat/completions", json=data, headers=headers, timeout=TIMEOUT) as resp:
            rj = await resp.json()
            content = rj["choices"][0]["message"]["content"]
            try:
                jstart = content.index("{")
                return json.loads(content[jstart:])
            except Exception:
                return {"direction": "no_link", "confidence": 0.0, "comment": "json_parse_fail"}
    except Exception as e:
        logger.warning(f"{model} call failed: {e}")
        return {"direction": "no_link", "confidence": 0.0, "comment": "api_error"}

async def ensemble_decide_async(edge_vars: Tuple[str,str], evidence_texts: List[str], model_names: List[str]):
    var_i, var_j = edge_vars
    prompt = build_prompt(var_i, var_j, "\n".join(evidence_texts))
    async with aiohttp.ClientSession() as session:
        tasks = [call_openai_api(session, m, prompt) for m in model_names]
        results = await asyncio.gather(*tasks)
    return aggregate_results(var_i, var_j, results)

def aggregate_results(var_i, var_j, results):
    dir_conf = {}
    for r in results:
        d = r.get("direction", "no_link")
        dir_conf.setdefault(d, []).append(r.get("confidence", 0.0))
    best_dir = max(dir_conf.items(), key=lambda kv: (sum(kv[1])/len(kv[1]), len(kv[1])))[0]
    avg_conf = sum(dir_conf[best_dir])/len(dir_conf[best_dir])
    return {
        "edge": f"{var_i}-{var_j}",
        "direction": best_dir,
        "confidence": avg_conf,
        "votes": len(results),
    }

def ensemble_decide(edge_vars, evidence_texts, model_names=None):
    """
    Synchronous wrapper for convenience (used in main.py)
    """
    if not model_names:
        model_names = ["gpt-4o-mini"]
    return asyncio.run(ensemble_decide_async(edge_vars, evidence_texts, model_names))
