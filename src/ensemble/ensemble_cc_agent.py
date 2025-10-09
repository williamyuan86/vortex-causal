# src/ensemble/ensemble_cc_agent.py
import json
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# optional HF transformers call (if you have local model); otherwise fallback to rules
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

def prompt_for_edge(var_i, var_j, evidence_text):
    """
    Construct a JSON-constrained prompt for LLMs.
    """
    return f"""
You are a causal reasoning assistant. Given evidence, decide if variable "{var_i}" causes "{var_j}".
Evidence: {evidence_text}
Answer in JSON: {{ "edge": "{var_i}-{var_j}", "direction": "i->j" or "j->i" or "no_link", "confidence": 0.0-1.0, "comment": "brief reason" }}
"""

def llm_call(prompt, model_name=None):
    """
    If HF available and model_name provided, call local model. Otherwise fall back to simple heuristic parser.
    """
    if HF_AVAILABLE and model_name:
        pipe = pipeline("text-generation", model=model_name, device=0)
        out = pipe(prompt, max_new_tokens=128, do_sample=False)[0]['generated_text']
        # try to find JSON
        try:
            jstart = out.index('{')
            j = json.loads(out[jstart:])
            return j
        except Exception:
            return {"direction":"no_link","confidence":0.0,"comment":"parse_fail"}
    else:
        # fallback simple heuristic: look for keywords
        text = prompt.lower()
        if "cause" in text or "lead to" in text:
            return {"direction":"i->j", "confidence":0.6, "comment":"heuristic found causal keywords"}
        if "prevent" in text or "reduce" in text:
            return {"direction":"j->i", "confidence":0.55, "comment":"heuristic inverse keywords"}
        return {"direction":"no_link", "confidence":0.2, "comment":"no strong heuristic evidence"}

def ensemble_decide(edge_vars: Tuple[str,str], evidence_texts: List[str], model_names: List[str]=None):
    """
    edge_vars: (var_i, var_j)
    evidence_texts: list of strings (from retriever)
    model_names: list of hf model names or None to use fallback
    Returns aggregated decision and final confidence.
    """
    var_i, var_j = edge_vars
    prompt = prompt_for_edge(var_i, var_j, "\n".join(evidence_texts))
    results = []
    # call each model (or fallback) in model_names
    if model_names:
        for m in model_names:
            res = llm_call(prompt, model_name=m)
            results.append(res)
    else:
        # two fallback calls to mimic ensemble diversity
        results.append(llm_call(prompt, model_name=None))
        results.append(llm_call(prompt+" Another pass for stability.", model_name=None))
    # aggregate: majority direction with mean confidence
    dir_counts = {}
    for r in results:
        d = r.get("direction","no_link")
        dir_counts.setdefault(d, []).append(r.get("confidence",0.0))
    # choose the direction with highest average confidence
    best_dir = max(dir_counts.items(), key=lambda kv: (sum(kv[1])/len(kv[1]), len(kv[1])) )[0]
    avg_conf = sum(dir_counts[best_dir])/len(dir_counts[best_dir])
    return {"edge": f"{var_i}-{var_j}", "direction": best_dir, "confidence": avg_conf, "evidence_count": len(evidence_texts)}
