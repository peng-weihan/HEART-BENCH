"""
llm_client.py — shared LLM client for the experiments.
"""

import json
import re
import urllib.request
import urllib.error
from config import LLM_API_KEY, LLM_API_BASE, LLM_MODEL, LLM_TIMEOUT


def _sanitize_json(text: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


def call_llm(system_prompt: str, user_prompt: str, model: str = None) -> str:
    """Call an OpenAI-compatible chat endpoint and return the raw text."""
    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    payload = {
        "model": model or LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
        resp_text = resp.read().decode("utf-8", errors="replace")

    resp_json = json.loads(resp_text)
    content = (
        resp_json.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    # Strip markdown code block if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        content = "\n".join(lines)
    return content.strip() or "{}"


def call_llm_json(system_prompt: str, user_prompt: str, model: str = None) -> dict:
    """Call the LLM and parse the response as a JSON dict."""
    raw = call_llm(system_prompt, user_prompt, model)
    raw = _sanitize_json(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw, "parse_error": True}
