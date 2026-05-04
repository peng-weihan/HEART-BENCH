"""Shared HTTP headers for OpenAI-compatible chat/completions (urllib)."""

from __future__ import annotations

import os


def openai_compatible_request_headers(api_key: str) -> dict:
    """
    urllib has no default browser-like User-Agent; many gateways (Cloudflare, etc.)
    respond 403 to ``Python-urllib/3.x``. Override with env ``LLM_USER_AGENT``.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    ua = os.getenv("LLM_USER_AGENT", "").strip()
    if not ua:
        ua = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
    headers["User-Agent"] = ua
    return headers
