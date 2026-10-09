"""HTTP headers and explicit generation controls for the main MCQ runners."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


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


_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def mcq_reasoning_effort(value: str | None) -> str | None:
    """`omit` leaves the request field out; `none` is an explicit API value."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "omit":
        return None
    if normalized not in _REASONING_EFFORTS:
        raise argparse.ArgumentTypeError(
            "reasoning effort must be none, minimal, low, medium, high, xhigh, max, or omit"
        )
    return normalized


def mcq_temperature(value: str | float | None) -> float | None:
    """Parse a finite API temperature, or explicitly omit it for the endpoint default."""
    if value is None or (isinstance(value, str) and value.strip().lower() == "omit"):
        return None
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("temperature must be a number from 0 to 2, or omit") from exc
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise argparse.ArgumentTypeError("temperature must be finite and between 0 and 2")
    return temperature


def add_mcq_generation_arguments(parser: argparse.ArgumentParser) -> None:
    """Configuration for the four main MCQ settings, independent of construction/judges."""
    parser.add_argument(
        "--reasoning-effort", type=mcq_reasoning_effort,
        default=os.getenv("MCQ_REASONING_EFFORT", "").strip() or "high",
        help="API reasoning_effort (default: MCQ_REASONING_EFFORT or high); omit excludes the field",
    )
    parser.add_argument(
        "--temperature", type=mcq_temperature,
        default=os.getenv("MCQ_TEMPERATURE", "").strip() or "0",
        help="API temperature (default: MCQ_TEMPERATURE or 0); omit uses the endpoint default",
    )


def mcq_request_parameters(
    model: str, temperature: float | None = 0, reasoning_effort: str | None = "high",
) -> dict:
    """Return exactly the non-message fields serialized in the MCQ request body.

    These record submitted settings, not proof of a gateway's internal behavior.
    No provider-specific translation or unsupported-parameter fallback is applied.
    """
    parameters = {"model": model}
    effort = mcq_reasoning_effort(reasoning_effort)
    temperature = mcq_temperature(temperature)
    if effort is not None:
        parameters["reasoning_effort"] = effort
    if temperature is not None:
        parameters["temperature"] = temperature
    return parameters


def load_mcq_resume(
    path: str, parameters: dict, question_ids: set[str], run_context: dict,
) -> list[dict]:
    """Reuse successful responses only within one recorded condition and configuration.

    Legacy or mixed configurations must not be silently relabeled as the current run.
    Failed/invalid responses are retried, rather than duplicated in the output.
    """
    if not path:
        return []
    previous = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = previous.get("predictions") if isinstance(previous, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Resume file must contain a predictions list")
    if previous.get("run_context") != run_context:
        raise ValueError(
            "Cannot resume: missing or different run_context (setting/top_k). "
            "Use predictions from the same evaluation condition, or start a new run."
        )
    kept, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Resume predictions must be objects")
        qid = row.get("question_id")
        if qid not in question_ids or not row.get("ok") or row.get("predicted") is None:
            continue
        if row.get("request_parameters") != parameters:
            raise ValueError(
                f"Cannot resume {qid}: missing or different request_parameters. "
                "Use predictions with verified matching settings, or start a new run."
            )
        if qid in seen:
            raise ValueError(f"Resume file has duplicate successful responses for {qid}")
        seen.add(qid)
        kept.append(row)
    return kept
