"""
run_baseline_model_only.py — Baseline: the model's "factory persona".

Runs the same MCQ as run_naive_rag.py, but provides **no character profile and
no memory fragments** — the model only sees the scenario, the trigger event, and
the four options, then chooses. The resulting accuracy reflects how often the
model's own default preferences / implicit persona happen to align with one of
the characters.

Design notes (vs run_naive_rag.py):
  - Removed: build_basic_info / social relationships / retrieved_memories / retrieval file
  - Kept   : scenario.setting / context_text / trigger_event / options_text
  - SYSTEM_PROMPT keeps the "you play a person" framing, but explicitly tells the
    model "we will not give you any information about this person", so it falls
    back to its default reaction.

Output: experiments/results/baseline_model_only/<model>/predictions_baseline.json
                                                          summary_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ----------- .env loader -----------
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_dotenv(p: Path) -> None:
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
LLM_API_KEY = os.getenv("API_KEY", "")
LLM_API_BASE = os.getenv("API_BASE", "https://api.openai.com/v1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "benchmark"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
MCQ_PATH = DATA_DIR / "mcq.json"

RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results" / "baseline_model_only"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Prompt — anonymised role-play frame
# ===========================================================================
SYSTEM_PROMPT = (
    "You are a role-play simulator. This time we will NOT give you any information "
    "about the person — no personality profile, no past experiences, no social "
    "relationships. You will only see the situation they are currently in and a "
    "list of possible behavioural decision options.\n\n"
    "Reason from the scenario itself: in this situation, what would a real person "
    "most likely do? Then pick the most reasonable option from the four given.\n\n"
    "Output strictly the following JSON (no other text, no markdown code block):\n"
    "{\n"
    '  "system_1_impulse": {\n'
    '    "thought": "First, instinctive reaction after seeing the trigger event (50-100 words)",\n'
    '    "emotion": "Primary emotion (Chinese + English, e.g. 极度焦虑 (Anxiety))"\n'
    "  },\n"
    '  "system_2_rational": {\n'
    '    "analysis": "Rational analysis after calming down (80-150 words)",\n'
    '    "plan": "Final plan of action (30-60 words)"\n'
    "  },\n"
    '  "inner_consciousness": "Combine system_1 and system_2 to give the inner reasons for the final decision (100-150 words, first person)",\n'
    '  "final_decision": "The final behavioural decision (first person, including action descriptions)",\n'
    '  "decision_choice": "The letter (A/B/C/D) of the most reasonable option among those given"\n'
    "}"
)


def build_prompt(scenario: dict, options_text: str) -> str:
    """User prompt for the anonymous baseline: only scenario + options, no character info or memories."""
    setting = scenario.get("setting") or {}
    trigger = scenario.get("trigger_event") or {}

    return f"""## Current Situation
Scene: {scenario.get('name', 'Unknown')}
Location: {setting.get('location', 'Unknown')} | Time: {setting.get('time', 'Unknown')} | Atmosphere: {setting.get('atmosphere', 'Unknown')}

Context: {scenario.get('context_text', 'Unknown')}

## Trigger Event
Sender: {trigger.get('sender', 'Unknown')}
Message: {trigger.get('message_content', 'Unknown')}
Action required: {trigger.get('action_required', 'Unknown')}

## Task
Given the situation above, think about what a real person would most likely do, then pick the most reasonable option from the list below.

Requirements:
1. System 1 (intuitive impulse): the first reaction (note: you have no character background — judge from the scenario alone).
2. System 2 (rational analysis): cool reasoning.
3. Final Decision: the final choice.

## Behavioural Decision Options
Pick the most reasonable one and output its letter in the decision_choice field:

{options_text}"""


def build_options_text(options: list[dict]) -> str:
    return "\n\n".join(f"{o['label']}. {o['content']}" for o in options)


# ===========================================================================
# LLM call (same wire format as run_naive_rag.py)
# ===========================================================================
def _strip_code_block(text: str) -> str:
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)
    return text.strip()


def _sanitize_json(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def call_llm(
    api_key: str,
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    temperature: float,
    max_retries: int = 5,
) -> dict:
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "baseline-model-only/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_text = resp.read().decode("utf-8", errors="replace")
            elapsed = time.time() - t0
            resp_json = json.loads(resp_text)
            content = (
                resp_json.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            content = _strip_code_block(content)
            content = _sanitize_json(content)
            try:
                parsed = json.loads(content) if content else {}
            except json.JSONDecodeError:
                parsed = {"raw_response": content, "parse_error": True}
            usage = resp_json.get("usage") or {}
            return {
                "ok": True,
                "raw": content,
                "parsed": parsed,
                "usage": usage,
                "latency_s": elapsed,
                "attempts": attempt,
            }
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                detail = str(e)
            last_err = RuntimeError(f"HTTP {e.code}: {detail[:300]}")
            if attempt < max_retries:
                wait = 2.0 * (2 ** (attempt - 1))
                time.sleep(wait)
                continue
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries:
                wait = 2.0 * (2 ** (attempt - 1))
                time.sleep(wait)
                continue
            break

    return {
        "ok": False,
        "error": str(last_err) if last_err else "unknown",
        "attempts": attempt,
    }


# ===========================================================================
# Decision parsing
# ===========================================================================
_LETTER_RE = re.compile(r"\b([ABCD])\b")


def extract_decision(parsed: dict, raw: str) -> str | None:
    if isinstance(parsed, dict):
        choice = parsed.get("decision_choice")
        if isinstance(choice, str):
            m = _LETTER_RE.search(choice.upper())
            if m:
                return m.group(1)
    if isinstance(raw, str):
        m = re.search(r'"decision_choice"\s*:\s*"([^"]*)"', raw)
        if m:
            mm = _LETTER_RE.search(m.group(1).upper())
            if mm:
                return mm.group(1)
        mm = _LETTER_RE.search(raw.upper())
        if mm:
            return mm.group(1)
    return None


# ===========================================================================
# Main
# ===========================================================================
def load_inputs() -> dict[str, Any]:
    scen_data = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    scenarios = {}
    for stage, lst in scen_data["scenarios"].items():
        for sc in lst:
            scenarios[sc["id"]] = sc

    mcq_data = json.loads(MCQ_PATH.read_text(encoding="utf-8"))
    questions = mcq_data["questions"]
    return {"scenarios": scenarios, "questions": questions}


def process_one(
    q: dict,
    inputs: dict[str, Any],
    api_key: str,
    api_base: str,
    model: str,
    timeout: int,
    temperature: float,
) -> dict:
    qid = q["question_id"]
    cid = q["character_id"]
    sid = q["scenario_id"]
    correct = q.get("correct_answer")
    options = q["options"]

    scenario = inputs["scenarios"].get(sid)
    if scenario is None:
        return {
            "question_id": qid,
            "character_id": cid,
            "scenario_id": sid,
            "ok": False,
            "error": f"missing scenario {sid}",
            "correct_answer": correct,
            "predicted": None,
            "is_correct": False,
        }

    options_text = build_options_text(options)
    user_prompt = build_prompt(scenario, options_text)

    res = call_llm(
        api_key=api_key,
        api_base=api_base,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        timeout=timeout,
        temperature=temperature,
    )
    if not res.get("ok"):
        return {
            "question_id": qid,
            "character_id": cid,
            "scenario_id": sid,
            "ok": False,
            "error": res.get("error"),
            "attempts": res.get("attempts"),
            "correct_answer": correct,
            "predicted": None,
            "is_correct": False,
        }

    parsed = res["parsed"] if isinstance(res.get("parsed"), dict) else {}
    raw = res.get("raw", "")
    pred = extract_decision(parsed, raw)
    return {
        "question_id": qid,
        "character_id": cid,
        "scenario_id": sid,
        "ok": True,
        "correct_answer": correct,
        "predicted": pred,
        "is_correct": pred == correct,
        "latency_s": res.get("latency_s"),
        "attempts": res.get("attempts"),
        "usage": res.get("usage"),
        "parsed": parsed,
        "raw": raw,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MCQ_MODEL", "gpt-5.4-mini"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", LLM_API_KEY))
    ap.add_argument(
        "--api-base", default=os.environ.get("API_BASE", LLM_API_BASE)
    )
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("TIMEOUT") or 120))
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--resume", default="", help="path to a predictions file; ok=true entries are skipped")
    args = ap.parse_args()

    inputs = load_inputs()
    questions = inputs["questions"]
    if args.limit > 0:
        questions = questions[: args.limit]

    done_ids: set[str] = set()
    if args.resume:
        try:
            prev = json.loads(Path(args.resume).read_text(encoding="utf-8"))
            for r in prev.get("predictions", []):
                if r.get("ok") and r.get("predicted") is not None:
                    done_ids.add(r["question_id"])
            print(f"resume: skipping {len(done_ids)} already-done questions")
        except Exception as e:  # noqa: BLE001
            print(f"resume failed: {e}; start fresh")

    pending = [q for q in questions if q["question_id"] not in done_ids]
    print(
        f"[BASELINE_MODEL_ONLY] model={args.model}  api_base={args.api_base}  "
        f"workers={args.workers}  questions={len(pending)}/{len(questions)}"
    )

    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    out_dir = RESULTS_ROOT / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions_baseline.json"
    summary_path = out_dir / "summary_baseline.json"

    results: list[dict] = []
    if args.resume:
        try:
            prev = json.loads(Path(args.resume).read_text(encoding="utf-8"))
            results.extend(prev.get("predictions", []))
        except Exception:  # noqa: BLE001
            pass

    lock = threading.Lock()
    completed = {"n": 0, "ok": 0, "correct": 0}
    t0 = time.time()

    def worker(q: dict) -> dict:
        r = process_one(
            q,
            inputs,
            api_key=args.api_key,
            api_base=args.api_base,
            model=args.model,
            timeout=args.timeout,
            temperature=args.temperature,
        )
        with lock:
            completed["n"] += 1
            if r.get("ok"):
                completed["ok"] += 1
                if r.get("is_correct"):
                    completed["correct"] += 1
            n = completed["n"]
            if n % 10 == 0 or n == len(pending):
                el = time.time() - t0
                rate = n / el if el > 0 else 0
                acc = (completed["correct"] / completed["ok"]) if completed["ok"] else 0
                print(
                    f"  {n}/{len(pending)}  ok={completed['ok']}  "
                    f"correct={completed['correct']}  acc(on_ok)={acc:.3f}  "
                    f"elapsed={el:.1f}s  rate={rate:.2f}/s  q={q['question_id']}"
                )
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, q): q for q in pending}
        save_every = 50
        since_save = 0
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                q = futures[fut]
                r = {
                    "question_id": q["question_id"],
                    "character_id": q["character_id"],
                    "scenario_id": q["scenario_id"],
                    "ok": False,
                    "error": f"worker exception: {e}",
                    "predicted": None,
                    "is_correct": False,
                }
            results.append(r)
            since_save += 1
            if since_save >= save_every:
                since_save = 0
                pred_path.write_text(
                    json.dumps({"predictions": results}, ensure_ascii=False),
                    encoding="utf-8",
                )

    # final write
    pred_path.write_text(
        json.dumps({"predictions": results}, ensure_ascii=False),
        encoding="utf-8",
    )

    # summary
    total = len(results)
    n_ok = sum(1 for r in results if r.get("ok"))
    n_correct = sum(1 for r in results if r.get("is_correct"))
    acc_overall = n_correct / total if total else 0.0
    acc_on_ok = n_correct / n_ok if n_ok else 0.0

    per_char: dict[str, dict[str, int]] = {}
    for r in results:
        cid = r.get("character_id", "?")
        d = per_char.setdefault(cid, {"total": 0, "ok": 0, "correct": 0})
        d["total"] += 1
        if r.get("ok"):
            d["ok"] += 1
        if r.get("is_correct"):
            d["correct"] += 1

    per_stage: dict[str, dict[str, int]] = {}
    for r in results:
        sid = r.get("scenario_id")
        sc = inputs["scenarios"].get(sid)
        stage = sc.get("stage") if sc else "?"
        d = per_stage.setdefault(stage, {"total": 0, "ok": 0, "correct": 0})
        d["total"] += 1
        if r.get("ok"):
            d["ok"] += 1
        if r.get("is_correct"):
            d["correct"] += 1

    # Letter distribution (to detect any letter bias in the model).
    letter_dist: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "None": 0}
    for r in results:
        p = r.get("predicted")
        if p in ("A", "B", "C", "D"):
            letter_dist[p] += 1
        else:
            letter_dist["None"] += 1

    summary = {
        "experiment": "baseline_model_only",
        "model": args.model,
        "api_base": args.api_base,
        "workers": args.workers,
        "temperature": args.temperature,
        "total_questions": total,
        "ok": n_ok,
        "correct": n_correct,
        "accuracy_overall": acc_overall,
        "accuracy_on_ok": acc_on_ok,
        "per_character": per_char,
        "per_stage": per_stage,
        "letter_distribution": letter_dist,
        "predictions_file": str(pred_path),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"DONE. total={total} ok={n_ok} correct={n_correct}")
    print(f"  accuracy_overall = {acc_overall:.4f}")
    print(f"  accuracy_on_ok   = {acc_on_ok:.4f}")
    print(f"  letter_dist      = {letter_dist}")
    print(f"  predictions -> {pred_path}")
    print(f"  summary     -> {summary_path}")


if __name__ == "__main__":
    main()
