"""
run_mem0.py — mem0 baseline runner.

For every MCQ in mcq.json:
- Use the pre-computed mem0 retrieval (cache/mem0/retrieval_all.jsonl) to fetch
  the top-K **mem0-extracted facts** for that (character_id, scenario_id).
- Build a prompt in the same shape as run_naive_rag.py and call the LLM.
- Parse decision_choice (A/B/C/D), compare with correct_answer, report accuracy.

Differences from run_naive_rag.py:
1. Retrieval source: mem0's retrieval_all.jsonl, where results[*].memory is an
   English fact extracted by the mem0 LLM. Each raw memory typically becomes 4-5 facts.
2. Memory text is taken directly from the mem0 `memory` field; we do NOT look up
   content_full in characters.json (per the design choice).
3. top-K is NOT de-duplicated; we take K facts in mem0's original ranking, even if
   multiple facts come from the same raw memory — that is mem0's default behaviour.

Inputs:
  cache/mem0/retrieval_all.jsonl       (704 lines = 11 char × 64 scenario × top-150 facts)
  benchmark/characters.json            (character basic info)
  benchmark/scenarios.json
  benchmark/mcq.json

Outputs:
  experiments/results/mem0/<model>/predictions_topK.json
  experiments/results/mem0/<model>/summary_topK.json
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

# Reuse the .env loader directly (do not import config because TIMEOUT in .env
# may be empty and would break config-time int parsing).
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
CACHE_DIR = PROJECT_ROOT / "cache"
CHARACTERS_PATH = DATA_DIR / "characters.json"
SCENARIOS_PATH = DATA_DIR / "scenarios.json"
MCQ_PATH = DATA_DIR / "mcq.json"
MEM0_RETRIEVAL_PATH = CACHE_DIR / "mem0" / "retrieval_all.jsonl"

RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results" / "mem0"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Prompt (mirrors run_naive_rag.py)
# ===========================================================================
SYSTEM_PROMPT = (
    "You are a role-play simulator. You will receive a person's past experiences and "
    "memory fragments together with the situation they are currently facing. Through "
    "those experiences understand who this person is — their thinking habits, "
    "emotional patterns, and behavioural tendencies — and then simulate the most "
    "likely real reaction in the current situation.\n\n"
    "Output strictly the following JSON (no other text, no markdown code block):\n"
    "{\n"
    '  "system_1_impulse": {\n'
    '    "thought": "First, instinctive reaction after seeing the trigger event (50-100 words)",\n'
    '    "emotion": "Primary emotion (Chinese + English, e.g. 极度焦虑 (Anxiety))",\n'
    '    "citation": "Which memories were activated (cite memory IDs and brief descriptions)"\n'
    "  },\n"
    '  "system_2_rational": {\n'
    '    "analysis": "Rational analysis after calming down (80-150 words)",\n'
    '    "plan": "Concrete plan of action (30-60 words)"\n'
    "  },\n"
    '  "inner_consciousness": "Combine system_1 emotional impulses with system_2 rational reasoning to give the inner reasons for the final decision (100-150 words, first person, naturally weave together emotional tone, core reasons, and value orientation; do not list bullet points)",\n'
    '  "final_decision": "The final behavioural decision: first an inner-monologue line saying \'I plan to do/say...\', then the actual outward behaviour (first person, matching this character\'s tone and expression habits, including action descriptions)",\n'
    '  "decision_choice": "If the question provides behavioural-decision options, output the letter of the option that best matches this character (e.g. A); otherwise output an empty string"\n'
    "}"
)


def build_basic_info(char: dict) -> str:
    """Neutral identity info: only character ID + occupation (matches naive_rag, prevents leakage)."""
    parts = [
        f"- Character ID: {char.get('id', 'N/A')}",
        f"- Occupation: {char.get('occupation', 'N/A')}",
    ]
    return "\n".join(parts)


def anonymize_mem_id(mem_id: str) -> str:
    """Strip leaky trait tags (N_HIGH/C_LOW/E_HIGH/.../NEUTRAL) from mem_id.
    Original format: MEM_CHAR_01_N_HIGH_0049 -> MEM_CHAR_01_0049
                     MEM_CHAR_11_NEUTRAL_0001 -> MEM_CHAR_11_0001
    """
    if not isinstance(mem_id, str):
        return mem_id
    m = re.match(r"^(MEM_CHAR_\d+)_(?:[NCEAO]_(?:HIGH|LOW)|NEUTRAL)_(\d+)$", mem_id)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return mem_id


def build_prompt(
    char: dict,
    scenario: dict,
    retrieved_facts: list[dict],
    options_text: str | None = None,
) -> str:
    """Same structure as naive_rag's build_prompt, only the memory list fields differ.
    Each fact = {rank, score, memory, mem_id, timeline}.
    """
    mem_str = "\n".join(
        f"  - [{anonymize_mem_id(rec.get('mem_id', '?'))}][{rec.get('timeline', '?')}] "
        f"{rec.get('memory', '')}"
        for rec in retrieved_facts
    )

    setting = scenario.get("setting") or {}
    trigger = scenario.get("trigger_event") or {}

    prompt = f"""## Background
{build_basic_info(char)}

## Key Social Relationships
  N/A

## Past Experiences
The following are important fragments from this person's life — use them to understand who this person is:
{mem_str}

## Current Situation
Scene: {scenario.get('name', 'Unknown')}
Location: {setting.get('location', 'Unknown')} | Time: {setting.get('time', 'Unknown')} | Atmosphere: {setting.get('atmosphere', 'Unknown')}

Context: {scenario.get('context_text', 'Unknown')}

## Trigger Event
Sender: {trigger.get('sender', 'Unknown')}
Message: {trigger.get('message_content', 'Unknown')}
Action required: {trigger.get('action_required', 'Unknown')}

## Task
Using the experiences above, understand this person's thinking patterns, emotional tendencies, and behavioural habits, then simulate the real reaction they would have in the current situation.

Requirements:
1. System 1 (intuitive impulse): the person's first reaction; cite the activated memories.
2. System 2 (rational analysis): how the person would analyse and reason after calming down.
3. Final Decision: two parts — inner_consciousness is the inner monologue 'I plan to do/say ...' (the last layer of consciousness before outward behaviour, fusing emotional tone, core reasons, and value orientation); response_text is what the person actually says/sends."""

    if options_text:
        prompt += f"""

## Behavioural Decision Options
Below are possible behavioural decisions different people might take in this scenario. Pick the one that best matches you (in this character's role) and output the corresponding letter in the decision_choice field:

{options_text}"""

    return prompt


def build_options_text(options: list[dict]) -> str:
    """[{label, content, ...}, ...] -> 'A. ...\\n\\nB. ...'"""
    return "\n\n".join(f"{o['label']}. {o['content']}" for o in options)


# ===========================================================================
# LLM call
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
    """Call an OpenAI-compatible chat completions endpoint. Returns a dict (raw / parsed / usage / latency)."""
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
                    "User-Agent": "mem0-rag/1.0",
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
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = 2.0 * (2 ** (attempt - 1))
                time.sleep(wait)
                continue
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
# Inputs
# ===========================================================================
def load_inputs(top_k: int) -> dict[str, Any]:
    chars_data = json.loads(CHARACTERS_PATH.read_text(encoding="utf-8"))
    characters = {c["id"]: c for c in chars_data["characters"]}

    scen_data = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    scenarios = {}
    for stage, lst in scen_data["scenarios"].items():
        for sc in lst:
            scenarios[sc["id"]] = sc

    mcq_data = json.loads(MCQ_PATH.read_text(encoding="utf-8"))
    questions = mcq_data["questions"]

    # mem0 retrieval: JSONL, each row = (char_id, scenario_id, query_text, results[150]).
    # Indexed as (char_id, scenario_id) -> [fact dict, ...].
    mem0_index: dict[tuple[str, str], list[dict]] = {}
    n_rows = 0
    with MEM0_RETRIEVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = (row["char_id"], row["scenario_id"])
            mem0_index[key] = row.get("results") or []
            n_rows += 1
    print(f"loaded mem0 retrieval: {n_rows} (char,scenario) entries")

    return {
        "characters": characters,
        "scenarios": scenarios,
        "questions": questions,
        "mem0_index": mem0_index,
        "top_k": top_k,
    }


def get_topk_facts(mem0_index: dict, char_id: str, scenario_id: str, top_k: int) -> list[dict]:
    facts = mem0_index.get((char_id, scenario_id))
    if not facts:
        return []
    return facts[:top_k]


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

    char = inputs["characters"].get(cid)
    scenario = inputs["scenarios"].get(sid)
    facts = get_topk_facts(inputs["mem0_index"], cid, sid, inputs["top_k"])

    if char is None or scenario is None:
        return {
            "question_id": qid,
            "character_id": cid,
            "scenario_id": sid,
            "ok": False,
            "error": f"missing char or scenario (char={char is not None}, scen={scenario is not None})",
            "correct_answer": correct,
            "predicted": None,
            "is_correct": False,
        }

    options_text = build_options_text(options)
    user_prompt = build_prompt(char, scenario, facts, options_text=options_text)

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
            "num_facts": len(facts),
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
        "num_facts": len(facts),
        "latency_s": res.get("latency_s"),
        "attempts": res.get("attempts"),
        "usage": res.get("usage"),
        "parsed": parsed,
        "raw": raw,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--model", default=os.environ.get("MCQ_MODEL", "gpt-5.4-mini"))
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", LLM_API_KEY))
    ap.add_argument(
        "--api-base", default=os.environ.get("API_BASE", LLM_API_BASE)
    )
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("TIMEOUT", 120)))
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--out-prefix",
        default=time.strftime("%Y%m%d_%H%M%S"),
        help="filename prefix (defaults to timestamp)",
    )
    ap.add_argument(
        "--resume",
        default="",
        help="path to a predictions file; ok=true entries are skipped",
    )
    args = ap.parse_args()

    if args.top_k > 150:
        print(f"WARNING: top-k={args.top_k} > 150 (mem0 retrieval pool size). "
              f"Will use 150 max.")

    inputs = load_inputs(top_k=args.top_k)
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
        f"model={args.model}  api_base={args.api_base}  top_k={args.top_k}  "
        f"workers={args.workers}  questions={len(pending)}/{len(questions)}"
    )

    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    out_dir = RESULTS_ROOT / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"predictions_top{args.top_k}.json"
    summary_path = out_dir / f"summary_top{args.top_k}.json"

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

    pred_path.write_text(
        json.dumps({"predictions": results}, ensure_ascii=False),
        encoding="utf-8",
    )

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

    summary = {
        "experiment": "mem0",
        "model": args.model,
        "api_base": args.api_base,
        "top_k": args.top_k,
        "workers": args.workers,
        "temperature": args.temperature,
        "retrieval_source": str(MEM0_RETRIEVAL_PATH),
        "total_questions": total,
        "ok": n_ok,
        "correct": n_correct,
        "accuracy_overall": acc_overall,
        "accuracy_on_ok": acc_on_ok,
        "per_character": per_char,
        "per_stage": per_stage,
        "predictions_file": str(pred_path),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"DONE. total={total} ok={n_ok} correct={n_correct}")
    print(f"  accuracy_overall = {acc_overall:.4f}")
    print(f"  accuracy_on_ok   = {acc_on_ok:.4f}")
    print(f"  predictions -> {pred_path}")
    print(f"  summary     -> {summary_path}")


if __name__ == "__main__":
    main()
