"""
evaluate_results.py — LLM-as-a-judge evaluation of Agent outputs
(predictions_*.json files under experiments/results/) against the GT annotations.

Scoring dimensions (0-5 scale):
  - emotional_tone    consistency of emotional tone with GT
  - core_reasoning    consistency of key reasoning with GT
  - value_orientation consistency of value orientation with GT
  - summary           consistency of the integrated-consciousness summary with GT

Final score (CD-HL Score):
  CD-HL Score = avg(emotional_tone, core_reasoning, value_orientation, summary) → normalised to 0-1

Input:  experiments/results/{method}/{model}/predictions_*.json
Output: experiments/evaluation/{method}/{model}/evaluation_{predictions_stem}.json

Usage:
    # Evaluate a single predictions file
    python evaluation/evaluate_results.py experiments/results/mem0/claude-sonnet-4-6/predictions_top150.json

    # Evaluate every predictions_*.json under one method × model directory
    python evaluation/evaluate_results.py --method mem0 --model claude-sonnet-4-6

    # Evaluate every model under one method
    python evaluation/evaluate_results.py --method mem0

    # No args: evaluate every method × model under experiments/results
    python evaluation/evaluate_results.py --all
"""

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ================= Configuration =================

def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_load_env_file(_BASE_DIR / ".env")

EXPERIMENTS_RESULTS_DIR    = _BASE_DIR / "experiments" / "results"
EXPERIMENTS_EVALUATION_DIR = _BASE_DIR / "experiments" / "evaluation"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))
from utils.http_client import openai_compatible_request_headers  # noqa: E402
from utils.data_paths import (  # noqa: E402
    resolve_characters_json,
    resolve_gt_annotations_path,
    resolve_scenarios_json,
)

def _env_or(*names: str, default: str = "") -> str:
    """Return the first non-empty value from a list of env var names."""
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    return default


JUDGE_CONFIG = {
    "api_key":          _env_or("JUDGE_API_KEY", "API_KEY"),
    "api_base":         _env_or("JUDGE_API_BASE", "API_BASE").rstrip("/"),
    "model_name":       _env_or("JUDGE_MODEL_NAME", "MODEL_NAME"),
    "timeout":          _int_env("JUDGE_TIMEOUT", _int_env("TIMEOUT", 90)),
    # Disable reasoning / thinking for models that ship with it on by default
    # (gemini-3 / gpt-5 / o-series). "" = leave the flag off; "minimal" / "none" recommended.
    "reasoning_effort": _env_or("JUDGE_REASONING_EFFORT", default="minimal"),
}

MAX_WORKERS       = int(os.getenv("EVAL_WORKERS", "4"))
JUDGE_MAX_RETRIES = _int_env("JUDGE_MAX_RETRIES", 6)


# ================= 6-tier rubric (0-5) =================

RUBRIC = """
## Scoring rubric (6 tiers, applies to emotional_tone / core_reasoning / value_orientation / summary)

【General principle — be strict】
- Default starting point is 2. The score should only go up with sufficient evidence.
- "Roughly the same direction" deserves only a 2; the Agent must **reproduce both the core elements of the GT and the character's individuality** to deserve 3 or above.
- A 5 is extremely rare and only awarded when the Agent output and GT are nearly interchangeable.
- An Agent that is generic, templated, or lacking individual character imprint → **at most 2**.
- Do NOT give a high score just because the text "reads fluently" or "sounds reasonable"; compare against the GT line by line.

5 = Outstanding: nearly interchangeable with GT in content, intensity, and character imprint; no obvious omissions.
4 = Excellent: every core element is hit and the character's unique logic is visible; only minor secondary details are missing.
3 = Adequate: main elements are hit, but 1-2 key details are missing or the wording is slightly generic.
2 = Marginal: direction is right, but the content is clearly incomplete, too generic, or character traits are weak (default starting point).
1 = Failing: direction is wrong, contradicts GT, or conflicts with the character profile.
0 = Missing or invalid: Agent output is empty, malformed, or unrelated to this dimension.

## Per-dimension anchors

**emotional_tone**
5 = Emotion type, intensity, and layers (including secondary emotions and bodily sensations) all match GT exactly.
4 = Main emotion type and intensity match; layered structure is preserved; only individual secondary emotions or bodily sensations are missing.
3 = Main emotion type matches, but intensity is off or layers are missing (e.g. only "anxiety", missing "moral conflict").
2 = Only the broad direction matches (both negative or both positive); specific type drifts or emotion is flattened.
1 = Emotion direction is opposite or completely unrelated to GT.
0 = No emotion expressed or content missing.

**core_reasoning**
5 = Fully reproduces GT's core driver and clearly shows the character's unique cognitive logic of "because I went through XX → I instinctively YY".
4 = Core driver matches and character imprint is visible; only details are lacking.
3 = Main driver matches but lacks the character-specific causal chain, or sounds more like rational deliberation than instinct.
2 = Touches some logic but is heavily generic; the character's specific cognitive pattern is not visible.
1 = Driver is entirely different from GT, or contradicts the character profile.
0 = No reasoning expressed or content missing.

**value_orientation**
5 = Trade-off stance matches GT exactly: priority items, abandoned items, priority order, and "unacceptable" items all align.
4 = Priority and abandoned items match; only secondary items or ordering details deviate slightly.
3 = Direction of trade-off matches, but lacks the explicit "unacceptable" stance from GT or the priority is fuzzy.
2 = Some directions of trade-off match; stance is vague or significant items are missing.
1 = Stance is opposite to GT or undecidable.
0 = No value orientation expressed or content missing.

**summary**
5 = Integrated-consciousness narrative aligns with the GT summary on emotional impact, motivation layers, stance, action direction, and character individuality.
4 = Key elements all match; only expression details or non-core plot points differ slightly.
3 = Main direction and behavioural decision align, but lack the GT's emotional depth, key impact points, or character individuality.
2 = Only some elements match; overall flat, or lacks the layers and individual imprint of the GT.
1 = Direction contradicts GT or is entirely off.
0 = Agent integrated-consciousness is empty, malformed, or unrelated to the scenario.
"""

SYSTEM_PROMPT = f"""You are a professional psychological / personality researcher and a **strict** LLM-as-a-judge evaluator.

Your task: compare the Agent output with the expert-annotated Ground Truth (GT) and **strictly** score four dimensions (each 0-5).

【Stance】
- Your default stance is sceptical: the Agent output must be **well evidenced** to earn 3 or higher.
- Do NOT inflate scores because the output "reads fluently" or "sounds reasonable"; compare line by line against the GT.
- If the Agent is missing a core element from the GT (a specific emotion layer / character-unique logic / explicit stance), deduct points even when other parts are correct.
- Average-quality output = 2; reaching 3 needs clear achievement; 4 needs excellence; 5 is extremely rare.
- Mediocrity is not a high score; only details that genuinely match the GT and bear character imprint deserve high marks.

{RUBRIC}

## Output format

Output strictly the following JSON (no markdown code block, no explanation, no extra text):
{{"emotional_tone": int 0-5, "core_reasoning": int 0-5, "value_orientation": int 0-5, "summary": int 0-5}}"""


# ================= LLM Judge =================

def _sanitize_json(text: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


class LLMJudge:
    def __init__(self, config: dict):
        self.api_key          = config["api_key"]
        self.api_base         = config["api_base"]
        self.model_name       = config["model_name"]
        self.timeout          = config["timeout"]
        self.reasoning_effort = config.get("reasoning_effort", "")

    def _call(self, user_prompt: str) -> dict:
        import urllib.request
        import urllib.error
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        if self.reasoning_effort:
            # OpenAI-compatible field; effective for Gemini 3 (aihubmix) / GPT-5 series.
            payload["reasoning_effort"] = self.reasoning_effort
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=openai_compatible_request_headers(self.api_key),
            method="POST",
        )
        for attempt in range(JUDGE_MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    content = json.loads(resp.read().decode("utf-8")) \
                        .get("choices", [{}])[0].get("message", {}).get("content", "")
                if content.startswith("```"):
                    content = "\n".join(l for l in content.split("\n") if not l.startswith("```"))
                return json.loads(_sanitize_json(content.strip()))
            except urllib.error.HTTPError as e:
                # Retry on 403/429/5xx with exponential backoff, capped at 60s.
                if attempt < JUDGE_MAX_RETRIES and e.code in (403, 408, 429, 500, 502, 503, 504):
                    sleep = min(60, 2 ** attempt + (attempt * 0.7))
                    time.sleep(sleep)
                    continue
                raise RuntimeError(f"Judge call failed: HTTP {e.code} {e.reason}")
            except Exception as e:
                if attempt < JUDGE_MAX_RETRIES:
                    sleep = min(30, 2 ** attempt)
                    time.sleep(sleep)
                else:
                    raise RuntimeError(f"Judge call failed: {e}")


# ================= Prompt Builder =================

def build_judge_prompt(char_data: dict, scenario_data: dict, gt: dict, agent_ic: str) -> str:
    bf = char_data.get("big_five", {})
    bf_str = " | ".join(f"{k}: {v:.2f}" for k, v in bf.items())

    trigger = scenario_data.get("trigger_event", {})
    gt_ic   = gt.get("inner_consciousness", {})

    return f"""## Character info
- ID: {char_data['id']}  Archetype: {char_data.get('archetype', '')}
- Big Five: {bf_str}
- Self-value logic: {char_data.get('self_value_logic', '')}

## Scenario info
- Scene: {scenario_data.get('name', '')} ({scenario_data.get('stage', '')})
- Context: {scenario_data.get('context_text', '')}
- Trigger event: {trigger.get('message_content', '')}
- Decision required: {trigger.get('action_required', '')}

## Ground Truth (expert annotation)

### GT integrated consciousness
{gt_ic.get('summary', '')}

- Emotional tone: {gt_ic.get('emotional_tone', '')}
- Key reasoning : {gt_ic.get('core_reasoning', '')}
- Value orient. : {gt_ic.get('value_orientation', '')}

## Agent output

### Agent integrated consciousness
{agent_ic}

## Scoring task (strict mode)
Following the rubric, score the four dimensions **strictly** (each 0-5).
- Default starting point is 2; scores ≥ 3 require sufficient evidence; 5 is extremely rare.
- emotional_tone / core_reasoning / value_orientation: compare with the three GT label fields.
- summary: compare with the GT integrated-consciousness summary field.
- If the Agent integrated consciousness is empty, malformed, or unrelated to the scenario, give 0 on the relevant dimensions.
- Output JSON scores only, no explanation."""


# ================= Scoring =================

def _clip_score(v) -> int:
    try:
        s = int(v)
    except Exception:
        return 0
    return max(0, min(5, s))


def compute_scores(judge_result: dict) -> dict:
    et = _clip_score(judge_result.get("emotional_tone", 0))
    cr = _clip_score(judge_result.get("core_reasoning", 0))
    vo = _clip_score(judge_result.get("value_orientation", 0))
    sm = _clip_score(judge_result.get("summary", 0))

    # Mean over 0-5, normalised to 0-1
    consciousness_raw = (et + cr + vo + sm) / 4
    cdhl_score = round(consciousness_raw / 5, 4)

    return {
        "emotional_tone":    et,
        "core_reasoning":    cr,
        "value_orientation": vo,
        "summary":           sm,
        "cdhl_score":        cdhl_score,
    }


# ================= Single Case =================

def evaluate_single(judge: LLMJudge, char_data, scenario_data, gt, agent_ic,
                    scen_id, char_id, question_id):
    prompt = build_judge_prompt(char_data, scenario_data, gt, agent_ic)
    raw    = judge._call(prompt)
    scores = compute_scores(raw)
    return {
        "question_id":  question_id,
        "scenario_id":  scen_id,
        "character_id": char_id,
        "scores":       scores,
    }


# ================= Summary =================

def build_summary(evaluations: list) -> dict:
    if not evaluations:
        return {}
    n = len(evaluations)
    keys = ["emotional_tone", "core_reasoning", "value_orientation", "summary", "cdhl_score"]
    sums = {k: 0.0 for k in keys}
    for e in evaluations:
        for k in keys:
            sums[k] += e["scores"].get(k, 0)
    return {
        "num_samples": n,
        **{f"mean_{k}": round(sums[k] / n, 4) for k in keys},
    }


def print_summary(evaluations: list, header: str = ""):
    s = build_summary(evaluations)
    n = s['num_samples']
    # Mean of the 0-5 scores normalised to 0-1 for display.
    et = round(s['mean_emotional_tone']    / 5, 4)
    cr = round(s['mean_core_reasoning']    / 5, 4)
    vo = round(s['mean_value_orientation'] / 5, 4)
    sm = round(s['mean_summary']           / 5, 4)
    print(f"\n{'='*60}")
    if header:
        print(f"  {header}")
    print(f"  Evaluation Results ({n} samples)")
    print(f"  {'─'*50}")
    print(f"  Per-dimension (raw 0-5 / normalized 0-1):")
    print(f"    Emotional Tone:     {s['mean_emotional_tone']:.3f}  →  {et:.4f}")
    print(f"    Core Reasoning:     {s['mean_core_reasoning']:.3f}  →  {cr:.4f}")
    print(f"    Value Orientation:  {s['mean_value_orientation']:.3f}  →  {vo:.4f}")
    print(f"    Summary:            {s['mean_summary']:.3f}  →  {sm:.4f}")
    print(f"  {'─'*50}")
    print(f"  CD-HL Score          (0-1): {s['mean_cdhl_score']:.4f}")
    print(f"{'='*60}")


# ================= Checkpoint =================

def save_checkpoint(output_path: Path, evaluations: list, source_meta: dict):
    tmp = output_path.with_suffix(".tmp")
    data = {
        "meta": {
            "judge_model": JUDGE_CONFIG["model_name"],
            "score_range": "0-5",
            "timestamp":   int(time.time()),
            **source_meta,
        },
        "summary":     build_summary(evaluations),
        "evaluations": evaluations,
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(output_path)


# ================= Predictions Loader =================

def load_predictions(pred_file: Path) -> list:
    """Read experiments/results/.../predictions_*.json and return the prediction list."""
    with open(pred_file, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "predictions" in data:
        return data["predictions"]
    raise ValueError(f"Unrecognized predictions format: {pred_file}")


def extract_agent_ic(pred_item: dict) -> str:
    """Extract the Agent's inner_consciousness text from a prediction item."""
    parsed = pred_item.get("parsed") or {}
    ic = parsed.get("inner_consciousness")
    if isinstance(ic, str):
        return ic
    if isinstance(ic, dict):
        # Tolerate field-name variants
        return ic.get("summary") or ic.get("text") or json.dumps(ic, ensure_ascii=False)
    return ""


# ================= Per-File Evaluation =================

def evaluate_predictions_file(
    pred_file: Path,
    judge: LLMJudge,
    gt_flat: dict,
    chars_db: dict,
    scenarios_db: dict,
) -> Path:
    """Evaluate a single predictions_*.json file; output goes to evaluation_*.json under the same path."""
    print(f"\n{'#'*60}")
    print(f"# File: {pred_file.relative_to(_BASE_DIR)}")
    print(f"{'#'*60}")

    predictions = load_predictions(pred_file)

    # Parse method/model from the path.
    try:
        rel = pred_file.relative_to(EXPERIMENTS_RESULTS_DIR)
        parts = rel.parts
        method = parts[0] if len(parts) >= 1 else ""
        model  = parts[1] if len(parts) >= 2 else ""
    except ValueError:
        # Not under experiments/results: drop the evaluation file at the evaluation root.
        rel = Path(pred_file.name)
        method, model = "", ""

    # Output path: experiments/evaluation/{method}/{model}/evaluation_<predictions stem>.json
    if method and model:
        out_dir = EXPERIMENTS_EVALUATION_DIR / method / model
    else:
        out_dir = EXPERIMENTS_EVALUATION_DIR / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"evaluation_{pred_file.stem}.json"

    source_meta = {
        "method":           method,
        "model":            model,
        "predictions_file": str(pred_file.relative_to(_BASE_DIR)),
    }

    # Resume from checkpoint
    evaluations = []
    done_keys: set = set()
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                ckpt = json.load(f)
            evaluations = ckpt.get("evaluations", [])
            done_keys = {
                e.get("question_id") or f"{e['scenario_id']}|{e['character_id']}"
                for e in evaluations
            }
            print(f"Checkpoint: {len(evaluations)} already done, resuming...")
        except Exception:
            evaluations = []

    # Build the task list
    tasks = []
    skipped = 0
    for item in predictions:
        if not item.get("ok", True):
            skipped += 1
            continue

        scen_id     = item.get("scenario_id")
        char_id     = item.get("character_id")
        question_id = item.get("question_id") or f"{scen_id}|{char_id}"

        if question_id in done_keys:
            continue

        agent_ic = extract_agent_ic(item)
        if not agent_ic:
            skipped += 1
            continue

        gt = gt_flat.get(char_id, {}).get(scen_id)
        char_data = chars_db.get(char_id)
        scen_data = scenarios_db.get(scen_id)

        if not gt or not char_data or not scen_data:
            print(f"  Skip {question_id}: missing GT/char/scenario data")
            skipped += 1
            continue

        tasks.append((char_data, scen_data, gt, agent_ic, scen_id, char_id, question_id))

    print(f"  Method: {method} | Model: {model}")
    print(f"  Tasks: {len(tasks)} | Skipped: {skipped} | Done(checkpoint): {len(done_keys)}")

    if not tasks:
        print("  All cases already evaluated.")
        if evaluations:
            print_summary(evaluations, header=f"{method}/{model} ({pred_file.stem})")
        return output_path

    write_lock = threading.Lock()
    pbar = tqdm(total=len(tasks), desc=f"Eval {method}/{model}", unit="case") if HAS_TQDM else None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {
            pool.submit(evaluate_single, judge, *task): task[-3:]  # (scen_id, char_id, question_id)
            for task in tasks
        }
        for future in as_completed(future_map):
            scen_id, char_id, question_id = future_map[future]
            try:
                result = future.result()
                with write_lock:
                    evaluations.append(result)
                    save_checkpoint(output_path, evaluations, source_meta)
            except Exception as e:
                print(f"  FAIL: {question_id}: {e}")
            finally:
                if pbar:
                    pbar.update(1)

    if pbar:
        pbar.close()

    save_checkpoint(output_path, evaluations, source_meta)
    print_summary(evaluations, header=f"{method}/{model} ({pred_file.stem})")
    try:
        print(f"  Saved to: {output_path.relative_to(_BASE_DIR)}")
    except ValueError:
        print(f"  Saved to: {output_path}")
    return output_path


# ================= Main =================

def discover_pred_files(method: str = None, model: str = None, target: Path = None) -> list:
    """Discover every predictions_*.json file matching the CLI args."""
    if target is not None:
        # single file
        return [target] if target.is_file() else []

    if method and model:
        d = EXPERIMENTS_RESULTS_DIR / method / model
        return sorted(d.glob("predictions_*.json"))

    if method and not model:
        d = EXPERIMENTS_RESULTS_DIR / method
        return sorted(d.glob("*/predictions_*.json"))

    # everything
    return sorted(EXPERIMENTS_RESULTS_DIR.glob("*/*/predictions_*.json"))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Agent outputs in experiments/results/.")
    p.add_argument("file", nargs="?", help="single predictions_*.json path")
    p.add_argument("--method", help="experiment method name (e.g. mem0, oracle_full)")
    p.add_argument("--model",  help="model name (e.g. claude-sonnet-4-6)")
    p.add_argument("--all", action="store_true", help="evaluate every predictions_*.json under experiments/results")
    return p.parse_args()


def run_evaluation():
    args = parse_args()

    target = Path(args.file).resolve() if args.file else None
    if target and not target.exists():
        print(f"Error: {target} not found.")
        return

    pred_files = discover_pred_files(method=args.method, model=args.model, target=target)
    if not pred_files:
        print("Error: No predictions_*.json files matched.")
        print(f"  experiments root: {EXPERIMENTS_RESULTS_DIR}")
        return

    # 1. Load GT annotations
    gt_path = resolve_gt_annotations_path(_BASE_DIR / "benchmark")
    if gt_path is None:
        print(
            "Error: GT annotations not found. Set GT_ANNOTATIONS_PATH or add "
            "benchmark/ground_truth.json"
        )
        return
    print(f"Loading GT: {gt_path}")
    with open(gt_path, encoding="utf-8") as f:
        gt_data = json.load(f).get("annotations", {})
    gt_flat = {}
    for cid, stages in gt_data.items():
        gt_flat[cid] = {}
        for stage_anns in stages.values():
            gt_flat[cid].update(stage_anns)
    total_gt = sum(len(v) for v in gt_flat.values())
    print(f"Loaded GT annotations: {total_gt} entries")

    # 2. Load characters and scenarios
    chars_path = resolve_characters_json(_BASE_DIR / "benchmark")
    if chars_path is None:
        print(
            "Error: Characters JSON not found. Set CHARACTERS_JSON_PATH or add "
            "benchmark/characters.json"
        )
        return
    with open(chars_path, encoding="utf-8") as f:
        chars_db = {c["id"]: c for c in json.load(f)["characters"]}

    scenarios_db = {}
    scen_path = resolve_scenarios_json(_BASE_DIR / "benchmark")
    if scen_path is not None:
        with open(scen_path, encoding="utf-8") as f:
            raw = json.load(f).get("scenarios", {})
            items = [s for stage in raw.values() for s in stage] if isinstance(raw, dict) else raw
            for s in items:
                scenarios_db[s["id"]] = s

    if not JUDGE_CONFIG["api_key"]:
        print("Error: JUDGE_API_KEY not set in .env")
        return

    judge = LLMJudge(JUDGE_CONFIG)

    print(f"\n{'='*60}")
    print(f"  Agent Personality Evaluation (0-5 score)")
    print(f"  Judge:    {JUDGE_CONFIG['model_name']}")
    print(f"  Workers:  {MAX_WORKERS}")
    print(f"  Files:    {len(pred_files)} predictions_*.json")
    print(f"{'='*60}")

    for pf in pred_files:
        try:
            evaluate_predictions_file(pf, judge, gt_flat, chars_db, scenarios_db)
        except Exception as e:
            print(f"  FAIL on {pf}: {e}")

    print(f"\nAll done. Evaluated {len(pred_files)} file(s).")


if __name__ == "__main__":
    run_evaluation()
