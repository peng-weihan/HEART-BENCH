
import json
import random
import re
import time
import os
import hashlib
from pathlib import Path
import urllib.request
import urllib.error

MAX_RETRIES = 2  # max retry attempts after failure (MAX_RETRIES + 1 attempts total)


def _sanitize_json(text: str) -> str:
    """Sanitise JSON text returned by the LLM by stripping illegal control chars (keeping \\n \\r \\t)."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


def _safe_filename_token(text: str) -> str:
    """Convert arbitrary model name into a safe filename token."""
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", (text or "").strip())
    token = token.strip("._-")
    return token or "unknown_model"


def _load_retrieval_cache(path: Path) -> dict:
    """Load retrieval cache payload from disk."""
    if not path.exists():
        return {"meta": {}, "items": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"meta": {}, "items": {}}
    if not isinstance(payload, dict):
        return {"meta": {}, "items": {}}
    items = payload.get("items", {})
    if not isinstance(items, dict):
        items = {}
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    return {"meta": meta, "items": items}


def _save_retrieval_cache(path: Path, payload: dict) -> None:
    """Atomically save retrieval cache payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _load_benchmark_checkpoint(path: Path) -> tuple[dict, list]:
    """Load benchmark checkpoint safely."""
    if not path.exists():
        return {}, []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}, []
    if not isinstance(payload, dict):
        return {}, []
    meta = payload.get("metadata", {})
    results = payload.get("results", [])
    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(results, list):
        results = []
    return meta, results


def _case_key(scenario_id: str, character_id: str) -> str:
    return f"{scenario_id}::{character_id}"


def _retrieval_cache_key(
    retrieval_mode: str,
    char_id: str,
    scenario_id: str,
    stage: str,
    top_k: int,
    context_text: str,
) -> str:
    digest = hashlib.sha1((context_text or "").encode("utf-8")).hexdigest()
    return f"{retrieval_mode}|{char_id}|{scenario_id}|{stage}|topk={top_k}|q={digest}"

# ================= Configuration =================
def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


_ENV_PATH = Path(__file__).with_name(".env")
_load_env_file(_ENV_PATH)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_api_key = os.getenv("API_KEY", "").strip() or os.getenv("ANNOTATE_API_KEY", "").strip()
_api_base = os.getenv("API_BASE", "").strip() or os.getenv("ANNOTATE_API_BASE", "").strip()

API_CONFIG = {
    "api_key": _api_key,
    "api_base": _api_base,
    "model_name": os.getenv("MODEL_NAME", "gpt-5-mini"),
    "timeout": _int_env("TIMEOUT", 60),
}

from utils.data_paths import (
    activated_top5_memory_ids,
    resolve_activated_memories_path,
    resolve_characters_json,
    resolve_gt_annotations_path,
    resolve_mc_questions_path,
    resolve_scenarios_json,
)
from utils.memory_retrieval import RandomMemoryRetriever, create_memory_index
from utils.http_client import openai_compatible_request_headers

# ================= Classes =================
class Character:
    def __init__(self, data):
        self.id = data['id']
        self.name = data.get('name', '')
        self.description = data.get('description', '')
        self.big_five = data.get('big_five', {})
        self.value_orientation = data.get('value_orientation', {})
        self.self_value_logic = data.get('self_value_logic', '')
        self.semantic_memory = data.get('semantic_memory', {})
        self.episodic_memory = data.get('episodic_memory_set', [])

    def retrieve_memories(self, context, top_k=3, memory_index=None, stage=None):
        if memory_index is None:
            raise RuntimeError("memory_index is required. Set EMBEDDING_API_KEY in .env.")
        return memory_index.query(self.id, context, top_k=top_k, stage=stage)

class Scenario:
    def __init__(self, data):
        self.id = data['id']
        self.name = data.get('name', '')
        self.category = data.get('category', '')
        self.stage = data.get('stage', '')
        self.age = data.get('age')
        self.context = data['context_text']
        self.trigger = data['trigger_event']
        self.setting = data.get('setting', {})
        self.assessed_dimensions = data.get('assessed_dimensions', data.get('stress_factors', {}))

class OpenAICompatibleLLM:
    def __init__(self, api_config: dict):
        self.api_key = api_config["api_key"]
        self.api_base = api_config["api_base"].rstrip("/")
        self.model_name = api_config["model_name"]
        self.timeout = api_config["timeout"]

    def generate(self, prompt):
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a role-play simulator. You will be given a person's past experiences and memory fragments, "
                        "along with the situation this person is currently facing. "
                        "Use those experiences to deeply understand this person — their thought habits, emotional patterns, "
                        "and behavioural tendencies — and simulate the most likely authentic reaction they would have in the "
                        "current situation.\n\n"
                        "Output strictly in the following JSON format (do not output anything else, do not use markdown code blocks):\n"
                        "{\n"
                        '  "system_1_impulse": {\n'
                        '    "thought": "The person\'s first reaction and gut instinct after seeing the trigger event (50-100 words)",\n'
                        '    "emotion": "Primary emotion (Chinese + English, e.g. 极度焦虑 (Anxiety))",\n'
                        '    "citation": "Which memories were triggered (cite memory IDs and brief description)"\n'
                        "  },\n"
                        '  "system_2_rational": {\n'
                        '    "analysis": "The person\'s rational analysis after calming down (80-150 words)",\n'
                        '    "plan": "The action plan the person formulates (30-60 words)"\n'
                        "  },\n"
                        '  "inner_consciousness": "Combining the emotional impulse of system_1 and the rational reasoning of system_2, the inner reason this person reaches the final decision (100-150 words, first person, naturally fusing emotional tone / core rationale / value orientation; do not enumerate as bullet points)",\n'
                        '  "final_decision": "The final behavioural decision: first write one sentence of inner monologue describing \'what I am about to do/say\', then describe the actual outward behaviour (first person, matching the person\'s tone and expression, including action description)",\n'
                        '  "decision_choice": "If the question provides a list of decision options, output the letter that best fits this character here (e.g. A); otherwise leave it as an empty string"\n'
                        "}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=openai_compatible_request_headers(self.api_key),
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            raise RuntimeError(f"LLM HTTPError {e.code}: {detail}")
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {e}")

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

# ================= Engine =================
def build_prompt(char, scenario, retrieved_memories, options_text=None):
    """Build prompt with only raw memories and factual info — no personality summaries."""
    mem_str = "\n".join(
        f"  - [{m.get('id', '?')}][{m.get('timeline', '?')}] {m.get('content_full', m.get('content_summary', ''))}"
        for m in retrieved_memories
    )

    relationships = char.semantic_memory.get('core_social_relationships', [])
    rel_str = "\n".join(
        f"  - {r['target']}: {r['relation']}" for r in relationships
    ) if relationships else "  N/A"

    prompt = f"""## Background
- Capabilities: {char.semantic_memory.get('capabilities', 'N/A')}

## Key Social Relationships
{rel_str}

## Past Experiences
Below are key episodes from this person's life. Use them to understand who they are:
{mem_str}

## Current Situation
Scenario: {scenario.name}
Location: {scenario.setting.get('location', 'unknown')} | Time: {scenario.setting.get('time', 'unknown')} | Atmosphere: {scenario.setting.get('atmosphere', 'unknown')}

Context: {scenario.context}

## Trigger Event
Sender: {scenario.trigger.get('sender', 'unknown')}
Message: {scenario.trigger.get('message_content', 'unknown')}
Action required: {scenario.trigger.get('action_required', 'unknown')}

## Task
Using the episodes above, infer this person's thought patterns, emotional tendencies, and behavioural habits, then simulate how they would actually react in the current situation.

Requirements:
1. System 1 (gut impulse): the person's first reaction; cite the memories that were activated.
2. System 2 (rational analysis): how the person would analyse and reason once calmed down.
3. Final Decision: two parts — `inner_consciousness` is the inner monologue of "what I am about to do/say" (the last layer of consciousness before the outward act, fusing emotional tone, core reasoning, and value orientation); `response_text` is what the person actually says or sends."""

    if options_text:
        prompt += f"""

## Behavioural Decision Options
Below are possible behavioural decisions different characters might make in this scenario. Pick the one that best fits you (as this character) and output the corresponding letter in the `decision_choice` field:

{options_text}"""

    return prompt


def print_result(result):
    """Print a structured result to console."""
    s1 = result.get('system_1_impulse', {})
    s2 = result.get('system_2_rational', {})

    print("-" * 40)
    print(f"  [System 1] Impulse:")
    print(f"    Thought: {s1.get('thought', 'N/A')}")
    print(f"    Emotion: {s1.get('emotion', 'N/A')}")
    print(f"    Citation: {s1.get('citation', 'N/A')}")
    print(f"\n  [System 2] Rationality:")
    print(f"    Analysis: {s2.get('analysis', 'N/A')}")
    print(f"    Plan: {s2.get('plan', 'N/A')}")
    print(f"\n  [Inner Consciousness]: {result.get('inner_consciousness', 'N/A')}")
    print(f"\n  [Final Decision]: {result.get('final_decision', 'N/A')}")
    print(f"\n  [Decision Choice]: {result.get('decision_choice', 'N/A')}")


def run_benchmark():
    import sys

    # Parse --random / --mem0 / --pre-annotated / --personadb flags
    use_random = "--random" in sys.argv
    use_mem0 = "--mem0" in sys.argv
    use_pre_annotated = "--pre-annotated" in sys.argv
    use_personadb = "--personadb" in sys.argv
    argv_filtered = [
        a
        for a in sys.argv[1:]
        if a
        not in (
            "--random",
            "--mem0",
            "--pre-annotated",
            "--personadb",
        )
    ]
    if sum(1 for x in (use_random, use_mem0, use_pre_annotated, use_personadb) if x) > 1:
        raise RuntimeError("Use at most one of --random, --mem0, --pre-annotated, --personadb.")

    data_dir = Path(__file__).resolve().parent.parent.parent / 'benchmark'
    results_dir = Path(__file__).resolve().parent.parent / 'results' / 'benchmark'
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    chars_path = resolve_characters_json(data_dir)
    if chars_path is None:
        raise RuntimeError(
            "Characters JSON not found. Set CHARACTERS_JSON_PATH or add "
            "benchmark/characters.json"
        )
    scen_path = resolve_scenarios_json(data_dir)
    if scen_path is None:
        raise RuntimeError(
            "Scenarios JSON not found. Set SCENARIOS_JSON_PATH or add "
            "benchmark/scenarios.json"
        )
    with open(chars_path, "r", encoding="utf-8-sig") as f:
        chars_data = json.load(f)["characters"]
    with open(scen_path, "r", encoding="utf-8-sig") as f:
        scenarios_data = json.load(f)["scenarios"]
        if isinstance(scenarios_data, dict):
            scenarios_data = [s for stage_list in scenarios_data.values() for s in stage_list]

    characters = [Character(c) for c in chars_data]
    scenarios = [Scenario(s) for s in scenarios_data]
    char_memory_index: dict[str, dict[str, dict]] = {}
    for char_data in chars_data:
        cid = char_data.get("id", "")
        episodic = char_data.get("episodic_memory_set", []) or []
        char_memory_index[cid] = {
            m.get("id", ""): m
            for m in episodic
            if isinstance(m, dict) and m.get("id")
        }

    # 2. Optional CLI filters: python main.py [--random] [scenario_id] [char_id]
    filter_scenario = argv_filtered[0] if len(argv_filtered) > 0 else None
    filter_char = argv_filtered[1] if len(argv_filtered) > 1 else None

    if filter_scenario:
        scenarios = [s for s in scenarios if s.id == filter_scenario]
        if not scenarios:
            print(f"Error: Scenario '{filter_scenario}' not found.")
            return
    if filter_char:
        characters = [c for c in characters if c.id == filter_char]
        if not characters:
            print(f"Error: Character '{filter_char}' not found.")
            return

    # 3. Check API Key
    if not API_CONFIG["api_key"]:
        raise RuntimeError(
            "API_KEY not set. Please edit .env file in the same folder as main_v2.py "
            "and fill in your API key."
        )
    if not (API_CONFIG.get("api_base") or "").strip():
        raise RuntimeError(
            "API_BASE not set. Set API_BASE (or ANNOTATE_API_BASE) in .env to your OpenAI-compatible endpoint; "
            "no default relay is used."
        )

    llm = OpenAICompatibleLLM(API_CONFIG)

    # Load GT annotations → build options pool for multiple-choice
    gt_path = resolve_gt_annotations_path(data_dir)
    options_pool = {}  # scenario_id -> [(char_id, final_decision), ...]
    if gt_path is not None:
        with open(gt_path, encoding="utf-8") as f:
            gt_data = json.load(f).get("annotations", {})
        for cid, stages in gt_data.items():
            for stage_anns in stages.values():
                for sid, ann in stage_anns.items():
                    fd = ann.get("final_decision", "")
                    if fd:
                        options_pool.setdefault(sid, []).append((cid, fd))
        try:
            gt_disp = str(gt_path.relative_to(data_dir))
        except ValueError:
            gt_disp = str(gt_path)
        print(f"Loaded GT options pool from {gt_disp}: {len(options_pool)} scenarios with choices.\n")
    else:
        print(
            "WARNING: GT annotations not found "
            "(set GT_ANNOTATIONS_PATH or add "
            "benchmark/ground_truth.json); "
            "decision_choice will be skipped.\n"
        )

    mc_path = resolve_mc_questions_path(data_dir)
    mc_by_pair: dict[tuple[str, str], dict] = {}
    if mc_path is not None:
        with open(mc_path, encoding="utf-8") as f:
            mc_doc = json.load(f)
        for q in mc_doc.get("questions", []):
            if not q.get("has_multiple_choice"):
                continue
            sid, cid = q.get("scenario_id"), q.get("character_id")
            if sid and cid:
                mc_by_pair[(sid, cid)] = q
        try:
            mc_disp = str(mc_path.relative_to(data_dir))
        except ValueError:
            mc_disp = str(mc_path)
        print(
            f"Loaded MC questions from {mc_disp}: "
            f"{len(mc_by_pair)} (scenario, character) pairs "
            "([strict] pairs not in MC will be skipped).\n"
        )
    else:
        print(
            "MC questions file not found (MC_QUESTIONS_PATH or "
            "benchmark/mcq.json); "
            "options come from GT pool only.\n"
        )

    # Build memory retriever
    mem0_memory = None
    personadb_cfg = None
    if use_random:
        memory_index = RandomMemoryRetriever()
        retrieval_mode = "random"
        act_mem_data = {}
        mem_content_index = {}
        print("Using RANDOM memory retrieval (control group).")
        for char in characters:
            memory_index.build(char)
        print("Random retriever ready.\n")
    elif use_mem0:
        try:
            from mem0_integration import (
                make_mem0_memory,
                mem0_infer_enabled,
                mem0_retrieve_for_prompt,
                mem0_search_rerank_enabled,
            )
        except ImportError as e:
            raise RuntimeError(
                "Mem0 mode requires: pip install mem0ai (see mem0_integration/README.md)"
            ) from e
        mem0_memory = make_mem0_memory()
        memory_index = None
        act_mem_data = {}
        mem_content_index = {}
        retrieval_mode = "mem0"
        print(
            "Using MEM0 memory retrieval (search per scenario). "
            f"infer={mem0_infer_enabled()} search_rerank={mem0_search_rerank_enabled()} "
            "(default: Mem0 intelligent pipeline; set MEM0_RAW_MODE=1 for vector-only; "
            "re-ingest after toggling). "
            "Ingest: python -m mem0_integration.ingest_characters.\n"
        )
    elif use_personadb:
        try:
            from personadb_integration import (
                make_personadb,
                ingest_character_episodes_personadb,
                personadb_retrieve_for_prompt,
            )
        except ImportError as e:
            raise RuntimeError(
                "PersonaDB mode requires sentence-transformers: "
                "pip install -r personadb_integration/requirements-personadb.txt"
            ) from e
        personadb_cfg = make_personadb()
        memory_index = None
        act_mem_data = {}
        mem_content_index = {}
        retrieval_mode = "personadb_condense" if personadb_cfg.condense else "personadb"
        condense_note = " (with L1 condensation; PERSONADB_CONDENSE=1)" if personadb_cfg.condense else ""
        print(
            f"Using PERSONA-DB retrieval{condense_note}; ingest via "
            "python -m personadb_integration.ingest_characters.\n"
        )
    elif use_pre_annotated:
        memory_index = create_memory_index()
        retrieval_mode = "embedding"
        print("Building memory embedding index (default: live cosine retrieval; no fallback)...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        build_workers = max(1, int(os.getenv("EMBEDDING_BUILD_WORKERS", "4")))
        build_errors: list[str] = []
        with ThreadPoolExecutor(max_workers=build_workers) as build_pool:
            future_to_char = {
                build_pool.submit(memory_index.build, char): char.id
                for char in characters
            }
            with tqdm(total=len(future_to_char), desc="EmbeddingBuild", unit="char") as pbar:
                for future in as_completed(future_to_char):
                    cid = future_to_char[future]
                    try:
                        future.result()
                        pbar.set_postfix_str(f"{cid} [OK]")
                    except Exception as e:
                        build_errors.append(f"{cid}: {e}")
                        pbar.set_postfix_str(f"{cid} [FAIL]")
                    pbar.update(1)
        if build_errors:
            raise RuntimeError(
                "Embedding build failed for some characters:\n- " + "\n- ".join(build_errors)
            )
        print("Memory index ready.\n")
        act_mem_data = {}
        mem_content_index = {}

    # 4. Print header
    try:
        chars_disp = str(chars_path.relative_to(data_dir))
    except ValueError:
        chars_disp = str(chars_path)
    try:
        scen_disp = str(scen_path.relative_to(data_dir))
    except ValueError:
        scen_disp = str(scen_path)
    print(f"{'='*60}")
    print(f"  Agent Personality Benchmark")
    print(f"  Model: {API_CONFIG['model_name']} | API: {API_CONFIG['api_base']}")
    print(f"  Characters file: {chars_disp}")
    print(f"  Scenarios file: {scen_disp}")
    print(f"  Scenarios: {len(scenarios)} | Characters: {len(characters)}")
    print(f"  Retrieval: {retrieval_mode}")
    print(f"{'='*60}\n")

    # 5. Prepare tasks: retrieve memories (single-threaded) then build prompts
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    MAX_WORKERS = int(os.getenv("BENCHMARK_WORKERS", "8"))
    MEM0_TOP_K = int(os.getenv("MEM0_TOP_K", "15"))
    PERSONADB_DEFAULT_TOP_K = int(os.getenv("PERSONADB_TOP_K", "15"))
    PERSONADB_SEARCH_LIMIT = int(os.getenv("PERSONADB_SEARCH_LIMIT", "100"))
    EMBEDDING_TOP_K = int(os.getenv("BENCHMARK_MEMORY_TOP_K", "15"))
    tasks = []  # list of (scenario, char, prompt, retrieved, correct_letter, option_map)
    skipped_completed = 0
    retrieval_cache_path = data_dir / "retrieval_cache" / f"{retrieval_mode}.json"
    retrieval_cache_payload = _load_retrieval_cache(retrieval_cache_path)
    retrieval_cache_items: dict[str, list[str]] = retrieval_cache_payload.get("items", {})
    retrieval_cache_dirty = False
    retrieval_cache_hit = 0
    retrieval_cache_miss = 0
    model_tag = _safe_filename_token(API_CONFIG.get("model_name", ""))
    output_path = results_dir / f"{model_tag}.json"
    checkpoint_meta, checkpoint_results = _load_benchmark_checkpoint(output_path)
    completed_case_keys: set[str] = set()
    if checkpoint_results:
        if checkpoint_meta.get("retrieval_mode") == retrieval_mode:
            completed_case_keys = {
                _case_key(str(r.get("scenario_id", "")), str(r.get("character_id", "")))
                for r in checkpoint_results
                if isinstance(r, dict)
            }
            if completed_case_keys:
                print(
                    f"Resume enabled: loaded {len(completed_case_keys)} completed cases "
                    f"from {output_path.name}."
                )
        else:
            print(
                "Resume skipped: existing result file retrieval_mode mismatch, "
                "will start a fresh run."
            )

    print("Preparing prompts (retrieving memories)...")
    skipped_no_mc = 0
    skipped_invalid_mc = 0
    for scenario in scenarios:
        for char in characters:
            if _case_key(scenario.id, char.id) in completed_case_keys:
                skipped_completed += 1
                continue
            mc_q = mc_by_pair.get((scenario.id, char.id))
            # Strict MC mode: if MC file is present, only evaluate listed (scenario, character) pairs.
            if mc_path is not None and mc_q is None:
                skipped_no_mc += 1
                continue
            # If pair exists in MC file but options are missing/empty, skip it as invalid.
            if mc_path is not None and not (mc_q and mc_q.get("options")):
                skipped_invalid_mc += 1
                continue

            context_text = scenario.context + scenario.trigger.get("message_content", "")
            if use_mem0:
                retrieval_top_k = MEM0_TOP_K
            elif use_personadb:
                retrieval_top_k = PERSONADB_DEFAULT_TOP_K
            else:
                retrieval_top_k = EMBEDDING_TOP_K
            cache_key = _retrieval_cache_key(
                retrieval_mode=retrieval_mode,
                char_id=char.id,
                scenario_id=scenario.id,
                stage=scenario.stage or "",
                top_k=retrieval_top_k,
                context_text=context_text,
            )
            cached_ids = retrieval_cache_items.get(cache_key) or []
            cached_memories = []
            if cached_ids:
                idx = char_memory_index.get(char.id, {})
                cached_memories = [idx[mid] for mid in cached_ids if mid in idx]
            if cached_ids and len(cached_memories) == len(cached_ids):
                retrieved = cached_memories
                retrieval_cache_hit += 1
                used_cache = True
            else:
                retrieval_cache_miss += 1
                retrieved = []
                used_cache = False

            if use_random:
                if not retrieved:
                    retrieved = char.retrieve_memories(
                        context_text,
                        memory_index=memory_index,
                        stage=scenario.stage,
                        top_k=EMBEDDING_TOP_K,
                    )
            elif use_mem0:
                if not retrieved:
                    retrieved = mem0_retrieve_for_prompt(
                        mem0_memory,
                        char_id=char.id,
                        query=context_text,
                        scenario_stage=scenario.stage,
                        top_k=MEM0_TOP_K,
                    )
            elif use_personadb:
                if not retrieved:
                    retrieved = personadb_retrieve_for_prompt(
                        personadb_cfg,
                        char_id=char.id,
                        query=context_text,
                        scenario_stage=scenario.stage,
                        top_k=PERSONADB_DEFAULT_TOP_K,
                        search_limit=PERSONADB_SEARCH_LIMIT,
                    )
            elif use_pre_annotated:
                if not retrieved:
                    cand_ids = activated_top5_memory_ids(
                        act_mem_data.get(char.id, {}).get(scenario.id, {})
                    )
                    char_idx = mem_content_index.get(char.id, {})
                    retrieved = [char_idx[mid] for mid in cand_ids if mid in char_idx]
            else:
                if not retrieved:
                    retrieved = char.retrieve_memories(
                        context_text,
                        memory_index=memory_index,
                        stage=scenario.stage,
                        top_k=EMBEDDING_TOP_K,
                    )

            if not used_cache:
                new_ids = [m.get("id", "") for m in retrieved if m.get("id")]
                retrieval_cache_items[cache_key] = new_ids
                retrieval_cache_dirty = True

            options_text = None
            correct_letter = None
            option_map: dict[str, str] = {}
            if mc_q and mc_q.get("options"):
                opts = sorted(mc_q["options"], key=lambda x: (x.get("letter") or ""))
                options_text = "\n".join(
                    f"{o.get('letter', '?')}. {o.get('final_decision', '')}" for o in opts
                )
                option_map = dict(mc_q.get("decision_option_map") or {})
                if not option_map:
                    for o in opts:
                        letter = o.get("letter")
                        if letter:
                            option_map[str(letter)] = str(o.get("character_id", ""))
                cl = (mc_q.get("correct_letter") or "").strip().upper()
                correct_letter = cl[:1] if cl else None
            else:
                options_entries = options_pool.get(scenario.id, [])
                if len(options_entries) >= 2:
                    rng = random.Random(f"{scenario.id}|{char.id}")
                    shuffled = options_entries.copy()
                    rng.shuffle(shuffled)
                    letters = "ABCDE"[: len(shuffled)]
                    options_text = "\n".join(
                        f"{letters[i]}. {shuffled[i][1]}" for i in range(len(shuffled))
                    )
                    for i, (cid, _) in enumerate(shuffled):
                        option_map[letters[i]] = cid
                        if cid == char.id:
                            correct_letter = letters[i]

            prompt = build_prompt(char, scenario, retrieved, options_text)
            tasks.append((scenario, char, prompt, retrieved, correct_letter, option_map))

    total = len(tasks)
    if mc_path is not None:
        print(
            f"  Strict MC filtering: skipped {skipped_no_mc} pairs not in MC file; "
            f"skipped {skipped_invalid_mc} pairs with invalid MC options."
        )
    print(
        f"  Retrieval cache: {retrieval_cache_hit} hit / {retrieval_cache_miss} miss "
        f"(mode={retrieval_mode})."
    )
    if skipped_completed:
        print(f"  Resume: skipped {skipped_completed} completed cases.")
    if retrieval_cache_dirty:
        retrieval_cache_payload["items"] = retrieval_cache_items
        retrieval_cache_payload["meta"] = {
            "retrieval_mode": retrieval_mode,
            "updated_at": int(time.time()),
        }
        _save_retrieval_cache(retrieval_cache_path, retrieval_cache_payload)
        print(f"  Retrieval cache saved: {retrieval_cache_path}")
    print(f"  {total} tasks ready. Parallel workers: {MAX_WORKERS}\n")

    # 6. Run benchmark in parallel
    import threading
    metadata = {
        "model_name": API_CONFIG.get("model_name", ""),
        "api_base": API_CONFIG.get("api_base", ""),
        "retrieval_mode": retrieval_mode,
        "scenario_count": len(scenarios),
        "character_count": len(characters),
        "scenarios": [
            {"id": s.id, "name": s.name}
            for s in scenarios
        ],
        "characters": [
            {"id": c.id, "name": c.name}
            for c in characters
        ],
    }
    if filter_scenario:
        metadata["filter_scenario"] = filter_scenario
    if filter_char:
        metadata["filter_character"] = filter_char

    def _save_checkpoint(path, meta, data):
        tmp = path.with_suffix(".tmp")
        payload = {
            "metadata": meta,
            "results": data,
        }
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def _run_single(llm, scenario, char, prompt):
        """Call the LLM and parse the response, returning a result dict."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                response_json = llm.generate(prompt)
                return json.loads(_sanitize_json(response_json))
            except json.JSONDecodeError as e:
                if attempt < MAX_RETRIES:
                    time.sleep(1)
                    continue
                return {"error": f"JSON parse failed after {MAX_RETRIES+1} attempts: {e}"}
            except Exception as e:
                return {"error": str(e)}

    results = [
        r
        for r in checkpoint_results
        if isinstance(r, dict)
        and _case_key(str(r.get("scenario_id", "")), str(r.get("character_id", "")))
        in completed_case_keys
    ]
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_meta = {}
        for scenario, char, prompt, retrieved, correct_letter, option_map in tasks:
            fut = pool.submit(_run_single, llm, scenario, char, prompt)
            future_to_meta[fut] = (scenario, char, retrieved, correct_letter, option_map)

        with tqdm(total=total, desc="Benchmark", unit="case") as pbar:
            for future in as_completed(future_to_meta):
                scenario, char, retrieved, correct_letter, option_map = future_to_meta[future]
                result = future.result()

                entry = {
                    "scenario_id": scenario.id,
                    "scenario_name": scenario.name,
                    "character_id": char.id,
                    "character_name": char.name,
                    "retrieval_mode": retrieval_mode,
                    "retrieved_memories": [m.get('id', '') for m in retrieved],
                    "decision_correct": correct_letter,
                    "decision_options": option_map,
                    "response": result,
                }
                with write_lock:
                    results.append(entry)
                    _save_checkpoint(output_path, metadata, results)

                status = "OK" if "error" not in result else "FAIL"
                pbar.set_postfix_str(f"{char.name}×{scenario.name[:15]} [{status}]")
                pbar.update(1)

    print(f"\n{'='*60}")
    print(f"Benchmark complete. {len(results)} results saved to: {output_path.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_benchmark()
