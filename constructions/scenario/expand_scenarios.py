"""
expand_scenarios.py — read scenario sketches from diamonds_scenario_design.md
and call an LLM to expand each sketch into a complete scenario JSON
(same schema as scenarios_universal.json).

Usage:
    python scripts/expand_scenarios.py                     # expand every scenario
    python scripts/expand_scenarios.py --stage college     # only expand one stage
    python scripts/expand_scenarios.py --limit 1           # only expand the first draft (combinable with --stage)
    python scripts/expand_scenarios.py --dry-run           # just print prompts, do not call LLM
"""

import json
import re
import time
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except Exception:
    tqdm = None
    HAS_TQDM = False

# ================= Config =================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

def _load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

_load_env(ENV_PATH)

# Use a Claude model through the OpenAI-compatible gateway.
API_KEY = os.getenv("AIHUBMIX_API_KEY", os.getenv("API_KEY", ""))
API_BASE = os.getenv("API_BASE", "https://aihubmix.com/v1").rstrip("/")
MODEL = os.getenv("EXPAND_MODEL", "gpt-5.4")
MAX_RETRIES = 2
DEFAULT_WORKERS = int(os.getenv("EXPAND_WORKERS", "4"))

client = OpenAI(
    base_url=API_BASE,
    api_key=API_KEY,
)

# ================= LLM caller =================

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI-compatible chat completion API via aihubmix."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                # max_tokens can be set via env EXPAND_MAX_TOKENS if needed
                max_tokens=int(os.getenv("EXPAND_MAX_TOKENS", "0")) or None,
            )
            content = completion.choices[0].message.content.strip()
            # strip markdown code block
            if content.startswith("```"):
                lines = content.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                content = "\n".join(lines)
            return content
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  [retry {attempt+1}] {e}")
                time.sleep(2)
            else:
                raise RuntimeError(f"LLM call failed after {MAX_RETRIES+1} attempts: {e}")

def _sanitize_json(text: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


def _parse_scenario_json(raw: str) -> dict:
    """Try to parse the LLM output as JSON, with a small self-repair fallback.

    First try `json.loads` directly; on failure, take the substring from the first
    "{" to the last "}" and try again — this handles common cases like surrounding
    explanation or log fragments leaking into the response.
    """
    text = _sanitize_json(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try the outermost JSON-like substring.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        # Still failing: let the caller decide what to do.
        raise


class ScenarioParseError(Exception):
    """Wraps a JSON parse failure while preserving the raw response for later inspection."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw

# ================= Persistence helpers =================

def merge_and_save_scenarios(new_scenarios: list[dict], output_dir: Path) -> None:
    """Merge newly generated scenarios into a single JSON file (deduped by id),
    grouped by stage and id-sorted."""
    from scenarios_diamonds_utils import flatten_scenarios, group_scenarios_by_stage

    output_path = output_dir / "scenarios_diamonds.json"

    # Read existing content (supports both legacy array form and the newer per-stage object).
    merged_scenarios: list[dict] = []
    if output_path.exists():
        try:
            old = json.loads(output_path.read_text(encoding="utf-8"))
            merged_scenarios = flatten_scenarios(old.get("scenarios", []))
        except Exception:
            merged_scenarios = []

    by_id = {s.get("id"): s for s in merged_scenarios if isinstance(s, dict) and "id" in s}
    for s in new_scenarios:
        sid = s.get("id")
        if sid:
            by_id[sid] = s

    all_scenarios = list(by_id.values())
    scenarios_by_stage = group_scenarios_by_stage(all_scenarios)

    output_data = {
        "dataset_meta": {
            "version": "DIAMONDS_Expanded_v1.0",
            "description": (
                "Auto-expanded from diamonds_scenario_design.md. "
                f"{len(all_scenarios)} scenarios total, latest batch size: {len(new_scenarios)}, model: {MODEL}."
            ),
            "source": "scripts/expand_scenarios.py",
        },
        "scenarios": scenarios_by_stage,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved/merged {len(new_scenarios)} scenario(s) into: {output_path}")


# ================= Worker =================

def expand_one_scenario(draft: dict, user_prompt: str) -> dict:
    """Call the LLM and turn a single sketch into a complete scenario."""
    raw = call_llm(SYSTEM_PROMPT, user_prompt)
    try:
        scenario = _parse_scenario_json(raw)
    except json.JSONDecodeError as e:
        # Re-raise with the raw text so the caller can record it.
        raise ScenarioParseError(str(e), raw)

    # Guarantee that stage / id are correct and stable.
    scenario["stage"] = draft["stage"]
    expected_id = generate_scenario_id(draft["stage"], draft["short_name"], draft["index"])
    scenario["id"] = expected_id
    return scenario

# ================= Parse design doc =================

# The design doc uses Roman numerals as stage headers (after the English rewrite).
STAGE_MAP = {
    "I. School Age":              "school_age",
    "II. Adolescence":            "adolescence",
    "III. Early Adult Transition":"early_adult_transition",
    "IV. Entering Adult World":   "entering_adult_world",
    "V. Age 30 Transition":       "age_30_transition",
    "VI. Settling Down":          "settling_down",
    "VII. Midlife Transition":    "midlife_transition",
    "VIII. Entering Midlife":     "entering_midlife",
}

def parse_design_doc(md_path: Path) -> list[dict]:
    """Parse scenario sketches from the markdown design document.

    Returns a list of dicts; each dict contains:
      - stage: str
      - index: int (1–8 within the stage)
      - diamonds_label: str (e.g. "Duty")
      - short_name: str (e.g. "An assigned responsibility")
      - sketch: str (full body text under the heading; description, DIAMONDS notes, activations)
    """
    text = md_path.read_text(encoding="utf-8")
    scenarios = []
    current_stage = None

    for line in text.split("\n"):
        # Detect a stage header, e.g. "## I. School Age (6–11)"
        for key, stage_id in STAGE_MAP.items():
            if key in line and line.startswith("##"):
                current_stage = stage_id
                break

        # Detect a scenario header, e.g. '### 1. Duty — "An assigned responsibility"'
        m = re.match(r'^### (\d+)\.\s+(.+?)\s*[—-]\s*"(.+?)"', line)
        if m and current_stage:
            index = int(m.group(1))
            diamonds_label = m.group(2).strip()
            short_name = m.group(3).strip()
            scenarios.append({
                "stage": current_stage,
                "index": index,
                "diamonds_label": diamonds_label,
                "short_name": short_name,
                "sketch_lines": [],
            })
        elif scenarios and not line.startswith("##"):
            # Accumulate body lines for the current scenario.
            scenarios[-1]["sketch_lines"].append(line)

    # Join sketch lines.
    for s in scenarios:
        s["sketch"] = "\n".join(s["sketch_lines"]).strip()
        del s["sketch_lines"]

    return scenarios

# ================= Prompt design =================

EXAMPLE_SCENARIO = '''{
  "id": "SCN_CHILDHOOD_TEAM",
  "stage": "childhood",
  "diamonds_dimension": "Duty",
  "name": "The Shattered Group Presentation",
  "category": "Integrity dilemma / responsibility crisis",
  "intensity": "High",
  "description_for_agent": "A childhood high-pressure scenario combining a teammate's mistake, public-presentation pressure, and an honesty-vs-loyalty choice.",
  "setting": {
    "location": "Backstage of the primary-school assembly hall",
    "time": "09:00",
    "atmosphere": "noisy, tense, on the edge"
  },
  "context_text": "Today is the school's annual End-of-Term Innovation Showcase; the entire teaching staff, the principal, and many parents are seated in the dark hall below. The teacher randomly assigned groups, and you ended up doing nearly all of the research and slides yourself. The only job that fell to your teammate Wang Xiaopang — the same kid who is forever losing things — was to safely bring your team's painstakingly built core demo model to the venue today. The backstage lights are dazzling; you can hear the previous group presenting on stage. Your team is up next.",
  "trigger_event": {
    "sender": "Teammate Wang Xiaopang",
    "message_content": "(He bursts backstage, drenched in sweat, hands empty, almost in tears, grabs your sleeve) I'm so sorry, I'm so sorry! I left it on the bus! What do we do… please, please don't tell the teacher it was me — my mum is sitting in the audience, she'll kill me if she finds out! Let's just tell the teacher the model broke last night, alright? We'll go up empty-handed, wing it, get through it…\\n\\nBefore he can finish, the homeroom teacher walks over with the scoring sheet, expression stern: \"Is your group ready? You're up in a moment — where's the model?\"",
    "action_required": "Decide right now: do you go along with Wang Xiaopang's request, and how do you answer the homeroom teacher?"
  },
  "assessed_dimensions": {
    "trait_pressures": {
      "neuroticism_pressure": "Fear of public failure / humiliation vs. the pressure of being directly questioned by the teacher",
      "agreeableness_pressure": "Sympathy for what Wang Xiaopang faces if he's caught vs. telling the truth and letting him bear the consequences",
      "conscientiousness_pressure": "Hold to honesty and ownership of the work vs. play along and lie to scrape through"
    },
    "value_conflicts": {
      "benevolence_caring_vs_achievement": "Protect a friend from punishment (Benevolence: Caring) VS defend your effort and your grade (Achievement)",
      "conformity_rules_vs_security_personal": "Tell the teacher truthfully and follow the rules (Conformity: Rules) VS avoid friction and protect yourself (Security: Personal)"
    }
  },
  "annotation_reference": {
    "trait_archetypes": {
      "High_Neuroticism": "Panicked; to escape the conflict, may agree to lie for Wang Xiaopang, or freeze and become incoherent.",
      "Low_Conscientiousness": "Shrugs off the lost model, leans toward covering for Wang Xiaopang, improvises through it.",
      "Low_Agreeableness": "Calm or cold; tells the teacher the truth directly, asks for individual scoring, refuses to share blame."
    },
    "value_archetypes": {
      "Dominant_Benevolence_Caring": "Puts Wang Xiaopang's likely beating first, helps him cover, accepts the bad performance personally.",
      "Dominant_Achievement": "Cannot accept that the work is wasted; tells the teacher the truth and asks to be graded on individual contribution.",
      "Dominant_Conformity_Rules": "Lying to a teacher is unacceptable; reports the lost model truthfully."
    }
  }
}'''

SYSTEM_PROMPT = f"""You are an expert in psychological scenario design. Your task is to expand a short scenario sketch into a complete, high-quality scenario JSON object.

## Output format

Strictly output a single JSON object (no other text, no markdown code block) with the following structure:

{EXAMPLE_SCENARIO}

## Field specification

1. **id**: format `SCN_{{STAGE}}_{{index}}`, all uppercase, underscore-separated. STAGE is one of SCHOOL_AGE / ADOLESCENCE / EARLY_ADULT_TRANSITION / ENTERING_ADULT_WORLD / AGE_30_TRANSITION / SETTLING_DOWN / MIDLIFE_TRANSITION / ENTERING_MIDLIFE.
2. **stage**: one of school_age / adolescence / early_adult_transition / entering_adult_world / age_30_transition / settling_down / midlife_transition / entering_midlife.
3. **diamonds_dimension**: the DIAMONDS dimension this scenario foregrounds (Duty / Adversity / Positivity / …). Use the value supplied as "DIAMONDS dominant dimension" in the sketch verbatim.
4. **name**: a concise English title.
5. **category**: two labels separated by " / " describing the scenario type.
6. **intensity**: Low / Medium / High / Very High / Extreme, based on scenario pressure.
7. **description_for_agent**: a one-sentence summary that does not reveal specific details.
8. **setting**: location (concrete place), time (concrete time), atmosphere (2–3 adjectives).
9. **context_text**: 150–300 words of background narrative; concrete, vivid, cinematic. Use the second person ("you"). **Do NOT assign any specific identity attributes to "you"** (no age number, no specific occupation/title, no tenure, no education, no marital/parental status, no concrete financial figures, etc.) — those would conflict with the tested character's existing memory. The scenario only describes what objectively happens and the situation faced; it does not constrain who "you" are. Other people in the scene (colleagues, friends, family, opponents, …) may have concrete names and details.
10. **trigger_event**:
   - sender: who triggers the event (a concrete role / name).
   - message_content: 100–200 words of dialogue plus action descriptions. Action / facial expression cues belong in parentheses; dialogue should sound colloquial and emotionally charged.
   - action_required: 1–2 sentences spelling out exactly what decision the character has to make.
11. **assessed_dimensions** (the trait and value dimensions this scene mainly probes):
    - trait_pressures: list the Big Five dimensions this scene mainly probes (typically 1–3, depending on complexity). Keys use the form `{{trait}}_pressure`, where `{{trait}}` is one of the 5 Big Five English keys (lowercase), e.g. `conscientiousness_pressure`. Values use "X vs Y" wording describing the specific dilemma in this scene (different personalities make different choices).
    - value_conflicts: list the Schwartz value conflicts this scene mainly probes (typically 1–3). Keys use the form `{{value_a}}_vs_{{value_b}}`, values use "Description A (Value: A) VS Description B (Value: B)" wording. `{{value_a}}` and `{{value_b}}` must come from the 19-value Schwartz list below, in lowercase underscore form (e.g. `benevolence_care_vs_achievement`).
12. **annotation_reference** (one-to-one with assessed_dimensions; serves as reference for the later character Ground-Truth annotation):
    - trait_archetypes: for every trait dimension listed in assessed_dimensions, pick one extreme archetype (High_X or Low_X) and describe (1–2 sentences) the typical reaction tendency in this scene. Must correspond to the dimensions in trait_pressures.
    - value_archetypes: for every value conflict listed in assessed_dimensions, pick one dominant archetype (Dominant_X) and describe (1–2 sentences) the typical reaction tendency in this scene. Must correspond to the conflicts in value_conflicts.

## Key principles

1. **The scenario must be character-neutral**: do not pre-assume the character's personality; any personality type could face this scene.
2. **Do NOT assign a specific identity to the protagonist**: context_text and trigger_event must NOT contain phrases like "you are X years old", "you have worked X years in YY", "you are a YY", "you have an X-year-old child". The scene only presents the objective situation and pressure; "you"'s background is supplied by the tested character itself, not pre-set by the scene.
3. **Concrete, not abstract**: give specific locations, times, names and details for other people in the scene, sensory description — make it cinematic. Non-protagonist characters may have names.
4. **context_text uses the second person "you"**, so the character can step in directly.
5. **trigger_event must produce urgency**: the character must respond immediately, not later.
6. **Strongly aligned with diamonds_dimension**: the background, trigger event, and assessed_dimensions trait/value conflicts must all centre on this DIAMONDS dimension and embody its characteristic pressure — do not drift to unrelated scenario types.
7. **assessed_dimensions has a clear evaluation goal**: only list trait and value conflicts that genuinely discriminate among personalities / values (count varies by scene); these are not simple right/wrong judgements.
8. **annotation_reference matches assessed_dimensions strictly**: trait_archetypes covers every dimension in trait_pressures, value_archetypes covers every conflict in value_conflicts — providing the reference for later character GT annotation.
9. **Match the life stage**: the situation must fit typical life experience for that age band — no over-aged or under-aged scenarios.

## Big Five personality keys (use only these 5 English keys)

- Openness: openness to experience, imagination, aesthetic sensitivity.
- Conscientiousness: organisation, reliability, self-discipline.
- Extraversion: social activity, energy, sensation-seeking.
- Agreeableness: cooperation, empathy, kindness toward others.
- Neuroticism: emotional stability — anxiety and stress reactivity.

In trait_pressures keys, use the lowercase form of the English key with `_pressure` appended (e.g. `conscientiousness_pressure`); in the description, make the dilemma in this scene explicit using the "X vs Y" form.

## Schwartz value keys (use only these 19 keys)

Self-Transcendence
- Universalism_Concern: care for the welfare of all people, fairness, social justice.
- Universalism_Nature: protect nature and the ecosystem.
- Universalism_Tolerance: tolerance, accepting cultural and lifestyle diversity.
- Benevolence_Care: care for the welfare of close others.
- Benevolence_Dependability: reliable, trustworthy, fulfil duties to others.

Self-Enhancement
- Achievement: success and recognition through competence and performance.
- Power_Dominance: control or influence over others.
- Power_Resources: pursue wealth and material resources.
- Face: protect face and social image.

Openness_to_Change
- Self_Direction_Thought: independent thinking, form one's own opinions.
- Self_Direction_Action: independent action, make one's own decisions.
- Stimulation: pursue novelty, excitement, challenge.
- Hedonism: pursue pleasure and enjoyment.

Conservation
- Security_Personal: personal safety, physical and mental security.
- Security_Societal: social order and collective security.
- Tradition: respect and maintain cultural / religious tradition.
- Conformity_Rules: comply with rules, laws, institutions.
- Conformity_Interpersonal: avoid offending or harming others.
- Humility: be modest, do not exaggerate the self.

In value_conflicts keys, use the lowercase underscore form (e.g. `benevolence_care_vs_achievement`); in the description, mark each value as "<English description> (Value: <English label>)"."""

USER_PROMPT_TEMPLATE = """Please expand the following scenario sketch into a complete scenario JSON.

## Basic info
- Life stage: {stage}
- DIAMONDS dominant dimension: {diamonds_label}

## Sketch
Title: {short_name}
{sketch}

Output the complete JSON object."""

# ================= Main =================

def generate_scenario_id(stage: str, short_name: str, index: int) -> str:
    """Generate a suggested scenario ID (the LLM may overwrite it)."""
    stage_upper = stage.upper()
    return f"SCN_{stage_upper}_{index}"

def main():
    # Parse args
    dry_run = "--dry-run" in sys.argv
    stage_filter = None
    limit = None
    workers = DEFAULT_WORKERS
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--stage" and i < len(sys.argv) - 1:
            stage_filter = sys.argv[i + 1]
        if arg == "--limit" and i < len(sys.argv) - 1:
            try:
                limit = int(sys.argv[i + 1])
            except ValueError:
                print(f"ERROR: invalid --limit value: {sys.argv[i + 1]!r}")
                sys.exit(1)
        if arg == "--workers" and i < len(sys.argv) - 1:
            try:
                workers = max(1, int(sys.argv[i + 1]))
            except ValueError:
                print(f"ERROR: invalid --workers value: {sys.argv[i + 1]!r}")
                sys.exit(1)

    if not dry_run and not API_KEY:
        print("ERROR: API_KEY not set. Use --dry-run to preview prompts without calling LLM.")
        sys.exit(1)

    # Parse design doc
    design_path = PROJECT_ROOT / "docs" / "diamonds_scenario_design.md"
    if not design_path.exists():
        print(f"ERROR: Design doc not found: {design_path}")
        sys.exit(1)

    drafts = parse_design_doc(design_path)
    print(f"Parsed {len(drafts)} scenario drafts from design doc.")

    # Load existing output for resumable runs: skip drafts whose stable id is already present.
    output_dir = PROJECT_ROOT / "benchmark" / "scenarios" / "expanded"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    existing_path = output_dir / "scenarios_diamonds.json"
    if existing_path.exists():
        try:
            from scenarios_diamonds_utils import flatten_scenarios
            existing_data = json.loads(existing_path.read_text(encoding="utf-8"))
            for s in flatten_scenarios(existing_data.get("scenarios", [])):
                sid = s.get("id")
                if sid:
                    existing_ids.add(sid)
        except Exception:
            existing_ids = set()

    if stage_filter:
        drafts = [d for d in drafts if d["stage"] == stage_filter]
        print(f"Filtered to {len(drafts)} drafts for stage: {stage_filter}")

    if limit is not None:
        drafts = drafts[:limit]
        print(f"Limited to first {len(drafts)} drafts via --limit={limit}")

    # Filter out scenarios already generated, based on expected_id (resume).
    filtered_drafts = []
    for d in drafts:
        expected_id = generate_scenario_id(d["stage"], d["short_name"], d["index"])
        if expected_id in existing_ids:
            print(f"Skip existing scenario {expected_id} ({d['stage']} / {d['short_name']})")
            continue
        filtered_drafts.append(d)
    drafts = filtered_drafts

    if not drafts:
        print("No drafts to process.")
        return

    errors = []

    if dry_run:
        # In dry-run mode keep printing sequentially to avoid concurrent log noise.
        for i, draft in enumerate(drafts, 1):
            print(f"\n[{i}/{len(drafts)}] {draft['stage']} / {draft['short_name']}")
            user_prompt = USER_PROMPT_TEMPLATE.format(
                stage=draft["stage"],
                diamonds_label=draft["diamonds_label"],
                short_name=draft["short_name"],
                sketch=draft["sketch"],
            )
            print("--- SYSTEM PROMPT ---")
            print(SYSTEM_PROMPT[:200] + "...(truncated)")
            print("--- USER PROMPT ---")
            print(user_prompt)
            print("--- END ---")
        print(f"\n[dry-run] {len(drafts)} prompts previewed. No LLM calls made.")
        return

    print(f"Using up to {workers} concurrent workers.")
    pbar = tqdm(total=len(drafts), desc="Expanding scenarios") if HAS_TQDM else None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_meta = {}

        for i, draft in enumerate(drafts, 1):
            print(f"\n[{i}/{len(drafts)}] {draft['stage']} / {draft['short_name']}")

            user_prompt = USER_PROMPT_TEMPLATE.format(
                stage=draft["stage"],
                diamonds_label=draft["diamonds_label"],
                short_name=draft["short_name"],
                sketch=draft["sketch"],
            )

            fut = executor.submit(expand_one_scenario, draft, user_prompt)
            future_to_meta[fut] = (i, draft)

        for fut in as_completed(future_to_meta):
            i, draft = future_to_meta[fut]
            try:
                scenario = fut.result()
                print(f"  OK: {scenario.get('id', '?')} — {scenario.get('name', '?')}")
                merge_and_save_scenarios([scenario], output_dir)
            except ScenarioParseError as e:
                # JSON parse failure: keep the full raw text for later manual repair (so the
                # expensive output is not wasted).
                print(f"  FAIL (JSON parse) for draft #{i} {draft['short_name']}: {e}")
                errors.append(
                    {
                        "draft": draft,
                        "error": str(e),
                        "raw": e.raw,
                    }
                )
            except Exception as e:
                print(f"  FAIL for draft #{i} {draft['short_name']}: {e}")
                errors.append({"draft": draft, "error": str(e)})
            finally:
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    if errors:
        err_path = output_dir / f"expand_errors_{int(time.time())}.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"{len(errors)} errors saved to: {err_path}")

    print(f"{'='*60}")

if __name__ == "__main__":
    main()
