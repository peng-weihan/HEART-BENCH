"""
generate_memories_optimized.py — layered generation: produce 900 episodic memories
in parallel across 4 ecologically-stratified memory layers.

Memory layers (ecological-validity design):
  A_core    (25%)  high-intensity core-trait memories      intensity 0.70-0.90
  B_related (35%)  mid-intensity related memories          intensity 0.45-0.70
  C_daily   (25%)  positive / neutral daily memories       intensity 0.20-0.50
  D_noise   (15%)  noise / unrelated memories              intensity 0.15-0.45

Usage:
    python scripts/generate_memories_optimized.py gen --char 04
    python scripts/generate_memories_optimized.py gen --char 04 --workers 5 --resume
    python scripts/generate_memories_optimized.py gen --char 04 --max-tasks 2  # small-scale test
    python scripts/generate_memories_optimized.py stats --char 04
"""

import argparse, json, os, re, sys, time, threading, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ==================== Config ====================
API_KEY = os.getenv("AIHUBMIX_API_KEY", "")
API_BASE = os.getenv("API_BASE", "https://aihubmix.com/v1")
MODEL = os.getenv("GEN_MODEL", "claude-sonnet-4-6")
MAX_RETRIES = 3
WORKERS = 6
MEMS_PER_CALL = 3         # 3 per call — leaves enough output budget for ~600 words per memory
MAX_OUTPUT_TOKENS = 10000

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "benchmark" / "characters"

client = OpenAI(base_url=API_BASE, api_key=API_KEY)
lock = threading.Lock()
progress = {"done": 0, "total": 0, "failed": 0}
used_registry = {"context": set(), "summary": set(), "psych": set()}
used_contexts_list = []  # ordered, fed back into the prompt for de-duplication

# ==================== Memory-layer config ====================

MEMORY_LAYERS = {
    "A_core": {
        "ratio": 0.25,
        "intensity_range": (0.70, 0.90),
        "desc": "core trunk, high intensity",
        "prompt_hint": "high-emotion-conflict scenes that directly express the character's core personality pattern",
    },
    "B_related": {
        "ratio": 0.35,
        "intensity_range": (0.45, 0.70),
        "desc": "related to trunk, mid intensity",
        "prompt_hint": "everyday-life scenes where the character's trait surfaces only subtly, not dominantly",
    },
    "C_daily": {
        "ratio": 0.25,
        "intensity_range": (0.20, 0.50),
        "desc": "daily positive / neutral",
        "prompt_hint": "ordinary joys, relaxed moments, small warm events, routine activities",
    },
    "D_noise": {
        "ratio": 0.15,
        "intensity_range": (0.15, 0.45),
        "desc": "noise / unrelated",
        "prompt_hint": "random life fragments with no clear link to the character's core personality trunk",
    },
}

# ==================== Character profiles ====================

CHAR_PROFILES = {
    "01": {
        "name": "Lin Wanqing",
        "archetype": "High Neuroticism",
        "big_five": "O=0.85 C=0.5 E=0.3 A=0.55 N=0.95",
        "mem_prefix": "MEM_01",
        "core_patterns": [
            "catastrophic attribution — a small setback spirals into total self-rejection",
            "over-reading micro-expressions and tone of voice → preemptive self-attack",
            "rumination — replaying the event afterwards and layering on darker readings",
            "somatisation — anxiety converts into stomach pain, insomnia, hand tremor",
            "abandonment schema — extreme sensitivity to any cue of separation",
        ],
        "stage_themes": {
            "childhood": ["shame spiral after being publicly scolded by a teacher", "fear hiding in the bedroom while parents argue", "self-rejection after being shunned by peers",
                          "catastrophic associations after a failed test", "separation anxiety from moving / changing schools", "the suffocating feeling of being misunderstood with no way to defend"],
            "adolescence": ["ruminating over a fractured friendship", "exam anxiety expressed as bodily symptoms", "over-reading every signal from a crush",
                            "preemptive self-attack when isolated", "surviving in the gap of a family conflict", "the sharp gap between ideal and reality"],
            "college": ["the inner drama of social occasions", "insecure attachment in a romantic relationship", "alternating flow and anxiety while creating alone",
                        "insomnia loop under finals pressure", "silent accumulation of roommate friction", "catastrophic predictions about the future"],
            "early_career": ["self-doubt spiral after manager critique", "social-energy drain among colleagues", "anxious insomnia before a deadline",
                             "rumination after a work mistake", "feeling abandoned when overlooked", "instability anxiety of freelancing"],
            "growth": ["push-pull cycle in intimate relationships", "self-rejection at a creative block", "over-reading social-media comments",
                       "hypochondriacal anxiety about body signals", "re-processing the family-of-origin relationship", "fear of a career pivot"],
            "recent": ["late-night rumination on old wounds", "rebuilding trust in a new relationship", "brief calm at a creative breakthrough",
                       "cyclical waves of health anxiety", "telling solitude apart from loneliness", "the struggle to accept imperfection"],
            "settled": ["seedlings of midlife crisis", "building security in a long-term relationship", "confidence from a maturing creative voice",
                        "attempts at reconciliation with parents", "anxiety from signs of physical aging", "loss and relief as the social circle shrinks"],
            "midlife_transition": ["anxiety about the career ceiling", "projection anxiety onto the children's education", "fear of separation as parents grow old",
                                   "confusion when re-evaluating life's worth", "catastrophising on the back of a health issue", "rumination over an old relationship that broke"],
            "midlife_settled": ["new strategies for living alongside the anxiety", "awareness of a creative legacy", "deep trust in intimate relationships",
                                "looking back on the younger self with compassion", "accepting uncertainty", "warmth in passing experience on"],
        },
    },
    "04": {
        "name": "Character D — Low Conscientiousness",
        "archetype": "Low Conscientiousness",
        "big_five": "O=0.8 C=0.25 E=0.6 A=0.65 N=0.45",
        "mem_prefix": "MEM_04",
        "core_patterns": [
            "procrastinates until the last minute, then bursts to finish",
            "strong resistance to plans and process",
            "obsessing over the present interest at the cost of forgetting commitments",
            "rationalising loose behaviour with last-minute success",
            "stumbling into a creative breakthrough out of chaos",
        ],
        "stage_themes": {
            "childhood": ["homework procrastination → fight with parents", "distracted in class by something more interesting", "perpetually messy backpack and bedroom",
                          "cramming the night before a test", "three-minute enthusiasm at every after-school class", "marathon-finishing summer homework on the last day"],
            "adolescence": ["all-nighter before exams but in the wrong direction", "group projects only delivered after an ultimatum", "joining many clubs and quitting all of them",
                            "improvising on stage and accidentally winning", "only the first three pages of every notebook are serious", "still browsing unrelated content right before the mock exam"],
            "college": ["interest-only course selection causes schedule conflicts", "sleeps through an exam after a hackathon", "got caught reusing an old report template",
                        "thesis topic changed three times", "advisor emails always answered a week late", "late to an interview but stunning once there"],
            "early_career": ["jumps into the work without reading the docs and has to redo it", "forgets to update progress at standups", "strong code, no comments",
                             "client demo prepared an hour beforehand", "weekly report submitted late", "low quarterly targets, breakthrough innovation"],
            "growth": ["mentors a junior while own tasks slip", "cross-team comms slow → blocking", "same issue post-mortemed three times",
                       "calendar app abandoned within a week", "no accommodation booked for a holiday trip", "promotion review: capable but inconsistent"],
            "recent": ["all projects slipping in parallel", "remote work makes it even looser", "gym membership: one visit a month",
                       "another failed attempt at quitting smoking", "almost let the visa lapse", "back-to-back all-nighters → body alarms"],
            "settled": ["family responsibilities slipping", "missed the parent-teacher meeting", "renovation drags because of no follow-through",
                        "team grows large enough to require process anyway", "personal projects shelved, tech going stale", "marathon training: dropped out twice"],
            "midlife_transition": ["often late to school pick-up", "no real family financial plan", "the pivot is still stuck in the 'thinking' phase",
                                   "slow to act on parents' health issues", "watching a young, efficient colleague triggers reflection", "owning that 'looseness' is a real limitation"],
            "midlife_settled": ["selectively takes projects", "teams up with people who tolerate informality", "publishes a long essay about creativity and freedom",
                                "honest with juniors about own faults", "health forces a baseline routine", "annual goals capped at three for slack"],
        },
    },
    "08": {
        "name": "Character H — Low Agreeableness",
        "archetype": "Low Agreeableness",
        "big_five": "O=0.65 C=0.85 E=0.45 A=0.25 N=0.35",
        "mem_prefix": "MEM_08",
        "core_patterns": [
            "calls out errors in public regardless of social face",
            "treats softening = information loss; would rather have conflict than ambiguity",
            "corrects inefficient collaboration directly, ignoring feelings",
            "default-assumes competition is zero-sum, prioritises protecting outcomes",
            "interrogates vague statements until a verifiable answer is given",
        ],
        "stage_themes": {
            "childhood": ["points out the teacher's mistake to their face", "refuses to be grouped with weaker students", "tells a classmate their drawing is bad",
                          "tells a relative the dish they cooked is bad", "vetoes the weakest candidate", "reports a classmate who is slacking"],
            "adolescence": ["pushes opponents in debate until they fumble", "rejects an admirer flatly", "calls out free-riders publicly",
                            "argues with the teacher over an answer and won't back down", "exposes vote-canvassing tactics", "corrects a foreign teacher's grammar"],
            "college": ["points out the professor cited the wrong number", "shoots down an upperclassman's plan, room goes cold", "gives a real low score in peer review",
                        "refuses to cover a roommate's class for them", "throws back a low-quality interviewer question", "questions the logic of the defence committee's question"],
            "early_career": ["points straight at the boss's logical hole", "refuses vague phrasing", "writes the project's risks honestly and gets called in",
                             "long critical code review", "calls lunch gossip a waste of time", "refuses to clock in for someone else"],
            "growth": ["questions a connection-hire's qualifications", "directly recommends an underperformer be let go", "raises an objection at an industry conference",
                       "tells the team that the offsite doesn't help the work", "questions the CEO's optimistic forecast", "reports falsified data immediately"],
            "recent": ["insists on disclosing the real numbers", "refuses to take part in a sham investigation", "standards too high → can't retain new hires",
                       "calls the police on noisy neighbours", "tells a classmate they've put on weight; awkwardness ensues", "refuses to embellish a recommendation letter"],
            "settled": ["opposes inefficient process and gets noticed by leadership", "holds the technical line, won't yield", "drives removal of performance bias",
                        "strict at home, friction follows", "after layered friend-pruning, only blunt people are left", "tries 'conclusion first, then evidence'"],
            "midlife_transition": ["disagrees with the CFO over the scope of risk disclosure", "vetoes a founder's connection-based recommendation", "corners the opponent in a forum debate",
                                   "refuses to budge in negotiation and gets a better outcome", "argues inheritance strictly by law → family conflict", "points out a junior's mistake directly"],
            "midlife_settled": ["calls out industry problems in a keynote, big discussion follows", "kills wasteful projects and makes enemies", "pushes for institutional protection of speaking truth",
                                "honest peer evaluation of the successor at retirement handover", "looks back: blunt warnings averted disasters", "realises that delivery, too, can be optimised"],
        },
    },
}

# Per-character batch definitions.
# The 6th tuple element is the English stage label that the LLM must emit
# verbatim into the dataset.
BATCH_DEFS = {
    "01": [
        ("b01", 101, 136, 6, 7, "childhood"), ("b02", 137, 172, 8, 9, "childhood"),
        ("b03", 173, 208, 10, 11, "childhood"), ("b04", 209, 258, 12, 13, "adolescence"),
        ("b05", 259, 308, 14, 15, "adolescence"), ("b06", 309, 357, 16, 17, "adolescence"),
        ("b07", 358, 404, 18, 19, "college"), ("b08", 405, 450, 20, 21, "college"),
        ("b09", 451, 496, 22, 22, "college"), ("b10", 497, 547, 23, 24, "early_career"),
        ("b11", 548, 598, 25, 26, "early_career"), ("b12", 599, 649, 27, 28, "growth"),
        ("b13", 650, 691, 29, 30, "growth"), ("b14", 692, 733, 31, 32, "growth"),
        ("b15", 734, 773, 33, 33, "recent"), ("b16", 774, 816, 34, 35, "recent"),
        ("b17", 817, 860, 36, 37, "settled"), ("b18", 861, 903, 38, 40, "settled"),
        ("b19", 904, 963, 41, 45, "midlife_transition"), ("b20", 964, 1000, 46, 50, "midlife_settled"),
    ],
    "04": [
        ("b01", 101, 136, 6, 7, "childhood"), ("b02", 137, 172, 8, 9, "childhood"),
        ("b03", 173, 208, 10, 11, "childhood"), ("b04", 209, 258, 12, 13, "adolescence"),
        ("b05", 259, 308, 14, 15, "adolescence"), ("b06", 309, 357, 16, 17, "adolescence"),
        ("b07", 358, 404, 18, 19, "college"), ("b08", 405, 450, 20, 21, "college"),
        ("b09", 451, 496, 22, 22, "college"), ("b10", 497, 547, 23, 24, "early_career"),
        ("b11", 548, 598, 25, 26, "early_career"), ("b12", 599, 649, 27, 28, "growth"),
        ("b13", 650, 691, 29, 30, "growth"), ("b14", 692, 733, 31, 32, "growth"),
        ("b15", 734, 773, 33, 33, "recent"), ("b16", 774, 816, 34, 35, "recent"),
        ("b17", 817, 860, 36, 37, "settled"), ("b18", 861, 903, 38, 40, "settled"),
        ("b19", 904, 963, 41, 45, "midlife_transition"), ("b20", 964, 1000, 46, 50, "midlife_settled"),
    ],
    "08": [
        ("b01", 101, 136, 7, 8, "childhood"), ("b02", 137, 172, 9, 10, "childhood"),
        ("b03", 173, 208, 11, 12, "adolescence"), ("b04", 209, 258, 13, 14, "adolescence"),
        ("b05", 259, 308, 15, 16, "adolescence"), ("b06", 309, 357, 17, 18, "college"),
        ("b07", 358, 404, 19, 20, "college"), ("b08", 405, 450, 21, 22, "college"),
        ("b09", 451, 496, 23, 24, "early_career"), ("b10", 497, 547, 25, 27, "early_career"),
        ("b11", 548, 598, 28, 30, "growth"), ("b12", 599, 649, 31, 33, "growth"),
        ("b13", 650, 691, 34, 36, "recent"), ("b14", 692, 733, 37, 39, "recent"),
        ("b15", 734, 773, 40, 42, "settled"), ("b16", 774, 816, 43, 44, "settled"),
        ("b17", 817, 860, 45, 46, "midlife_transition"), ("b18", 861, 903, 47, 48, "midlife_transition"),
        ("b19", 904, 963, 49, 50, "midlife_settled"), ("b20", 964, 1000, 50, 50, "midlife_settled"),
    ],
}

# ==================== Compact prompt builders ====================

def build_system_prompt(char_key: str, layer: str = "A_core") -> str:
    """Build the system prompt for a given memory layer."""
    p = CHAR_PROFILES[char_key]
    patterns = "; ".join(p["core_patterns"])
    layer_cfg = MEMORY_LAYERS[layer]
    lo, hi = layer_cfg["intensity_range"]

    # Layer-specialised role / scene guidance.
    if layer == "A_core":
        role_guidance = f"""## Character
{p['name']} ({p['archetype']}) | {p['big_five']}
Core patterns: {patterns}

## Scene requirement
This is a **high-emotional-intensity core memory** that directly expresses the character's core personality pattern.
The scene should involve significant psychological conflict, a turning point, or emotional impact. emotion_intensity should be between {lo} and {hi}."""

    elif layer == "B_related":
        role_guidance = f"""## Character
{p['name']} ({p['archetype']}) | {p['big_five']}
Core patterns: {patterns}

## Scene requirement
This is an **everyday-life memory** in which the character's traits surface only subtly — they are NOT the dominant conflict of the scene.
For example: a small hesitation while shopping, a faint discomfort during chit-chat with a colleague, drifting off while cooking — the personality is the background tone, not the lead.
emotion_intensity should be between {lo} and {hi} (mid-low).
Do NOT write an intense conflict or emotional breakdown."""

    elif layer == "C_daily":
        role_guidance = f"""## Character
{p['name']}, an ordinary person.

## Scene requirement
This is a **calm, positive or neutral everyday memory**. This person also has happy, relaxed, satisfied moments.
Examples: sleeping in on a weekend, eating something delicious, finishing a small goal, a pleasant chat with a friend,
a walk for the view, the comfort after tidying a room, the small thrill of learning a new skill, grocery shopping, a movie.
emotion_intensity should be between {lo} and {hi} (low).
emotion_primary must be a positive or neutral emotion word (satisfied, calm, pleased, curious, comfortable, etc.); do NOT use anxiety / fear / anger.
This memory involves no psychological conflict or personality issue at all — just an ordinary person's ordinary joy."""

    else:  # D_noise
        role_guidance = f"""## Character
{p['name']} | {p['big_five']}

## Scene requirement
This is a **plain everyday memory** with no clear link to the character's core personality trunk.
The scene is a fully random life fragment: waiting for a bus, fixing a computer, queuing for a check-up, sorting after a move, reading a news item,
grocery shopping, picking up a parcel, paying utility bills, repairing an appliance, queuing for a number to be called.
emotion_intensity should be between {lo} and {hi} (low).
The memory does NOT need to express any specific personality trait — it is just an ordinary person's ordinary memory."""

    return f"""You are an expert in psychological narrative writing, producing episodic memories for Big-Five-based fictional characters.

{role_guidance}

## Output format
Output using ===-delimited blocks; each memory follows this structure (do NOT use JSON):

===MEM_ID===
timeline: stage(age X)
context: 15-30 word scene description
content_summary: 30-60 word event summary
triggers: ["word1","word2","word3","word4"]
psych_conclusion: 40-80 word psychological conclusion
behavior_policy: 30-60 word behaviour rule
emotion_primary: primary emotion word
emotion_secondary: secondary emotion word
emotion_intensity: 0.XX
relevance_tags: ["tag1","tag2","tag3","tag4"]
---content_full---
(Write a 600-word first-person narrative here, including: (1) sensory detail, (2) reconstructed dialogue, (3) inner monologue, (4) bodily reaction, (5) aftermath.)

## Hard constraints
- content_full MUST reach 600 words (excluding punctuation), about 750 total characters. This is the top-priority hard requirement; stopping at 400 words is not enough — keep expanding the detail until you reach 600 words.
- Writing strategy: open with sensory detail (~50 words) → dialogue scene (~100 words) → event development (~150 words) → inner monologue (~150 words) → bodily reaction (~50 words) → aftermath (~100 words).
- Each memory's `context` must be a completely distinct, concrete micro-scene (specific time, place, people, event); never a variation of the same archetype.
- Vary your narrative angle. No template-like openings. Do NOT lead or pad with overused phrases like "the desk lamp", "filled the air", "row upon row", "a wave of …".
- The `context` must cover a wide range of life domains: family, school, social, alone, outdoors, holidays, travel, shopping, doctor visits, contests, accidents, etc. — do not concentrate on one or two domains."""


def build_user_prompt(char_key: str, stage: str, age_start: int, age_end: int,
                      count: int, themes: list[str], start_id: int,
                      used_contexts: list[str] | None = None,
                      layer: str = "A_core") -> str:
    age_desc = f"age {age_start}" if age_start == age_end else f"age {age_start}-{age_end}"
    prefix = CHAR_PROFILES[char_key]["mem_prefix"]
    ids_list = ", ".join(f"{prefix}_{start_id+i:04d}" for i in range(count))
    layer_cfg = MEMORY_LAYERS[layer]

    # Layer-specialised scene guidance.
    if layer == "A_core":
        scene_guide = (
            f"The scene must directly trigger and express the character's core personality pattern.\n"
            f"emotion_intensity between {layer_cfg['intensity_range'][0]} and {layer_cfg['intensity_range'][1]}."
        )
    elif layer == "B_related":
        scene_guide = (
            f"The scene is everyday (grocery run, commute, chit-chat, household chores, browsing shops, etc.) but the character's trait surfaces faintly within it.\n"
            f"It is NOT an intense conflict — just a small ripple in the daily routine. emotion_intensity between "
            f"{layer_cfg['intensity_range'][0]} and {layer_cfg['intensity_range'][1]}."
        )
    elif layer == "C_daily":
        scene_guide = (
            f"The scene is calm or pleasant: a walk, a meal with friends, a movie, gaming, baking, playing with a pet, music, exercise, etc.\n"
            f"This is a relaxed, satisfied, warm moment in the character's life. emotion_intensity between "
            f"{layer_cfg['intensity_range'][0]} and {layer_cfg['intensity_range'][1]}.\n"
            f"emotion_primary MUST be a positive or neutral word (satisfied, calm, pleased, curious, comfortable, warm, etc.)."
        )
    else:  # D_noise
        scene_guide = (
            f"The scene is fully ordinary life trivia: picking up a parcel, waiting for the lift, taking the metro, paying a phone bill, getting shoes repaired, copying a key, etc.\n"
            f"No personality trait needs to surface — just plain everyday flow. emotion_intensity between "
            f"{layer_cfg['intensity_range'][0]} and {layer_cfg['intensity_range'][1]}."
        )

    prompt = (
        f"Generate {count} memories from \"{stage}\" ({age_desc}), IDs: {ids_list}\n"
        f"[Memory type: {layer_cfg['desc']}]\n"
        f"{scene_guide}\n"
        f"Each memory's context MUST be a completely distinct, concrete micro-scene of life.\n"
        f"Cover a variety of life domains; do not keep riffing on the same one or two scenes.\n"
        f"Each content_full MUST reach 600 words (excluding punctuation); 400 is absolutely not enough — keep expanding!\n"
        f"context / summary / psych_conclusion must be unique within the batch. Use the ===-delimited output format."
    )

    # Pass in the list of already-used contexts so the model can avoid them.
    if used_contexts:
        recent = used_contexts[-15:]  # last 15 contexts
        ctx_list = ", ".join(f'"{c}"' for c in recent)
        prompt += f"\n\nThe following contexts have already been used — avoid them entirely: {ctx_list}"

    return prompt


# ==================== LLM call ====================

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int) -> tuple[str, dict]:
    """Call the LLM. Returns (response_text, usage_dict)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.88,
                max_tokens=max_tokens,
            )
            text = r.choices[0].message.content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(l for l in lines if not l.startswith("```"))
            usage = {}
            if r.usage:
                usage = {
                    "prompt_tokens": r.usage.prompt_tokens,
                    "completion_tokens": r.usage.completion_tokens,
                    "total_tokens": r.usage.total_tokens,
                }
            return text, usage
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt + 1))
                log(f"  [retry {attempt+1}] {e}")
            else:
                raise


def parse_text_blocks(raw: str, char_num: str) -> list[dict]:
    """Parse the ===MEM_XX_YYYY=== delimited output into a list of dicts."""
    import ast
    blocks = re.split(rf"===MEM_{char_num}_(\d{{3,4}})===\s*\n", raw)
    mems = []
    for i in range(1, len(blocks) - 1, 2):
        mem_id_str = blocks[i]
        block_text = blocks[i + 1].strip()
        parts = block_text.split("---content_full---", 1)
        if len(parts) != 2:
            continue
        meta_text = parts[0].strip()
        content_full = parts[1].strip()
        meta = {}
        for line in meta_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            idx = line.find(":")
            if idx == -1:
                continue
            meta[line[:idx].strip()] = line[idx + 1:].strip()
        # Parse triggers and relevance_tags
        try:
            triggers = ast.literal_eval(meta.get("triggers", "[]"))
        except Exception:
            triggers = [t.strip().strip("\"'") for t in meta.get("triggers", "").strip("[]").split(",") if t.strip()]
        try:
            relevance_tags = ast.literal_eval(meta.get("relevance_tags", "[]"))
        except Exception:
            relevance_tags = [t.strip().strip("\"'") for t in meta.get("relevance_tags", "").strip("[]").split(",") if t.strip()]
        try:
            intensity = float(meta.get("emotion_intensity", "0.72"))
        except (ValueError, TypeError):
            intensity = 0.72

        mems.append({
            "id": f"MEM_{char_num}_{mem_id_str}",
            "timeline": meta.get("timeline", ""),
            "context": meta.get("context", ""),
            "content_summary": meta.get("content_summary", ""),
            "content_full": content_full,
            "triggers": triggers,
            "psych_conclusion": meta.get("psych_conclusion", ""),
            "behavior_policy": meta.get("behavior_policy", ""),
            "emotion_primary": meta.get("emotion_primary", ""),
            "emotion_secondary": meta.get("emotion_secondary", ""),
            "emotion_intensity": intensity,
            "relevance_tags": relevance_tags,
        })
    return mems


# ==================== De-duplication ====================

def normalize_text_key(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def detect_duplicates(mems: list[dict]) -> list[str]:
    errs = []
    local_ctx, local_sum, local_psy = set(), set(), set()
    with lock:
        g_ctx = set(used_registry["context"])
        g_sum = set(used_registry["summary"])
        g_psy = set(used_registry["psych"])
    for idx, mem in enumerate(mems, 1):
        ck = normalize_text_key(mem.get("context", ""))
        sk = normalize_text_key(mem.get("content_summary", ""))
        pk = normalize_text_key(mem.get("psych_conclusion", ""))
        if not ck or not sk or not pk:
            errs.append(f"item{idx}:empty_field")
            continue
        if ck in local_ctx or sk in local_sum or pk in local_psy:
            errs.append(f"item{idx}:dup_in_batch")
        if ck in g_ctx or sk in g_sum or pk in g_psy:
            errs.append(f"item{idx}:dup_with_global")
        local_ctx.add(ck); local_sum.add(sk); local_psy.add(pk)
    return errs


def register_uniques(mems: list[dict]) -> None:
    with lock:
        for mem in mems:
            used_registry["context"].add(normalize_text_key(mem.get("context", "")))
            used_registry["summary"].add(normalize_text_key(mem.get("content_summary", "")))
            used_registry["psych"].add(normalize_text_key(mem.get("psych_conclusion", "")))
            ctx = mem.get("context", "").strip()
            if ctx:
                used_contexts_list.append(ctx)

def parse_blocks_for_registry(raw_text: str, char_num: str) -> list[dict]:
    blocks = re.split(rf"===MEM_{char_num}_\d{{3,4}}===(?:\r?\n)", raw_text)
    mems = []
    for blk in blocks[1:]:
        parts = blk.split("---content_full---", 1)
        if len(parts) != 2:
            continue
        meta = {}
        for line in parts[0].splitlines():
            if ":" not in line:
                continue
            idx = line.find(":")
            meta[line[:idx].strip()] = line[idx+1:].strip()
        mems.append({
            "context": meta.get("context", ""),
            "content_summary": meta.get("content_summary", ""),
            "psych_conclusion": meta.get("psych_conclusion", ""),
        })
    return mems


# ==================== Formatting ====================
# stage_label_*() returns the English stage tokens used in the dataset.
# format_memory_block() emits "timeline: {stage}(age {age})".

def pick_age(mem_id, start_id, end_id, age_start, age_end):
    if age_start == age_end:
        return age_start
    span = end_id - start_id
    if span == 0:
        return age_start
    return int(round(age_start + (age_end - age_start) * (mem_id - start_id) / span))


def stage_label_01(age):
    if age <= 11: return "childhood"
    if age <= 17: return "adolescence"
    if age <= 22: return "college"
    if age <= 26: return "early_career"
    if age <= 32: return "growth"
    if age <= 35: return "recent"
    if age <= 40: return "settled"
    if age <= 45: return "midlife_transition"
    return "midlife_settled"


stage_label_04 = stage_label_01  # same mapping


def stage_label_08(age):
    if age <= 12: return "childhood"
    if age <= 18: return "adolescence"
    if age <= 24: return "college"
    if age <= 30: return "early_career"
    if age <= 36: return "growth"
    if age <= 42: return "recent"
    if age <= 46: return "settled"
    if age <= 48: return "midlife_transition"
    return "midlife_settled"


STAGE_LABEL_FN = {"01": stage_label_01, "04": stage_label_04, "08": stage_label_08}


def format_memory_block(mem: dict, mem_id: int, age: int, stage: str, char_num: str) -> str:
    triggers = mem.get("triggers", [])
    if isinstance(triggers, str):
        triggers = [t.strip() for t in triggers.split(",")]
    tags = mem.get("relevance_tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    try:
        intensity = float(mem.get("emotion_intensity", 0.72))
    except (ValueError, TypeError):
        intensity = 0.72

    return (
        f"===MEM_{char_num}_{mem_id:04d}===\n"
        f"timeline: {stage}(age {age})\n"
        f"context: {mem.get('context','')}\n"
        f"content_summary: {mem.get('content_summary','')}\n"
        f"triggers: {triggers}\n"
        f"psych_conclusion: {mem.get('psych_conclusion','')}\n"
        f"behavior_policy: {mem.get('behavior_policy','')}\n"
        f"emotion_primary: {mem.get('emotion_primary','')}\n"
        f"emotion_secondary: {mem.get('emotion_secondary','')}\n"
        f"emotion_intensity: {intensity:.2f}\n"
        f"relevance_tags: {tags}\n"
        f"---content_full---\n"
        f"{mem.get('content_full','')}\n\n"
    )


# ==================== Task construction ====================

def build_tasks(char_key: str, mems_per_call: int):
    """Split the 900-memory plan into sub-tasks, allocated by layer ratio."""
    profile = CHAR_PROFILES[char_key]
    batches = BATCH_DEFS[char_key]
    tasks = []

    # For each batch, split the memory count across layers proportionally.
    for batch_key, mem_start, mem_end, age_start, age_end, stage in batches:
        batch_total = mem_end - mem_start + 1
        # Allocate memory counts per layer.
        layer_counts = {}
        assigned = 0
        layer_names = list(MEMORY_LAYERS.keys())
        for i, layer_name in enumerate(layer_names):
            if i == len(layer_names) - 1:
                # Last layer absorbs the remainder.
                layer_counts[layer_name] = batch_total - assigned
            else:
                cnt = round(batch_total * MEMORY_LAYERS[layer_name]["ratio"])
                layer_counts[layer_name] = cnt
                assigned += cnt

        # Generate sub-tasks for each layer.
        pos = mem_start
        for layer_name in layer_names:
            count = layer_counts[layer_name]
            if count <= 0:
                continue
            remaining = count
            while remaining > 0:
                chunk = min(mems_per_call, remaining)
                all_themes = profile["stage_themes"].get(stage, ["everyday life"])
                rng = random.Random(pos * 131 + chunk * 17)
                pool = list(all_themes)
                rng.shuffle(pool)
                selected = pool[:max(8, chunk)]

                tasks.append({
                    "batch_key": batch_key,
                    "stage": stage,
                    "age_start": age_start,
                    "age_end": age_end,
                    "mem_start": pos,
                    "mem_end_batch": mem_end,
                    "count": chunk,
                    "themes": selected,
                    "layer": layer_name,
                })
                pos += chunk
                remaining -= chunk

    return tasks


# ==================== Parallel execution ====================

def process_task(task: dict, char_key: str, system_prompts: dict,
                 max_tokens: int) -> tuple[int, str, list[str], dict]:
    """Process one sub-task. Returns (mem_start, batch_text, errors, usage)."""
    char_num = char_key.zfill(2)
    start_id = task["mem_start"]
    count = task["count"]
    stage_fn = STAGE_LABEL_FN[char_key]
    layer = task.get("layer", "A_core")
    system_prompt = system_prompts[layer]

    # Snapshot the currently used contexts.
    with lock:
        recent_contexts = list(used_contexts_list)

    prompt = build_user_prompt(
        char_key, task["stage"], task["age_start"], task["age_end"],
        count, task["themes"], start_id,
        used_contexts=recent_contexts,
        layer=layer
    )

    errors = []
    total_usage = {}
    parsed_text = ""

    for attempt in range(1, 4):
        try:
            raw, usage = call_llm(system_prompt, prompt, max_tokens=max_tokens)
            total_usage = usage
            mems = parse_text_blocks(raw, char_num)
            if len(mems) < count:
                raise ValueError(f"insufficient:{len(mems)}/{count}")
            mems = mems[:count]
            dup_errs = detect_duplicates(mems)
            if dup_errs:
                raise ValueError("dup:" + ",".join(dup_errs[:3]))
            register_uniques(mems)

            blocks = []
            for i, mem in enumerate(mems):
                mid = start_id + i
                age = pick_age(mid, task["mem_start"], task["mem_end_batch"],
                               task["age_start"], task["age_end"])
                stg = stage_fn(age)
                blocks.append(format_memory_block(mem, mid, age, stg, char_num))
            parsed_text = "".join(blocks)
            break
        except Exception as e:
            err_str = str(e)
            if attempt == 3:
                errors.append(f"MEM_{char_num}_{start_id:04d}-{start_id+count-1:04d}: {e}")
                return start_id, "", errors, total_usage
            if "insufficient" in err_str:
                # Format problem — strengthen the format reminder.
                prompt += f"\n\nIMPORTANT: you MUST output {count} complete memories using the ===MEM_{char_num}_XXXX=== delimiter format. Each memory must include every field plus the ---content_full--- delimiter. The previous output had wrong formatting and could not be parsed; please re-output."
            else:
                prompt += "\nRetry: please generate completely different scenes; ensure batch-internal uniqueness. content_full must reach 600 words!"

    got = count if parsed_text else 0
    with lock:
        progress["done"] += got
        done = progress["done"]
        total = progress["total"]

    pct = done / total * 100 if total else 0
    elapsed = time.time() - progress.get("t0", time.time())
    speed = done / max(elapsed, 1) * 60
    tok_info = f"  in={total_usage.get('prompt_tokens','?')} out={total_usage.get('completion_tokens','?')}" if total_usage else ""
    log(f"[{done:>3}/{total}] {pct:5.1f}%  MEM_{char_num}_{start_id:04d}~{start_id+got-1:04d}  [{layer}]  ({speed:.0f}/min){tok_info}")

    return start_id, parsed_text, errors, total_usage


# ==================== Logging ====================

PROGRESS_LOG = None

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if PROGRESS_LOG:
        with lock:
            with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")


# ==================== API connectivity test ====================

def test_api():
    log("[TEST] Testing API connectivity...")
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Please reply OK"}],
            max_tokens=32,
        )
        reply = r.choices[0].message.content.strip()
        usage = {}
        if r.usage:
            usage = {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens}
        log(f"[TEST] OK! Model={MODEL}, reply='{reply}', usage={usage}")
        return True
    except Exception as e:
        log(f"[TEST] FAILED: {e}")
        return False


# ==================== Main ====================

def main():
    global PROGRESS_LOG

    parser = argparse.ArgumentParser(description="Optimized memory generation")
    parser.add_argument("--char", required=True, choices=["01", "04", "08"],
                        help="Character ID")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--mems-per-call", type=int, default=MEMS_PER_CALL,
                        help="Number of memories per API call (default 20)")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Run only the first N sub-tasks (debug)")
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default=None, help="Override the model name")
    args = parser.parse_args()

    global MODEL
    if args.model:
        MODEL = args.model

    char_key = args.char
    char_num = char_key.zfill(2)
    PROGRESS_LOG = DATA_DIR / f"_char{char_num}_opt_progress.log"
    PROGRESS_LOG.write_text("", encoding="utf-8")

    system_prompts = {layer: build_system_prompt(char_key, layer) for layer in MEMORY_LAYERS}
    log(f"System prompts built for {len(system_prompts)} layers")

    tasks = build_tasks(char_key, args.mems_per_call)
    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    progress["total"] = sum(t["count"] for t in tasks)

    log(f"=== Optimized Generation: CHAR_{char_num} ===")
    log(f"Model: {MODEL}, Workers: {args.workers}, Tasks: {len(tasks)}, "
        f"Total: {progress['total']} memories, mems_per_call={args.mems_per_call}")

    if not test_api():
        log("API test failed, aborting.")
        sys.exit(1)

    # Resume: check existing
    out_path = DATA_DIR / f"_char{char_num}_opt_generated.txt"
    existing_ids = set()
    if args.resume and out_path.exists():
        raw = out_path.read_text(encoding="utf-8")
        existing_ids = {int(m.group(1)) for m in re.finditer(rf"===MEM_{char_num}_(\d{{4}})===", raw)}
        existing_mems = parse_blocks_for_registry(raw, char_num)
        register_uniques(existing_mems)
        log(f"Resume: found {len(existing_ids)} existing")
        tasks = [t for t in tasks if t["mem_start"] not in existing_ids]
        progress["done"] = len(existing_ids)
        progress["total"] = len(existing_ids) + sum(t["count"] for t in tasks)

    if not tasks:
        log("All tasks already completed!")
        return

    log(f"Starting {len(tasks)} tasks with {args.workers} workers...")

    t0 = time.time()
    progress["t0"] = t0
    results = {}
    all_errors = []
    total_input_tokens = 0
    total_output_tokens = 0

    partial_path = DATA_DIR / f"_char{char_num}_opt_partial.txt"
    if not args.resume:
        partial_path.write_text("", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_task, t, char_key, system_prompts, args.max_output_tokens): t
            for t in tasks
        }
        for fut in as_completed(futures):
            start_id, text, errors, usage = fut.result()
            if text:
                results[start_id] = text
                with lock:
                    with open(partial_path, "a", encoding="utf-8") as f:
                        f.write(text)
            all_errors.extend(errors)
            if usage:
                total_input_tokens += usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("completion_tokens", 0)
            for e in errors:
                log(f"  ERROR: {e}")

    elapsed = time.time() - t0

    # Sort and write
    sorted_text = "".join(results[k] for k in sorted(results.keys()))
    if args.resume and out_path.exists():
        existing = out_path.read_text(encoding="utf-8")
        out_path.write_text(existing + sorted_text, encoding="utf-8")
    else:
        out_path.write_text(sorted_text, encoding="utf-8")

    final_raw = out_path.read_text(encoding="utf-8")
    final_count = len(re.findall(rf"===MEM_{char_num}_\d{{4}}===", final_raw))
    summaries = re.findall(r"content_summary:\s*(.+)", final_raw)
    unique_summaries = len(set(summaries))

    log(f"{'='*60}")
    log(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log(f"Total memories: {final_count}/900")
    log(f"Unique summaries: {unique_summaries}/{len(summaries)}")
    log(f"Token usage: input={total_input_tokens:,}, output={total_output_tokens:,}, total={total_input_tokens+total_output_tokens:,}")
    if total_input_tokens > 0:
        log(f"Est. cost (Sonnet): input=${total_input_tokens/1e6*3:.2f}, output=${total_output_tokens/1e6*15:.2f}, "
            f"total=${total_input_tokens/1e6*3 + total_output_tokens/1e6*15:.2f}")
    if all_errors:
        log(f"Errors ({len(all_errors)}):")
        for e in all_errors[:10]:
            log(f"  - {e}")
    log(f"Output: {out_path}")
    log(f"{'='*60}")


def build_expand_prompt(char_key: str, mem: dict) -> str:
    """Build the per-memory expansion prompt.

    Used to grow a short content_full to ~600 words.
    """
    p = CHAR_PROFILES[char_key]
    patterns = "; ".join(p["core_patterns"][:3])
    return f"""Character: {p['name']} ({p['archetype']}) | {p['big_five']}
Core patterns: {patterns}

Below is the content_full of one episodic memory. It is currently shorter than 600 words. Preserving the original plot and style, expand it to exactly 600-650 words (excluding punctuation).

Expansion strategy:
- Do NOT alter existing dialogue or key sentences
- Add small amounts of detail along these dimensions: sensory environment, extended inner monologue, finer bodily reactions, deeper aftermath
- Keep the first-person voice and the character's personality consistent
- Stay strictly within 600-650 words (excluding punctuation); do NOT exceed 700 words
- Output only the expanded content_full as plain text — no headings, no markers, no JSON

## Memory metadata
timeline: {mem.get('timeline','')}
context: {mem.get('context','')}
content_summary: {mem.get('content_summary','')}
emotion: {mem.get('emotion_primary','')} / {mem.get('emotion_secondary','')} (intensity={mem.get('emotion_intensity','0.7')})

## Current content_full (to expand)
{mem.get('content_full','')}"""


def expand_short_memories(char_key: str, threshold: int = 550,
                          workers: int = 6, max_tokens: int = 3000):
    """Read the generated file, find entries shorter than `threshold` words in
    content_full, and expand them one by one."""
    char_num = char_key.zfill(2)
    out_path = DATA_DIR / f"_char{char_num}_opt_generated.txt"
    if not out_path.exists():
        log(f"File not found: {out_path}")
        return

    raw = out_path.read_text(encoding="utf-8")

    # Locate every original block.
    pattern = rf"(===MEM_{char_num}_(\d{{3,4}})===\n)(.*?)(?====MEM_{char_num}_\d{{3,4}}===|\Z)"
    blocks = list(re.finditer(pattern, raw, re.DOTALL))

    short_items = []
    for m in blocks:
        mem_id = int(m.group(2))
        block_text = m.group(3)
        parts = block_text.split("---content_full---", 1)
        if len(parts) != 2:
            continue
        meta_text = parts[0].strip()
        content_full = parts[1].strip()
        word_count = len(content_full.split())
        if word_count < threshold:
            # Parse meta
            meta = {}
            for line in meta_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                idx = line.find(":")
                if idx == -1:
                    continue
                meta[line[:idx].strip()] = line[idx + 1:].strip()
            meta["content_full"] = content_full
            short_items.append({
                "mem_id": mem_id,
                "match": m,
                "meta": meta,
                "word_count": word_count,
            })

    if not short_items:
        log("No short memories found! All entries meet threshold.")
        return

    log(f"=== Expand Mode: CHAR_{char_num} ===")
    log(f"Threshold: {threshold} words, Found {len(short_items)} short entries to expand")
    log(f"Model: {MODEL}, Workers: {workers}")

    t0 = time.time()
    total_input = 0
    total_output = 0
    expanded_count = 0
    failed_ids = []
    replacements = {}  # mem_id -> new_content_full

    def expand_one(item):
        nonlocal total_input, total_output
        mem_id = item["mem_id"]
        prompt = build_expand_prompt(char_key, item["meta"])
        try:
            new_text, usage = call_llm(
                "You are an expert in psychological narrative writing, specialised in expanding first-person episodic memories. Output plain text only.",
                prompt,
                max_tokens=max_tokens,
            )
            new_words = len(new_text.split())
            if new_words < item["word_count"]:
                raise ValueError(f"expanded shorter: {new_words} < {item['word_count']}")
            with lock:
                nonlocal expanded_count
                expanded_count += 1
                if usage:
                    total_input += usage.get("prompt_tokens", 0)
                    total_output += usage.get("completion_tokens", 0)
            log(f"  [expand] MEM_{char_num}_{mem_id:04d}: {item['word_count']}→{new_words} words")
            return mem_id, new_text
        except Exception as e:
            log(f"  [expand FAIL] MEM_{char_num}_{mem_id:04d}: {e}")
            return mem_id, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(expand_one, item): item for item in short_items}
        for fut in as_completed(futures):
            mem_id, new_text = fut.result()
            if new_text:
                replacements[mem_id] = new_text
            else:
                failed_ids.append(mem_id)

    # Replace content_full in the original text.
    if replacements:
        new_raw = raw
        for m in reversed(blocks):  # replace from the end to avoid offset shifts
            mem_id = int(m.group(2))
            if mem_id not in replacements:
                continue
            block_text = m.group(3)
            parts = block_text.split("---content_full---", 1)
            if len(parts) != 2:
                continue
            meta_part = parts[0]
            old_content = parts[1]
            new_block = meta_part + "---content_full---\n" + replacements[mem_id] + "\n\n"
            start = m.start(3)
            end = m.end(3)
            new_raw = new_raw[:start] + new_block + new_raw[end:]

        out_path.write_text(new_raw, encoding="utf-8")

    elapsed = time.time() - t0
    log(f"{'='*60}")
    log(f"Expand done in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log(f"Expanded: {expanded_count}/{len(short_items)}, Failed: {len(failed_ids)}")
    log(f"Token usage: input={total_input:,}, output={total_output:,}")
    if total_input > 0:
        log(f"Est. cost: input=${total_input/1e6*3:.2f}, output=${total_output/1e6*15:.2f}, "
            f"total=${total_input/1e6*3 + total_output/1e6*15:.2f}")
    if failed_ids:
        log(f"Failed IDs: {failed_ids[:20]}")
    log(f"{'='*60}")


def main_cli():
    global PROGRESS_LOG, MODEL

    parser = argparse.ArgumentParser(description="Optimized memory generation")
    sub = parser.add_subparsers(dest="command", help="Command")

    # generate subcommand (default behavior)
    gen = sub.add_parser("generate", aliases=["gen"], help="Generate memories")
    gen.add_argument("--char", required=True, choices=["01", "04", "08"])
    gen.add_argument("--workers", type=int, default=WORKERS)
    gen.add_argument("--mems-per-call", type=int, default=MEMS_PER_CALL)
    gen.add_argument("--max-tasks", type=int, default=0)
    gen.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    gen.add_argument("--resume", action="store_true")
    gen.add_argument("--model", default=None)

    # expand subcommand
    exp = sub.add_parser("expand", help="Expand short content_full entries")
    exp.add_argument("--char", required=True, choices=["01", "04", "08"])
    exp.add_argument("--threshold", type=int, default=550,
                     help="Expand entries below this word count (default 550)")
    exp.add_argument("--workers", type=int, default=6)
    exp.add_argument("--model", default=None)

    # stats subcommand
    st = sub.add_parser("stats", help="Show content_full length stats")
    st.add_argument("--char", required=True, choices=["01", "04", "08"])

    args = parser.parse_args()

    if args.model if hasattr(args, 'model') and args.model else None:
        MODEL = args.model

    if args.command in ("expand",):
        char_num = args.char.zfill(2)
        PROGRESS_LOG = DATA_DIR / f"_char{char_num}_opt_progress.log"
        expand_short_memories(args.char, threshold=args.threshold,
                              workers=args.workers)

    elif args.command in ("stats",):
        char_num = args.char.zfill(2)
        out_path = DATA_DIR / f"_char{char_num}_opt_generated.txt"
        if not out_path.exists():
            print(f"File not found: {out_path}")
            return
        raw = out_path.read_text(encoding="utf-8")
        pattern = rf"===MEM_{char_num}_\d{{3,4}}===\n(.*?)(?====MEM_{char_num}_\d{{3,4}}===|\Z)"
        lengths = []
        for m in re.finditer(pattern, raw, re.DOTALL):
            parts = m.group(1).split("---content_full---", 1)
            if len(parts) == 2:
                cf = parts[1].strip()
                hanzi = len(cf.split())
                lengths.append(hanzi)
        if not lengths:
            print("No memories found.")
            return
        lengths.sort()
        n = len(lengths)
        print(f"Total memories: {n}")
        print(f"Avg words: {sum(lengths)/n:.0f}")
        print(f"Min: {min(lengths)}, Max: {max(lengths)}, Median: {lengths[n//2]}")
        print(f"P25: {lengths[n//4]}, P75: {lengths[3*n//4]}")
        for lo, hi, label in [(0,450,"<450"), (450,500,"450-499"), (500,550,"500-549"),
                               (550,600,"550-599"), (600,9999,"600+")]:
            cnt = sum(1 for l in lengths if lo <= l < hi)
            print(f"  {label}: {cnt} ({cnt/n*100:.0f}%)")

    else:
        # Default: generate (also handles no subcommand for backward compat)
        if args.command is None:
            # Re-parse with old-style args for backward compatibility
            main()
            return
        if hasattr(args, 'model') and args.model:
            MODEL = args.model
        char_key = args.char
        char_num = char_key.zfill(2)
        PROGRESS_LOG = DATA_DIR / f"_char{char_num}_opt_progress.log"
        PROGRESS_LOG.write_text("", encoding="utf-8")

        # Build a system prompt per layer.
        system_prompts = {layer: build_system_prompt(char_key, layer) for layer in MEMORY_LAYERS}
        for layer, sp in system_prompts.items():
            log(f"System prompt [{layer}]: {len(sp)} chars")

        tasks = build_tasks(char_key, args.mems_per_call)
        if args.max_tasks > 0:
            tasks = tasks[:args.max_tasks]

        progress["total"] = sum(t["count"] for t in tasks)

        # Per-layer task distribution stats.
        layer_dist = {}
        for t in tasks:
            layer_dist[t["layer"]] = layer_dist.get(t["layer"], 0) + t["count"]
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(layer_dist.items()))

        log(f"=== Layered Generation: CHAR_{char_num} ===")
        log(f"Model: {MODEL}, Workers: {args.workers}, Tasks: {len(tasks)}, "
            f"Total: {progress['total']} memories, mems_per_call={args.mems_per_call}")
        log(f"Layer distribution: {dist_str}")

        if not test_api():
            log("API test failed, aborting.")
            sys.exit(1)

        out_path = DATA_DIR / f"_char{char_num}_opt_generated.txt"
        existing_ids = set()
        if args.resume and out_path.exists():
            raw_text = out_path.read_text(encoding="utf-8")
            existing_ids = {int(m.group(1)) for m in re.finditer(rf"===MEM_{char_num}_(\d{{4}})===", raw_text)}
            existing_mems = parse_blocks_for_registry(raw_text, char_num)
            register_uniques(existing_mems)
            log(f"Resume: found {len(existing_ids)} existing")
            tasks = [t for t in tasks if t["mem_start"] not in existing_ids]
            progress["done"] = len(existing_ids)
            progress["total"] = len(existing_ids) + sum(t["count"] for t in tasks)

        if not tasks:
            log("All tasks already completed!")
            return

        log(f"Starting {len(tasks)} tasks with {args.workers} workers...")

        t0 = time.time()
        progress["t0"] = t0
        results = {}
        all_errors = []
        total_input_tokens = 0
        total_output_tokens = 0

        partial_path = DATA_DIR / f"_char{char_num}_opt_partial.txt"
        if not args.resume:
            partial_path.write_text("", encoding="utf-8")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_task, t, char_key, system_prompts, args.max_output_tokens): t
                for t in tasks
            }
            for fut in as_completed(futures):
                start_id, text, errors, usage = fut.result()
                if text:
                    results[start_id] = text
                    with lock:
                        with open(partial_path, "a", encoding="utf-8") as f:
                            f.write(text)
                all_errors.extend(errors)
                if usage:
                    total_input_tokens += usage.get("prompt_tokens", 0)
                    total_output_tokens += usage.get("completion_tokens", 0)
                for e in errors:
                    log(f"  ERROR: {e}")

        elapsed = time.time() - t0

        sorted_text = "".join(results[k] for k in sorted(results.keys()))
        if args.resume and out_path.exists():
            existing = out_path.read_text(encoding="utf-8")
            out_path.write_text(existing + sorted_text, encoding="utf-8")
        else:
            out_path.write_text(sorted_text, encoding="utf-8")

        final_raw = out_path.read_text(encoding="utf-8")
        final_count = len(re.findall(rf"===MEM_{char_num}_\d{{4}}===", final_raw))
        summaries = re.findall(r"content_summary:\s*(.+)", final_raw)
        unique_summaries = len(set(summaries))

        log(f"{'='*60}")
        log(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")
        log(f"Total memories: {final_count}/900")
        log(f"Unique summaries: {unique_summaries}/{len(summaries)}")
        log(f"Token usage: input={total_input_tokens:,}, output={total_output_tokens:,}, total={total_input_tokens+total_output_tokens:,}")
        if total_input_tokens > 0:
            log(f"Est. cost (Sonnet): input=${total_input_tokens/1e6*3:.2f}, output=${total_output_tokens/1e6*15:.2f}, "
                f"total=${total_input_tokens/1e6*3 + total_output_tokens/1e6*15:.2f}")
        if all_errors:
            log(f"Errors ({len(all_errors)}):")
            for e in all_errors[:10]:
                log(f"  - {e}")
        log(f"Output: {out_path}")
        log(f"{'='*60}")


if __name__ == "__main__":
    main_cli()
