"""
extract_character.py — pipeline for extracting character traits and memories from a novel / text.
Usage:
  python extract_character.py --input novel.txt --character "Lin Daiyu" --output extracted.json
  python extract_character.py --input novel.txt --all --output all_characters.json
"""
import json, os, sys, argparse, urllib.request, urllib.error
from pathlib import Path

# ── Config ──
def _load_env(p):
    if not p.exists(): return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ: os.environ[k] = v

_load_env(Path(__file__).with_name(".env"))

API_CFG = {
    "api_key":  os.getenv("API_KEY", ""),
    "api_base": os.getenv("API_BASE", "https://aihubmix.com/v1").rstrip("/"),
    "model":    os.getenv("MODEL_NAME", "gpt-5-mini"),
    "timeout":  int(os.getenv("TIMEOUT", "120")),
}

# ── LLM ──
class LLM:
    def __init__(self, cfg=API_CFG):
        self.cfg = cfg

    def call(self, system_prompt, user_prompt, temperature=0.7):
        url = f"{self.cfg['api_base']}/chat/completions"
        payload = json.dumps({
            "model": self.cfg["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temperature,
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg['api_key']}",
        }, method="POST")
        with urllib.request.urlopen(req, timeout=self.cfg["timeout"]) as resp:
            r = json.loads(resp.read().decode("utf-8", errors="replace"))
        return r.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    def call_json(self, system_prompt, user_prompt, temperature=0.4, retries=2):
        import re
        for attempt in range(retries + 1):
            raw = self.call(system_prompt, user_prompt, temperature)
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            raw = raw.strip()
            # Strip trailing commas before } / ]
            raw = re.sub(r',\s*([}\]])', r'\1', raw)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                if attempt < retries:
                    print(f"  [RETRY] JSON parse failed: {e}, retrying...")
                else:
                    raise

# ── Text chunking ──
def chunk_text(text, chunk_size=4000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks

llm = LLM()

# ── Character detection ──
def detect_characters(chunks):
    sample = "\n---\n".join(chunks[:5])
    result = llm.call_json(
        "You are a literary-analysis expert. Identify the main characters from the text and return a JSON array.",
        f"From the following text fragments, identify all main characters (high-frequency, with character development).\n"
        f"Return format: [{{\"name\": \"<character name>\", \"description\": \"<one-sentence description>\"}}]\n\n{sample}"
    )
    return result

# ── Trait extraction ──
def extract_traits(name, chunks):
    relevant = [c for c in chunks if name in c][:8]
    text = "\n---\n".join(relevant) if relevant else "\n---\n".join(chunks[:5])
    return llm.call_json(
        "You are an expert in psychology and literary analysis. From the text, extract the character's Big Five personality traits and Schwartz value orientation.",
        f"Character: {name}\n\nRelevant text:\n{text}\n\n"
        f"Please return JSON:\n"
        f'{{"big_five": {{"openness": 0.0-1.0, "conscientiousness": 0.0-1.0, '
        f'"extraversion": 0.0-1.0, "agreeableness": 0.0-1.0, "neuroticism": 0.0-1.0}},\n'
        f'"description": "overall description of the character",\n'
        f'"value_orientation": {{"framework": "schwartz_refined_19", "scores": {{...19 values...}}, '
        f'"dominant_values": [...], "suppressed_values": [...], '
        f'"value_narrative": "..."}}}}'
    )

# ── Memory extraction ──
MEMORY_SCHEMA_HINT = (
    '{"id":"MEM_XX_01","timeline":{"period":"childhood","age":8},'
    '"context":{"location":"...","people_involved":["..."],"trigger":"..."},'
    '"content_summary":"one-sentence summary","content_full":"detailed narrative (500+ words)",'
    '"emotion_signature":{"primary":"...","intensity":0.8,"valence":-0.6},'
    '"relevance_tags":["..."]}'
)

def extract_memories(name, chunks, prefix="XX", max_memories=20):
    relevant = [c for c in chunks if name in c]
    if not relevant:
        relevant = chunks[:5]
    memories = []
    # Process in batches of up to 5 chunks.
    for i in range(0, len(relevant), 5):
        batch = "\n---\n".join(relevant[i:i+5])
        batch_result = llm.call_json(
            "You are an expert in literary psychology. Extract the character's episodic memories from the text.",
            f"Character: {name}\n\nText:\n{batch}\n\n"
            f"Extract the key events the character experienced in these passages as episodic memories. Each memory format:\n"
            f"{MEMORY_SCHEMA_HINT}\n\n"
            f"Return a JSON array; ids start at MEM_{prefix}_{len(memories)+1:02d}. "
            f"Try to extract every event with psychological meaning."
        )
        if isinstance(batch_result, list):
            memories.extend(batch_result)
        if len(memories) >= max_memories:
            break
    return memories[:max_memories]

# ── Validation and assembly ──
def validate_and_assemble(name, traits, memories):
    char = {
        "id": f"CHAR_EXTRACTED_{name}",
        "name": name,
        "big_five": traits.get("big_five", {}),
        "description": traits.get("description", ""),
        "value_orientation": traits.get("value_orientation", {}),
        "episodic_memory_set": memories,
    }
    # Validate big_five
    bf = char["big_five"]
    for k in ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]:
        v = bf.get(k)
        if v is None or not (0 <= float(v) <= 1):
            print(f"  [WARN] big_five.{k} = {v}, clamping to [0,1]")
            bf[k] = max(0.0, min(1.0, float(v or 0.5)))
    # Validate memories
    print(f"  {name}: {len(memories)} memories extracted")
    if len(memories) < 10:
        print(f"  [WARN] fewer than 10 memories")
    return char

# ── Entry points ──
def extract_character(input_path, character_name, output_path):
    text = Path(input_path).read_text(encoding="utf-8")
    chunks = chunk_text(text)
    print(f"Text chunked: {len(chunks)} chunks")

    prefix = character_name[:2].upper()
    print(f"Extracting traits for: {character_name}")
    traits = extract_traits(character_name, chunks)
    print(f"Extracting memories for: {character_name}")
    memories = extract_memories(character_name, chunks, prefix=prefix)
    char = validate_and_assemble(character_name, traits, memories)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(char, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")
    return char

def extract_all(input_path, output_path):
    text = Path(input_path).read_text(encoding="utf-8")
    chunks = chunk_text(text)
    print(f"Text chunked: {len(chunks)} chunks")

    print("Detecting characters...")
    characters = detect_characters(chunks)
    print(f"Found {len(characters)} characters: {[c['name'] for c in characters]}")

    results = []
    for i, c in enumerate(characters):
        name = c["name"]
        prefix = f"{i+1:02d}"
        print(f"\n[{i+1}/{len(characters)}] Processing: {name}")
        traits = extract_traits(name, chunks)
        memories = extract_memories(name, chunks, prefix=prefix)
        char = validate_and_assemble(name, traits, memories)
        results.append(char)

    output = {"characters": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nAll saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract character traits and memories from a text.")
    parser.add_argument("--input", required=True, help="path to the input text file")
    parser.add_argument("--character", help="specific character name")
    parser.add_argument("--all", action="store_true", help="extract all characters")
    parser.add_argument("--output", required=True, help="path of the output JSON")
    args = parser.parse_args()

    if args.all:
        extract_all(args.input, args.output)
    elif args.character:
        extract_character(args.input, args.character, args.output)
    else:
        print("Specify either --character or --all.")
        sys.exit(1)
