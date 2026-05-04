"""Show side-by-side: original content_summary vs what mem0 actually stored."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
CHARS = json.loads((ROOT / "characters_phase11.json").read_text())

# Build mem_id -> original record
orig_by_id = {}
for c in CHARS["characters"]:
    for m in c["episodic_memory_set"]:
        orig_by_id[m["id"]] = (c["id"], m)

# Walk one snapshot file, pick a few stored memories
snap = ROOT / ".mem0" / "snapshots" / "CHAR_01" / "age_06.jsonl"
shown = 0
for line in snap.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    payload = row["payload"] or {}
    mid = payload.get("mem_id")
    stored_text = payload.get("data") or payload.get("memory") or ""
    if not mid or mid not in orig_by_id:
        continue
    cid, orig = orig_by_id[mid]
    print("=" * 78)
    print(f"mem_id    : {mid}   ({cid})")
    print(f"timeline  : {orig['timeline']}")
    print(f"--- ORIGINAL content_summary (Chinese, what we sent to mem0) ---")
    print(orig["content_summary"])
    print(f"--- STORED in qdrant (LLM-extracted fact, what gets searched) ---")
    print(stored_text)
    print(f"--- METADATA preserved as-is (rides along, queryable by filter) ---")
    print(f"  triggers          : {payload.get('triggers')}")
    print(f"  psych_conclusion  : {(payload.get('psych_conclusion') or '')[:80]}")
    print(f"  emotion_signature : {payload.get('emotion_signature')}")
    print(f"  content_full kept : {bool(payload.get('content_full'))}  "
          f"(len={len(payload.get('content_full') or '')})")
    print()
    shown += 1
    if shown >= 4:
        break
