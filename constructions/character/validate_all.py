import json
import os
from pathlib import Path

data_path = Path(__file__).parent.parent / 'data' / 'characters' / 'characters_phase3.json'

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print("1. JSON valid OK")

SCHWARTZ_19 = [
    "self_direction_thought","self_direction_action","stimulation","hedonism",
    "achievement","power_dominance","power_resources","face",
    "security_personal","security_societal","tradition",
    "conformity_rules","conformity_interpersonal","humility",
    "benevolence_caring","benevolence_dependability",
    "universalism_concern","universalism_nature","universalism_tolerance"
]

for ch in data['characters']:
    cid = ch['id']
    print(f"\n=== {cid} ===")

    # memories
    mems = ch.get('episodic_memory_set', [])
    print(f"  memories: {len(mems)}")

    # life_threads
    threads = ch.get('life_threads', [])
    print(f"  life_threads: {len(threads)}")
    covered = set()
    for t in threads:
        covered.update(t['memory_ids'])
    actual = {m['id'] for m in mems}
    missing = actual - covered
    print(f"    coverage: {len(covered)}/{len(mems)}, missing: {missing or 'none'}")

    # value_orientation
    vo = ch.get('value_orientation', {})
    scores = vo.get('scores', {})
    missing_vals = [v for v in SCHWARTZ_19 if v not in scores]
    bad_range = {k:v for k,v in scores.items() if not (0<=v<=1)}
    print(f"  value_orientation: {len(scores)}/19 values", end="")
    if missing_vals: print(f", MISSING: {missing_vals}", end="")
    if bad_range: print(f", OUT OF RANGE: {bad_range}", end="")
    print()

    # life_snapshots
    snaps = ch.get('life_snapshots', [])
    print(f"  life_snapshots: {len(snaps)}")
    for s in snaps:
        print(f"    {s['snapshot_id']}: {s['stage']} (age {s['age']}) - {s['label']}")

print("\n=== SUMMARY ===")
all_ok = True
for ch in data['characters']:
    issues = []
    if len(ch.get('life_threads',[])) < 4: issues.append("threads<4")
    if len(ch.get('value_orientation',{}).get('scores',{})) != 19: issues.append("values!=19")
    if len(ch.get('life_snapshots',[])) != 6: issues.append("snapshots!=6")
    status = "PASS" if not issues else f"FAIL: {issues}"
    print(f"  {ch['id']}: {status}")
    if issues: all_ok = False
print(f"\nOverall: {'ALL PASS' if all_ok else 'HAS ISSUES'}")
