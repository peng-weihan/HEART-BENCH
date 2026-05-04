#!/usr/bin/env python3
"""
For every <character, scenario> test unit, intelligently select the 3 best
distractor options.

Implemented in pure stdlib Python — no extra dependencies.

Selection strategy:
1. Exclude options identical to the correct answer.
2. Prefer options that are semantically similar but behaviourally different
   (raises difficulty).
3. Ensure the options are sufficiently different from one another.
4. Take personality-trait contrast into account (e.g. High vs Low Neuroticism).

NOTE: the LLM-facing prompt strings later in this file are intentionally in
Chinese — they steer the LLM that operates on the Chinese narrative dataset
and must keep matching it.
"""

from openai import OpenAI
import os
import re
import random
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import math

def load_final_gt():
    """Load the final GT data."""
    gt_path = Path("benchmark/annotations/final_gt_annotations.json")
    with open(gt_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_all_annotations():
    """Load the final GT data, using only consensus-validated annotations as candidate options."""
    final_gt = load_final_gt()
    scenario_annotations = defaultdict(list)  # scenario_id -> [(char_id, decision)]

    # Extract each character's unique, validated decision per scenario from the final GT
    for pair in final_gt['pairs']:
        char_id = pair['character_id']
        scenario_id = pair['scenario_id']
        final_decision = pair['final_decision']

        if final_decision and len(final_decision.strip()) > 10:
            scenario_annotations[scenario_id].append((char_id, final_decision))

    return scenario_annotations

def load_character_profiles():
    """Load character personality traits."""
    character_traits = {
        "CHAR_01": {"type": "N-High", "main_trait": "Neuroticism", "level": "high"},
        "CHAR_02": {"type": "N-Low", "main_trait": "Neuroticism", "level": "low"},
        "CHAR_03": {"type": "C-High", "main_trait": "Conscientiousness", "level": "high"},
        "CHAR_04": {"type": "C-Low", "main_trait": "Conscientiousness", "level": "low"},
        "CHAR_05": {"type": "E-High", "main_trait": "Extraversion", "level": "high"},
        "CHAR_06": {"type": "E-Low", "main_trait": "Extraversion", "level": "low"},
        "CHAR_07": {"type": "A-High", "main_trait": "Agreeableness", "level": "high"},
        "CHAR_08": {"type": "A-Low", "main_trait": "Agreeableness", "level": "low"},
        "CHAR_09": {"type": "O-High", "main_trait": "Openness", "level": "high"},
        "CHAR_10": {"type": "O-Low", "main_trait": "Openness", "level": "low"},
        "CHAR_11": {"type": "Neutral", "main_trait": "Neutral", "level": "neutral"},
    }
    return character_traits

def simple_tokenize(text):
    """Simple Chinese tokenisation."""
    if not text:
        return []

    # Strip punctuation, keep alphanumerics and whitespace
    cleaned = re.sub(r'[^\w\s]', ' ', text)

    # Whitespace-based split for English
    chars = [c for c in cleaned.split() if c.strip()]

    # Also build 2-grams
    tokens = chars[:]
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])

    return tokens

def calculate_text_similarity_simple(text1, text2):
    """Compute text similarity via simple lexical overlap."""
    if not text1 or not text2:
        return 0.0

    tokens1 = simple_tokenize(text1.lower())
    tokens2 = simple_tokenize(text2.lower())

    if not tokens1 or not tokens2:
        return 0.0

    # Jaccard similarity
    set1 = set(tokens1)
    set2 = set(tokens2)

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0.0

    jaccard = intersection / union

    # Similarity with token-frequency info
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)

    all_words = set(counter1.keys()).union(set(counter2.keys()))

    if not all_words:
        return 0.0

    # Cosine similarity
    dot_product = sum(counter1[word] * counter2[word] for word in all_words)
    magnitude1 = math.sqrt(sum(counter1[word] ** 2 for word in all_words))
    magnitude2 = math.sqrt(sum(counter2[word] ** 2 for word in all_words))

    if magnitude1 == 0 or magnitude2 == 0:
        return jaccard

    cosine = dot_product / (magnitude1 * magnitude2)

    # Combine both similarity metrics
    return (jaccard + cosine) / 2

def clean_decision_text(decision_text):
    """Clean a decision text — strip away formatting noise."""
    if not decision_text:
        return ""

    text = re.sub(r'\s+', ' ', decision_text.strip())
    text = re.sub(r'["""]', '', text)
    text = re.sub(r'^(I (?:will )?choose|I (?:will )?decide|I choose|I decide)\s+', '', text, flags=re.IGNORECASE)

    return text.lower()

def extract_action_keywords(decision_text):
    """Extract key action keywords from a decision."""
    if not decision_text:
        return set()

    action_patterns = [
        r'\b(refuse|accept|agree|oppose|deny|approve)\b',
        r'\b(choose|decide|pick|select)\b',
        r'\b(leave|stay|walk away|go back|go home|go out)\b',
        r'\b(speak up|conceal|confess|lie|tell|keep secret)\b',
        r'\b(help|refuse|support|resist|assist|stop)\b',
        r'\b(join|withdraw|enter|leave|participate|retreat)\b',
        r'\b(persist|give up|compromise|yield|stand firm|back down)\b',
        r'\b(proactive|passive|positive|negative|brave|timid)\b',
    ]

    keywords = set()
    for pattern in action_patterns:
        matches = re.findall(pattern, decision_text, flags=re.IGNORECASE)
        keywords.update(m.lower() for m in matches)

    return keywords

def is_same_decision_llm(client: OpenAI, decision1: str, decision2: str) -> bool:
    """Use Claude Sonnet 4.6 to judge whether two decisions are substantively the same."""

    system_prompt = """You are a personality-psychology expert specialised in analysing behavioural differences. You will receive two behavioural-decision texts.

Task: judge whether these two decisions exhibit a **significant personality / behavioural difference** — i.e. whether they could serve as distinct options in a psychological test.

Difference dimensions to look at:
1. **Core decision direction**: completely opposite choices (e.g. accept vs refuse, proactive vs passive)
2. **Communication style**: direct vs tactful, concise vs detailed, formal vs casual
3. **Emotional expression**: anxious vs calm, confident vs uneasy, warm vs cold
4. **Social consideration**: weighing others' feelings, preserving the relationship vs direct expression
5. **Reason type**: moral vs pragmatic, emotional vs rational, personal vs social

Judging rule:
- If the two decisions differ **noticeably** along ANY one of the dimensions above, judge them "different".
- Only when the two decisions are **highly similar across ALL dimensions** should you judge them "same".
- Differences in expression style, tone or specific wording are **themselves important behavioural differences**.

Output: print only "same" or "different"; no other text."""

    user_prompt = f"""Decision A: {decision1}

Decision B: {decision2}

Judge whether these two decisions are substantively the same at the behavioural level:"""

    try:
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=10
        )

        result = response.choices[0].message.content.strip()
        return result == "same"

    except Exception as e:
        print(f"Error in LLM decision similarity judgment: {e}", file=sys.stderr)
        # On error, conservatively default to 'different'
        return False

def is_batch_same_decision_llm(client: OpenAI, decision_pairs: List[Tuple[str, str]]) -> List[bool]:
    """Use Claude Sonnet 4.6 to judge similarity for many decision pairs in batch."""

    if not decision_pairs:
        return []

    system_prompt = """You are a personality-psychology expert specialised in analysing behavioural differences. I will give you multiple decision-text pairs; judge each pair separately for whether it reflects a meaningful personality or behavioural difference.

Task: Decide whether each pair of decisions reflects a **significant personality or behavioural difference** that would qualify them as distinct options in a psychology test.

Dimensions of difference to consider:
1. **Core decision direction**: opposite choices (e.g. accept vs refuse, proactive vs passive)
2. **Communication style**: direct vs tactful, concise vs detailed, formal vs casual
3. **Emotional expression**: anxious vs calm, confident vs uneasy, warm vs cold
4. **Social consideration**: whether others' feelings are considered, relationship-preserving vs blunt
5. **Type of reasoning**: moral vs practical, emotional vs rational, personal vs social

Judgement rules:
- If two decisions differ **clearly on any of the dimensions above**, label them "different"
- Only when two decisions are **highly similar across all dimensions** should they be labelled "same"
- Differences in expression style, tone or specific wording **are themselves important behavioural differences**

Output format: for every pair, print only "same" or "different", one result per line. No other text."""

    # Build the user prompt for the batch judgement
    user_prompt = ""
    for i, (correct_decision, candidate_decision) in enumerate(decision_pairs, 1):
        user_prompt += f"Pair {i}:\nCorrect answer: {correct_decision}\nCandidate option: {candidate_decision}\n\n"

    user_prompt += "Please judge each decision pair one by one for substantive equivalence:"

    try:
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=len(decision_pairs) * 10 + 50
        )

        result = response.choices[0].message.content.strip()
        lines = [line.strip() for line in result.split('\n') if line.strip()]

        # Parse the result
        same_results = []
        for i, line in enumerate(lines):
            if i >= len(decision_pairs):
                break
            # Check whether the response says "same"
            line_lower = line.lower()
            is_same = "same" in line_lower and "different" not in line_lower
            same_results.append(is_same)

        # If the result count does not match, pad conservatively
        while len(same_results) < len(decision_pairs):
            same_results.append(False)

        return same_results[:len(decision_pairs)]

    except Exception as e:
        print(f"Error in batch LLM decision similarity judgment: {e}", file=sys.stderr)
        # On error, conservatively default everything to 'different'
        return [False] * len(decision_pairs)

def get_personality_contrast_score(char1, char2, character_traits):
    """Compute the personality-contrast score between two characters."""
    traits1 = character_traits.get(char1, {})
    traits2 = character_traits.get(char2, {})

    # Same trait at opposite levels scores higher (more pedagogically useful)
    if traits1.get("main_trait") == traits2.get("main_trait"):
        if traits1.get("level") != traits2.get("level"):
            return 2.0  # High contrast: e.g. High Neuroticism vs Low Neuroticism
        else:
            return 0.3  # Same level — not very meaningful

    # Cross-trait contrast
    return 1.0

def extract_decision_sentiment(decision_text):
    """Simple sentiment-tendency analysis."""
    if not decision_text:
        return 0

    positive_words = ['accept', 'agree', 'help', 'support', 'positive', 'proactive', 'brave', 'persist', 'participate', 'join']
    negative_words = ['refuse', 'oppose', 'leave', 'give up', 'negative', 'passive', 'timid', 'withdraw', 'avoid', 'hide']

    text_lower = decision_text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    return pos_count - neg_count

def calculate_option_quality_score(correct_decision, candidate_decision, target_char, candidate_char, character_traits):
    """Compute the quality score of a distractor option."""

    # 1. Text-similarity score (sweet spot 0.3-0.6)
    similarity = calculate_text_similarity_simple(
        clean_decision_text(correct_decision),
        clean_decision_text(candidate_decision)
    )

    if 0.25 <= similarity <= 0.6:
        similarity_score = 1.0  # Optimal similarity range
    elif similarity < 0.25:
        similarity_score = 0.6  # Too easy to tell apart
    elif similarity < 0.75:
        similarity_score = 0.8  # Still distinguishable
    else:
        similarity_score = 0.2  # Too similar; easy to confuse

    # 2. Personality-contrast score
    personality_score = get_personality_contrast_score(target_char, candidate_char, character_traits)

    # 3. Action-divergence score
    keywords_correct = extract_action_keywords(correct_decision)
    keywords_candidate = extract_action_keywords(candidate_decision)

    if keywords_correct and keywords_candidate:
        overlap = len(keywords_correct.intersection(keywords_candidate))
        total = len(keywords_correct.union(keywords_candidate))
        if total > 0:
            action_diversity = 1.0 - (overlap / total)
        else:
            action_diversity = 0.5
    else:
        action_diversity = 0.7  # If no salient keyword, give a middling score

    # 4. Emotion-tendency divergence
    sentiment_correct = extract_decision_sentiment(correct_decision)
    sentiment_candidate = extract_decision_sentiment(candidate_decision)
    sentiment_diff = abs(sentiment_correct - sentiment_candidate)
    sentiment_score = min(sentiment_diff / 3, 1.0)  # Normalise

    # Weighted total
    total_score = (
        similarity_score * 0.35 +       # Text similarity matters most
        personality_score * 0.25 +      # Personality contrast
        action_diversity * 0.25 +       # Action divergence
        sentiment_score * 0.15          # Emotion divergence
    )

    return total_score, {
        "similarity": similarity,
        "similarity_score": similarity_score,
        "personality_score": personality_score,
        "action_diversity": action_diversity,
        "sentiment_score": sentiment_score,
        "total_score": total_score
    }

def generate_variant_distractors(target_char, scenario_id, correct_decision, num_needed, character_traits, client):
    """Generate variation distractors for high-convergence scenarios."""

    # Pull the target character's traits
    target_traits = character_traits.get(target_char, {})
    target_type = target_traits.get("type", "Unknown")
    target_trait = target_traits.get("main_trait", "Unknown")
    target_level = target_traits.get("level", "Unknown")

    # Pick different personality traits to seed variations
    contrast_chars = []
    for char_id, traits in character_traits.items():
        if char_id == target_char:
            continue
        # Prefer characters with opposite or distinct dominant traits
        if traits.get("main_trait") == target_trait and traits.get("level") != target_level:
            contrast_chars.append((char_id, traits, 3))  # High priority: same trait, opposite level
        elif traits.get("main_trait") != target_trait:
            contrast_chars.append((char_id, traits, 2))  # Medium priority: different trait
        else:
            contrast_chars.append((char_id, traits, 1))  # Low priority: same type

    # Sort by priority and take the top num_needed
    contrast_chars.sort(key=lambda x: x[2], reverse=True)
    selected_chars = contrast_chars[:num_needed]

    generated_options = []

    # Generate one variation option per selected character
    for char_id, traits, priority in selected_chars:
        char_type = traits.get("type", "Unknown")
        main_trait = traits.get("main_trait", "Unknown")
        level = traits.get("level", "Unknown")

        system_prompt = f"""You are a personality-psychology expert. Following the Big Five personality theory, generate a behavioural decision for the specified character that matches their personality traits.

Task: Generate a decision that has a clear behavioural difference from the given correct answer and reflects the specified personality traits.

Requirements:
1. The decision must be reasonable and realistic
2. It should reflect the typical expression of the specified personality traits
3. It must differ clearly from the correct answer in behaviour or attitude
4. The expression should feel natural and include both action and inner thoughts
5. Length: roughly 150-300 words

Output format: print only the decision text. No JSON, no other formatting."""

        user_prompt = f"""Correct answer ({target_char} - {target_type}):
{correct_decision}

Please generate a different behavioural decision for the following character:
Character: {char_id} ({char_type})
Trait: {main_trait} {level} level

Generate a decision that reflects this character's personality traits and clearly differs from the correct answer:"""

        try:
            response = client.chat.completions.create(
                model="claude-sonnet-4-6",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )

            generated_decision = response.choices[0].message.content.strip()

            # Score the generated options
            score, details = calculate_option_quality_score(
                correct_decision, generated_decision, target_char, char_id, character_traits
            )

            generated_options.append({
                "character_id": char_id,
                "decision": generated_decision,
                "score": score * 0.8,  # Lower the score to distinguish generated content
                "details": details,
                "is_generated": True  # Mark as generated
            })

        except Exception as e:
            print(f"    [{char_id} variant generation FAILED] {e}")
            continue

    return generated_options

def select_best_distractors(target_char, scenario_id, correct_decision, all_candidates, character_traits, client):
    """Select the best 3 distractors for a given (character, scenario) — batched version."""

    if not correct_decision:
        return [], "correct answer is empty"

    # Gather every candidate that needs judging
    candidate_pairs = []
    candidate_info = []

    for candidate_char, decision in all_candidates:
        if candidate_char == target_char:
            continue

        if not decision or len(decision.strip()) < 10:
            continue  # Skip empty or too-short decisions

        candidate_pairs.append((correct_decision, decision))
        candidate_info.append((candidate_char, decision))

    if not candidate_pairs:
        return [], "no valid candidate options"

    # Process in batches (12 pairs per batch to keep tokens manageable)
    batch_size = 12
    valid_candidates = []

    print(f"  Processing {len(candidate_pairs)} candidates in {(len(candidate_pairs) + batch_size - 1) // batch_size} batches", file=sys.stderr)

    for i in range(0, len(candidate_pairs), batch_size):
        batch_pairs = candidate_pairs[i:i+batch_size]
        batch_info = candidate_info[i:i+batch_size]

        print(f"    Batch {i//batch_size + 1}: processing {len(batch_pairs)} pairs", file=sys.stderr)

        # Judge similarity in batch
        try:
            same_results = is_batch_same_decision_llm(client, batch_pairs)
        except Exception as e:
            print(f"    Batch failed, falling back to individual processing: {e}", file=sys.stderr)
            # Fall back to one-by-one processing
            same_results = []
            for correct, candidate in batch_pairs:
                same_results.append(is_same_decision_llm(client, correct, candidate))

        # Collect candidates judged 'different'
        for j, is_same in enumerate(same_results):
            if j >= len(batch_info):
                break
            if not is_same:
                candidate_char, decision = batch_info[j]
                # Compute quality score
                score, details = calculate_option_quality_score(
                    correct_decision, decision, target_char, candidate_char, character_traits
                )
                valid_candidates.append({
                    "character_id": candidate_char,
                    "decision": decision,
                    "score": score,
                    "details": details
                })

    if len(valid_candidates) < 3:
        return valid_candidates, f"not enough candidates: only {len(valid_candidates)} valid"

    # Sort by score
    valid_candidates.sort(key=lambda x: x["score"], reverse=True)

    # Smart selection strategy: maximise diversity
    selected = [valid_candidates[0]]  # Pick the top-scoring candidate first

    remaining = valid_candidates[1:]

    # For the remaining 2, weigh divergence from already-picked options
    for _ in range(min(2, len(remaining))):
        best_candidate = None
        best_combined_score = -1

        for candidate in remaining:
            # Compute divergence from already-selected options
            min_diversity = 1.0
            for selected_item in selected:
                diversity = 1.0 - calculate_text_similarity_simple(
                    clean_decision_text(candidate["decision"]),
                    clean_decision_text(selected_item["decision"])
                )
                min_diversity = min(min_diversity, diversity)

            # Combine quality score with diversity
            combined_score = candidate["score"] * 0.6 + min_diversity * 0.4

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_candidate = candidate

        if best_candidate:
            selected.append(best_candidate)
            remaining.remove(best_candidate)

    # If there are fewer than 3 candidates, try to synthesise variation options
    if len(selected) < 3:
        print(f"    [VARIANT GEN] {target_char} × {scenario_id}: have {len(selected)} candidates; trying to synthesise {3-len(selected)} variants")
        try:
            generated_options = generate_variant_distractors(
                target_char, scenario_id, correct_decision,
                3-len(selected), character_traits, client
            )
            for gen_option in generated_options:
                selected.append(gen_option)
        except Exception as e:
            print(f"    [VARIANT GEN FAILED] {e}")

    if len(selected) >= 3:
        return selected[:3], "success"

    return selected, f"not enough candidates: only {len(selected)} valid"

def process_single_gt_pair(args):
    """Process one GT pair — used as the concurrent worker function."""
    gt_pair, scenario_annotations, character_traits, api_key, api_base = args

    client = OpenAI(api_key=api_key, base_url=api_base)

    char_id = gt_pair['character_id']
    scenario_id = gt_pair['scenario_id']
    correct_decision = gt_pair['final_decision']
    gt_source = gt_pair['gt_source']

    # Collect all character annotations for this scenario
    all_candidates = scenario_annotations.get(scenario_id, [])

    # Select the best distractors
    selected_distractors, status = select_best_distractors(
        char_id, scenario_id, correct_decision, all_candidates, character_traits, client
    )

    if status == "success" and len(selected_distractors) == 3:
        # Build the option
        options = [
            {
                "label": "A",
                "content": correct_decision,
                "is_correct": True,
                "source_character": char_id,
                "source_model": gt_source
            }
        ]

        labels = ["B", "C", "D"]
        for i, distractor in enumerate(selected_distractors):
            option = {
                "label": labels[i],
                "content": distractor["decision"],
                "is_correct": False,
                "source_character": distractor["character_id"],
                "quality_score": distractor["score"],
                "quality_details": distractor["details"]
            }
            # If the option is synthetic, tag it
            if distractor.get("is_generated", False):
                option["is_generated"] = True
                option["source_model"] = "claude-sonnet-4-6-generator"

            options.append(option)

        # Shuffle option order
        random.shuffle(options)

        # Re-label and locate the correct answer
        correct_label = None
        for i, option in enumerate(options):
            option["label"] = chr(65 + i)  # A, B, C, D
            if option["is_correct"]:
                correct_label = option["label"]

        avg_quality = sum(d["score"] for d in selected_distractors) / len(selected_distractors)

        question = {
            "question_id": f"Q_{char_id}_{scenario_id}",
            "character_id": char_id,
            "scenario_id": scenario_id,
            "stage": gt_pair.get('stage'),
            "correct_answer": correct_label,
            "options": options,
            "distractor_quality": {
                "avg_score": avg_quality,
                "score_std": (sum((d["score"] - avg_quality) ** 2 for d in selected_distractors) / len(selected_distractors)) ** 0.5,
                "selected_characters": [d["character_id"] for d in selected_distractors],
                "min_score": min(d["score"] for d in selected_distractors),
                "max_score": max(d["score"] for d in selected_distractors)
            }
        }

        return ("success", question, avg_quality)
    else:
        # Record the failure case
        failed_case = {
            "character_id": char_id,
            "scenario_id": scenario_id,
            "reason": status,
            "available_candidates": len(all_candidates),
            "selected_count": len(selected_distractors)
        }
        return ("failed", failed_case, None)


def generate_mcq_dataset():
    """Generate the full multiple-choice question dataset."""

    # Initialise API config
    api_key = os.getenv("ANNOTATE_API_KEY")
    if not api_key:
        raise RuntimeError("ANNOTATE_API_KEY is not set")
    api_base = os.getenv("ANNOTATE_API_BASE")

    print("Loading data...", file=sys.stderr)
    final_gt = load_final_gt()
    scenario_annotations = load_all_annotations()  # Get the scenario-keyed data directly
    character_traits = load_character_profiles()

    print(f"Loaded annotations for {len(scenario_annotations)} scenarios", file=sys.stderr)

    # Validate the data
    total_annotations = sum(len(annotations) for annotations in scenario_annotations.values())
    print(f"Total annotations across all scenarios: {total_annotations}", file=sys.stderr)

    mcq_dataset = {
        "dataset_meta": {
            "version": "MCQ_Dataset_v1.1_LLM_similarity",
            "description": "Multiple choice questions using LLM-based similarity judgment",
            "source_gt": "final_gt_annotations.json",
            "selection_algorithm": "claude_sonnet_4_6_similarity_judgment",
            "total_questions": 0,
            "successful_questions": 0,
            "failed_questions": 0
        },
        "questions": [],
        "statistics": {
            "distractor_selection_stats": defaultdict(int),
            "failed_cases": [],
            "quality_distribution": []
        }
    }

    print(f"Processing all GT pairs using Claude Sonnet 4.6 for similarity judgment with 16 workers...", file=sys.stderr)

    # Prepare arguments for concurrent processing
    args_list = []
    for gt_pair in final_gt['pairs']:  # Process every pair
        args_list.append((gt_pair, scenario_annotations, character_traits, api_key, api_base))

    print(f"Starting concurrent processing of {len(args_list)} GT pairs", file=sys.stderr)

    # Concurrent processing
    from concurrent.futures import ThreadPoolExecutor, as_completed
    successful_count = 0
    failed_count = 0

    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=len(args_list), desc="Processing MCQ pairs")
    except ImportError:
        pbar = None

    with ThreadPoolExecutor(max_workers=16) as executor:
        # Submit every task
        future_to_index = {executor.submit(process_single_gt_pair, args): i for i, args in enumerate(args_list)}

        # Collect results
        for future in as_completed(future_to_index):
            try:
                result_type, result_data, avg_quality = future.result()

                if result_type == "success":
                    mcq_dataset["questions"].append(result_data)
                    if avg_quality is not None:
                        mcq_dataset["statistics"]["quality_distribution"].append(avg_quality)
                    successful_count += 1
                    mcq_dataset["statistics"]["distractor_selection_stats"]["successful"] += 1

                elif result_type == "failed":
                    mcq_dataset["statistics"]["failed_cases"].append(result_data)
                    failed_count += 1
                    mcq_dataset["statistics"]["distractor_selection_stats"]["failed"] += 1

                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_description(f"Processing MCQ pairs (Success: {successful_count}, Failed: {failed_count})")
                else:
                    processed = successful_count + failed_count
                    if processed % 50 == 0:
                        print(f"Processed {processed}/{len(args_list)} pairs (Success: {successful_count}, Failed: {failed_count})", file=sys.stderr)

            except Exception as e:
                print(f"Error processing GT pair: {e}", file=sys.stderr)
                failed_count += 1
                if pbar is not None:
                    pbar.update(1)

    if pbar is not None:
        pbar.close()

    print(f"Completed processing. Success: {successful_count}, Failed: {failed_count}", file=sys.stderr)

    # Update aggregate stats
    mcq_dataset["dataset_meta"]["total_questions"] = len(final_gt['pairs'])
    mcq_dataset["dataset_meta"]["successful_questions"] = len(mcq_dataset["questions"])
    mcq_dataset["dataset_meta"]["failed_questions"] = len(mcq_dataset["statistics"]["failed_cases"])

    return mcq_dataset

def main():
    """Main entry point."""

    # Seed the RNG for reproducibility
    random.seed(42)

    print("Generating MCQ dataset with intelligent distractor selection...", file=sys.stderr)
    print("Using basic Python libraries only (no external dependencies)", file=sys.stderr)

    mcq_dataset = generate_mcq_dataset()

    # Save the result
    output_path = Path("benchmark/mcq/mcq_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mcq_dataset, f, ensure_ascii=False, indent=2)

    # Print summary stats
    total = mcq_dataset["dataset_meta"]["total_questions"]
    successful = mcq_dataset["dataset_meta"]["successful_questions"]
    failed = mcq_dataset["dataset_meta"]["failed_questions"]

    print("\n=== MCQ DATASET GENERATION SUMMARY ===", file=sys.stderr)
    print(f"Total GT pairs: {total}", file=sys.stderr)
    print(f"Successful MCQ questions: {successful}", file=sys.stderr)
    print(f"Failed cases: {failed}", file=sys.stderr)
    print(f"Success rate: {successful/total*100:.1f}%", file=sys.stderr)

    if mcq_dataset["statistics"]["quality_distribution"]:
        quality_scores = mcq_dataset["statistics"]["quality_distribution"]
        avg_quality = sum(quality_scores) / len(quality_scores)
        max_quality = max(quality_scores)
        min_quality = min(quality_scores)
        print(f"\nDistractor Quality Analysis:", file=sys.stderr)
        print(f"  Average quality score: {avg_quality:.3f}", file=sys.stderr)
        print(f"  Quality range: {min_quality:.3f} - {max_quality:.3f}", file=sys.stderr)

    # Analyse failure causes
    if mcq_dataset["statistics"]["failed_cases"]:
        print("\n=== FAILURE ANALYSIS ===", file=sys.stderr)
        failure_reasons = defaultdict(int)
        for case in mcq_dataset["statistics"]["failed_cases"]:
            failure_reasons[case["reason"]] += 1

        for reason, count in failure_reasons.items():
            print(f"  {reason}: {count} cases", file=sys.stderr)

    print(f"\nMCQ dataset saved to: {output_path}", file=sys.stderr)

    return 0

if __name__ == "__main__":
    sys.exit(main())