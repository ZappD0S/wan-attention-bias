import json
import logging
import os
import textwrap
import time

from openai import OpenAI

logging.basicConfig(
    level=logging.DEBUG,  # change to logging.INFO to hide the raw prompt/response logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
BASE_URL = "https://api.z.ai/api/coding/paas/v4"
MODEL_NAME = "glm-5.1"
METAPROMPT_PATH = "image_prompt_generation/metaprompt.md"
TARGET_COUNT = 200
BATCH_SIZE = 10
MAX_AUDIT_PASSES = 5
OUTPUT_PATH = "image_prompt_generation/final_dataset.json"  # Centralized path

client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)


def generate_seeds(needed, existing_dataset):
    used_subjects = []
    for scene in existing_dataset:
        try:
            used_subjects.append(scene["appearance_prompt"]["segments"][0][1])
        except (KeyError, IndexError):
            continue

    used_context = "\n".join(used_subjects) if used_subjects else "None yet."

    seed_prompt = textwrap.dedent(f"""\
    You are a curator for a high-fidelity video dataset.
    Generate a list of exactly {needed} archetypal and easily recognizable subject concepts.

    GOAL: 
    Select subjects that have a strong, frequent presence in visual training data (photography, film, and media). 
    Include a balanced mix of "Real-world" and "Fiction" subjects. 
    Focus on well-known archetypes that are visually distinct and easy to recognize. 
    Avoid "mash-ups," bizarre hybrids, or overly complex descriptions that are rare in common data.

    DIVERSITY: 
    50/50 split between ANIMATE (people, creatures, animals) and INANIMATE (objects, vehicles, tools).

    CONSTRAINTS:
    1. Each concept MUST feature exactly TWO identical or highly similar subjects.
    2. Keep the subject descriptions simple and iconic.
    3. Return ONLY raw text. NO bolding, NO numbering, NO headers, and NO action descriptions.

    FORMAT:
    Theme | Two [Subjects]

    EXAMPLES:
    Fiction | Two armored knights
    Real-world | Two yellow excavators
    Fiction | Two humanoid androids
    Real-world | Two grizzly bears

    DO NOT USE THESE SUBJECTS:
    {used_context}""")

    logger.debug(f"--- SEED PROMPT ---\n{seed_prompt}\n")

    try:
        seed_res = client.chat.completions.create(
            model=MODEL_NAME, messages=[{"role": "user", "content": seed_prompt}]
        )

        raw_output = seed_res.choices[0].message.content
        logger.debug(f"--- SEED RESPONSE ---\n{raw_output}\n")

        lines = raw_output.strip().split("\n")
        # Only grab valid lines that follow the Theme | Subject format
        seeds = [line.replace("*", "").strip() for line in lines if "|" in line]
        return seeds[:needed]
    except Exception as e:
        logger.error(f"Error generating seeds: {e}")
        return []


def generate_batch(seeds_batch, metaprompt):
    dynamic_metaprompt = metaprompt.replace(
        "10 distinct scenes", f"{len(seeds_batch)} distinct scenes"
    )

    seeds_text = "\n".join(seeds_batch)

    user_prompt = textwrap.dedent(f"""\
    Process these {len(seeds_batch)} subjects into JSON format.

    SUBJECTS:
    {seeds_text}""")

    logger.debug(f"--- BATCH USER PROMPT ---\n{user_prompt}\n")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": dynamic_metaprompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content
        logger.debug(f"--- BATCH RESPONSE ---\n{raw_output}\n")

        return json.loads(raw_output).get("dataset", [])
    except Exception as e:
        logger.error(f"Error during JSON generation: {e}")
        return []


def audit_redundancies(dataset):
    logger.info("Starting audit for 'soft' redundancies...")
    summaries = []
    for idx, item in enumerate(dataset):
        try:
            sub = item["appearance_prompt"]["segments"][0][1]
            act_a = item["action_prompts"]["split_sentences"]["segments"][0][0]
            act_b = item["action_prompts"]["split_sentences"]["segments"][1][0]
            summaries.append(f"ID {idx}: {sub} | {act_a} | {act_b}")
        except (KeyError, IndexError):
            continue

    joint_summaries = "\n".join(summaries)
    audit_prompt = f"""Analyze these {len(summaries)} scenes for "soft" redundancies (conceptually similar subjects, actions, or materials).

    REDUNDANCY EXAMPLES:
    - Similar species performing similar actions.
    - Different objects made of the same distinct material (e.g., both "brushed gold").
    - Actions with the same visual outcome (e.g., "dissolving" vs "melting").

    OUTPUT: 
    Return a JSON object: {{"remove_ids": [ids]}}.
    IMPORTANT: If the dataset is diverse and no redundancies exist, you MUST return: {{"remove_ids": []}}.

    SCENES:
    {joint_summaries}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": audit_prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("remove_ids", [])
    except Exception as e:
        logger.error(f"Audit failed: {e}")
        return []


def ensure_dataset_full(dataset, target, metaprompt, batch_size):
    while len(dataset) < target:
        needed_now = min(batch_size, target - len(dataset))
        logger.info(f"Progress: {len(dataset)}/{target}. Requesting {needed_now} new scenes...")

        seeds = generate_seeds(needed_now, dataset)
        if len(seeds) >= needed_now:
            batch = generate_batch(seeds, metaprompt)
            dataset.extend(batch)
            save_dataset(dataset)  # Save progress immediately
            logger.info(f"Batch saved. Current count: {len(dataset)}")
        else:
            logger.warning(f"Could only generate {len(seeds)} seeds. Retrying...")
            time.sleep(2)

    return dataset


def save_dataset(dataset):
    output = {
        "safeguard_suffix": "The scene is filmed as a continuous shot with a static camera, ensuring no cuts and no new objects entering.",
        "dataset": dataset,
    }
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    try:
        with open(METAPROMPT_PATH, encoding="utf-8") as f:
            metaprompt = f.read()
    except FileNotFoundError:
        logger.error(f"Metaprompt file not found at path: {METAPROMPT_PATH}")
        return

    dataset = []
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, encoding="utf-8") as f:
                existing_data = json.load(f)
                dataset = existing_data.get("dataset", [])
                logger.info(f"Resuming: Loaded {len(dataset)} existing scenes from {OUTPUT_PATH}")
        except Exception as e:
            logger.error(f"Error loading existing dataset: {e}. Starting fresh.")

    pass_count = 0
    while pass_count < MAX_AUDIT_PASSES:
        pass_count += 1
        logger.info(f"=== AUDIT PASS {pass_count} ===")

        dataset = ensure_dataset_full(dataset, TARGET_COUNT, metaprompt, BATCH_SIZE)

        remove_ids = audit_redundancies(dataset)

        if not remove_ids:
            logger.info("No redundancies found. Dataset is optimal.")
            break

        logger.warning(f"Removing {len(remove_ids)} redundant scenes.")
        for idx in sorted(remove_ids, reverse=True):
            if idx < len(dataset):
                dataset.pop(idx)

        save_dataset(dataset)

    final_dataset = dataset[:TARGET_COUNT]
    save_dataset(final_dataset)
    logger.info(f"Process complete. Total scenes: {len(final_dataset)}")


if __name__ == "__main__":
    main()
