import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from scipy.stats import binomtest
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from .sam2_pipeline import run_sam2_pipeline


def load_qwen2_5_model() -> tuple[Any, Any]:
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def load_qwen3_model() -> tuple[Any, Any]:
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def generate_qwen_messages(processor, prompt: str, video_path: Path | None):
    message_content = [
        {"type": "text", "text": prompt},
    ]

    if video_path is not None:
        if not video_path.exists():
            return "Error: Video file not found."

        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Qwen prefers an even number of frames for its temporal patches (stride 2)
        nframes_to_use = (total_frames // 2) * 2

        cap.release()

        message_content += [
            {
                "type": "video",
                "video": str(video_path),
                "nframes": nframes_to_use,
                # "min_frames": nframes_to_use,
                # "max_frames": nframes_to_use,
            }
        ]

    messages = [
        {
            "role": "user",
            "content": message_content,
        }
    ]

    return messages


def run_qwen_generation(model, processor, prompt: str, video_path: Path | None) -> tuple[str, str]:
    messages = generate_qwen_messages(processor, prompt, video_path)
    input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)  # ty:ignore[invalid-assignment]

    inputs = processor(
        text=[input_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=2048)
        # generated_ids = model.generate(**inputs)

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return input_text, output_text[0]


def get_next_token_logits(model, processor, prompt, video_path: Path | None):
    messages = generate_qwen_messages(processor, prompt, video_path)
    image_inputs, video_inputs = process_vision_info(messages)  # ty:ignore[invalid-assignment]
    inputs = processor(
        text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits[0, -1, :]


def get_soft_score_blind(model, processor, video_path: Path, action_descr: str) -> float:

    prompt_description = """You are a forensic video analyst.
    Provide a detailed, objective, chronological log of the video.

    Guidelines:
    1. Break the video down by visual changes and movements.
    2. Describe specific body parts, objects, and interactions.
    3. Do NOT interpret intent or purpose.
    4. Focus purely on visual observables.

    Output the log now."""

    _, video_description = run_qwen_generation(model, processor, prompt_description, video_path)

    prompt_audit_template = """You are a strict action auditor. 
    Your task is to rate how well the 'Target Action' matches the 'Video Description'.

    Target Action: {target_action}
    Video Description: {video_description}

    Reasoning Rules:
    - Rely ONLY on the Video Description provided above. 
    - Be skeptical: if a specific detail is missing from the description, assume it did not happen.
    - Check for chronological consistency.

    Output a brief step-by-step reasoning."""

    prompt_audit = prompt_audit_template.format(
        target_action=action_descr, video_description=video_description
    )

    # Get the reasoning text (we pass video_path=None because we rely on the text description now)
    input_prompt_with_history, reasoning = run_qwen_generation(
        model, processor, prompt_audit, video_path=None
    )
    prompt_score = """
    Based on the reasoning above, assign a match score on a scale of 1 to 5.

    Criteria:
    5: Perfect Match (Unambiguous, clearly main focus).
    4: Strong Match (Main event, minor noise).
    3: Partial Match (Action occurred but mixed with others).
    2: Weak Match (Ambiguous or minor detail).
    1: No Match (Action not found or different action).

    Score: """

    forced_text = input_prompt_with_history + reasoning + prompt_score

    target_tokens = ["1", "2", "3", "4", "5"]

    target_ids = [processor.tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens]

    score_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=model.device)

    next_token_logits = get_next_token_logits(model, processor, forced_text, video_path=None)

    top_token_id = torch.argmax(next_token_logits).item()
    assert top_token_id in target_ids
    top_token_text = processor.tokenizer.decode([top_token_id])
    assert top_token_id in target_ids, f"most likely next token: '{top_token_text}'"

    score_logits = next_token_logits[target_ids]
    probs = torch.softmax(score_logits, dim=0)
    continuous_score = torch.sum(probs * score_values).item()

    return continuous_score


def get_soft_score_direct(model, processor, video_path, action_descr: str) -> float:
    prompt_template = """You are a strict video auditor. 
    Analyze the video content and determine if the following action occurs.

    Target Action: {target_action}

    Provide a step-by-step reasoning based on the visual evidence.
    Conclude by evaluating how well the video matches the action."""

    prompt = prompt_template.format(target_action=action_descr)

    input_prompt_with_history, reasoning = run_qwen_generation(model, processor, prompt, video_path)

    prompt_score = """
    Based on your reasoning, assign a match score on a scale of 1 to 5.

    Criteria:
    5: Perfect Match (Action is clearly the main focus).
    4: Strong Match (Action occurs, minor noise).
    3: Partial Match (Action is part of a larger sequence).
    2: Weak Match (Ambiguous or hard to see).
    1: No Match (Action does not happen).

    Score: """

    forced_text = input_prompt_with_history + reasoning + prompt_score

    target_tokens = ["1", "2", "3", "4", "5"]
    target_ids = [processor.tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens]
    target_ids_tensor = torch.tensor(target_ids, device=model.device)
    score_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=model.device)

    next_token_logits = get_next_token_logits(model, processor, forced_text, video_path=video_path)

    top_token_id = torch.argmax(next_token_logits).item()
    top_token_text = processor.tokenizer.decode([top_token_id])
    assert top_token_id in target_ids, f"most likely next token: '{top_token_text}'"

    score_logits = next_token_logits[target_ids_tensor]
    probs = torch.softmax(score_logits, dim=0)
    continuous_score = torch.sum(probs * score_values).item()

    return continuous_score


def get_discrete_score_blind(model, processor, video_path, action_descr: str) -> float:
    # STEP 1: Identical to Reference
    prompt_description = """You are a forensic video analyst.
    Provide a detailed, objective, chronological log of the video.

    Guidelines:
    1. Break the video down by visual changes and movements.
    2. Describe specific body parts, objects, and interactions.
    3. Do NOT interpret intent or purpose.
    4. Focus purely on visual observables.

    Output the log now."""

    _, video_description = run_qwen_generation(model, processor, prompt_description, video_path)

    # STEP 2: Merged Audit + Scoring instructions
    prompt_audit_template = """You are a strict action auditor. 
    Your task is to rate how well the 'Target Action' matches the 'Video Description'.

    Target Action: {target_action}
    Video Description: {video_description}

    Reasoning Rules:
    - Rely ONLY on the Video Description provided above. 
    - Be skeptical: if a specific detail is missing from the description, assume it did not happen.
    - Check for chronological consistency.

    Scoring Criteria:
    5: Perfect Match (Unambiguous, clearly main focus).
    4: Strong Match (Main event, minor noise).
    3: Partial Match (Action occurred but mixed with others).
    2: Weak Match (Ambiguous or minor detail).
    1: No Match (Action not found or different action).

    Instructions:
    1. Output a brief step-by-step reasoning.
    2. End your response strictly with: "Score: X" (where X is 1-5)."""

    prompt_audit = prompt_audit_template.format(
        target_action=action_descr, video_description=video_description
    )

    _, response_text = run_qwen_generation(model, processor, prompt_audit, video_path=None)

    return extract_score(response_text)


def get_discrete_score_direct(model, processor, video_path, action_descr: str) -> float:
    prompt_audit_template = """You are a strict action auditor and forensic video analyst.
    Your task is to determine if the 'Target Action' occurs in the video based on visual evidence.

    Target Action: {target_action}

    Reasoning Rules:
    - Focus purely on visual observables (movements, contacts, states).
    - Do NOT interpret intent or purpose.
    - Be skeptical: if the specific visual details are missing, assume it did not happen.

    Scoring Criteria:
    5: Perfect Match (Unambiguous, clearly main focus).
    4: Strong Match (Main event, minor noise).
    3: Partial Match (Action occurred but mixed with others).
    2: Weak Match (Ambiguous or minor detail).
    1: No Match (Action not found or different action).

    Instructions:
    1. Output a brief step-by-step reasoning based on the visual evidence.
    2. End your response strictly with: "Score: X" (where X is 1-5)."""

    prompt_audit = prompt_audit_template.format(target_action=action_descr)

    # Generate full text (Model sees video)
    _, response_text = run_qwen_generation(model, processor, prompt_audit, video_path=video_path)

    return extract_score(response_text)


def extract_score(model_output: str) -> int:
    if not model_output:
        raise ValueError("Model output is empty.")

    # 1. Primary Regex: Looks for "Score" followed by a number.
    # Matches: "Score: 5", "**Score**: 5", "Score - 5", "Score: 5/5"
    # flags=re.IGNORECASE makes it work even if model outputs "score: 5"
    match = re.search(r"Score\**\s*[:\-]?\s*(\d)", model_output, re.IGNORECASE)

    if match:
        score = int(match.group(1))
        # Clamp value just in case model hallucinates a 0 or 6
        return max(1, min(5, score))

    # Matches: "5", "5\nReason:..."
    # 2. Fallback: Look for a number at the very start of the string
    match_start = re.match(r"^\s*(\d)", model_output)
    if match_start:
        score = int(match_start.group(1))
        return max(1, min(5, score))

    # 3. Last Resort: Look for "X/5" anywhere in text
    # Matches: "I give this a 5/5"
    match_fraction = re.search(r"(\d)\s*/\s*5", model_output)
    if match_fraction:
        score = int(match_fraction.group(1))
        return max(1, min(5, score))

    raise ValueError(f"Could not extract a valid integer score from output: {model_output}")


def evaluate_pipeline(
    scoring_func,
    model,
    processor,
    videos_dir: Path,
    output_dir: Path,
    sam_model_id: str,
    target_fps: int,
    sam_margin: float,
):

    valid_dirs = [d for d in videos_dir.iterdir() if d.is_dir() and d.name != "debug"]
    total_calls = 0
    for video_dir in valid_dirs:
        with open(video_dir / "config.json") as f:
            data = json.load(f)
            num_objs = len(data["prompt_data"]["bboxes"])
            num_actions = len(data["prompt_data"]["action_prompts"]["split_sentences"]["segments"])
            total_calls += num_objs * num_actions

    margins = []
    correct_count: int = 0
    total_count: int = 0
    pbar = tqdm(total=total_calls, desc="Evaluating Actions", unit="call")

    for video_dir in valid_dirs:
        with open(video_dir / "config.json") as f:
            json_data = json.load(f)

        # load bboxes
        bboxes = json_data["prompt_data"]["bboxes"]
        current_output_dir = output_dir / video_dir.name
        current_output_dir.mkdir(exist_ok=True, parents=True)

        clean_videos = run_sam2_pipeline(
            video_dir / "video_0.mp4",
            current_output_dir,
            bboxes,
            sam_model_id,
            target_fps,
            sam_margin,
        )

        segments = json_data["prompt_data"]["action_prompts"]["split_sentences"]["segments"]
        action_descriptions = [s[0] for s in segments]
        # NOTE: the order of bboxes and action_descriptions is assumed to be the same

        for j, video_path in enumerate(clean_videos):
            margin: float = 0.0
            scores = {}
            for k, action_descr in enumerate(action_descriptions):
                is_correct = j == k
                sign = 1 if is_correct else -1

                score = scoring_func(model, processor, video_path, action_descr)

                margin += sign * score
                scores[is_correct] = score
                pbar.update(1)

            margins.append(margin)

            total_count += 1

            if scores[True] > scores[False]:
                correct_count += 1

    result = binomtest(correct_count, total_count, p=0.5, alternative="greater")
    print(f"Discrimination rate: {correct_count / total_count:.2f}")
    print(f"p-value: {result.pvalue:.4f}")
    print(f"95% CI: {result.proportion_ci(confidence_level=0.95)}")

    margins_arr = np.array(margins)
    boot_means = np.array(
        [
            np.random.choice(margins_arr, size=margins_arr.size, replace=True).mean()
            for _ in range(10_000)
        ]
    )

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"Mean margin: {margins_arr.mean():.3f} (95% CI: {ci_low:.3f}, {ci_high:.3f})")


def main():
    # --- Config ---
    OUTPUT_DIR = Path("./evaluation/output/")
    VIDEOS_DIR = Path("./multi_sample_inference/debug_output/")

    MARGIN, TARGET_FPS = 0.2, 16

    SAM_MODEL_ID = "facebook/sam2-hiera-large"

    # load Qwen model
    qwen_model, qwen_processor = load_qwen3_model()

    score_funcs = [
        get_soft_score_direct,
        get_soft_score_blind,
        get_discrete_score_blind,
        get_discrete_score_direct,
    ]

    for score_func in score_funcs:
        print(f"Score function: {score_func.__name__}")
        evaluate_pipeline(
            score_func,
            qwen_model,
            qwen_processor,
            VIDEOS_DIR,
            OUTPUT_DIR,
            SAM_MODEL_ID,
            TARGET_FPS,
            MARGIN,
        )


if __name__ == "__main__":
    main()
