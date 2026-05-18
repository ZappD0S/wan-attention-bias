import argparse
import json
import os
import random
import re
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from rich.console import Console
from rich.table import Table
from scipy.stats import binomtest
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from .sam2_pipeline import run_sam2_pipeline
from .utils import DataEntry, group_entries


class QwenEngine:
    """Manages the model state and handles generation and logit extraction."""

    def __init__(self, model_id: str, is_qwen3: bool = True):
        model_class = (
            Qwen3VLForConditionalGeneration if is_qwen3 else Qwen2_5_VLForConditionalGeneration
        )
        self.model = model_class.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = self.model.device

    def _get_nframes(self, video_path: Path | None) -> int | None:
        if video_path is None or not video_path.exists():
            return None
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return (total_frames // 2) * 2

    def generate(
        self, prompt: str, system_prompt: str | None = None, video_path: Path | None = None
    ) -> str:
        nframes = self._get_nframes(video_path)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if video_path:
            content.append({"type": "video", "video": str(video_path), "nframes": nframes})

        messages = [{"role": "user", "content": content}]

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        input_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True
        )

        if video_inputs:
            video_inputs, video_metadatas = zip(*video_inputs, strict=True)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=[input_text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,  # ty:ignore[invalid-argument-type]
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=2048,
                temperature=None,
                top_p=None,
                top_k=None,
            )  # ty:ignore[invalid-argument-type]

        trimmed_ids = [
            out[len(ins) :] for ins, out in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        output_text = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def get_scoring_logits(
        self, prompt1: str, reasoning: str, prompt2: str, video_path: Path | None = None
    ):
        nframes = self._get_nframes(video_path)
        msg1_content: list[dict[str, Any]] = [{"type": "text", "text": prompt1}]

        if video_path:
            msg1_content.append({"type": "video", "video": str(video_path), "nframes": nframes})

        messages = [
            {"role": "user", "content": msg1_content},
            {"role": "assistant", "content": [{"type": "text", "text": reasoning}]},
            {"role": "user", "content": [{"type": "text", "text": prompt2}]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_in, vid_in, vid_kw = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True
        )

        if vid_in:
            vid_in, vid_metas = zip(*vid_in, strict=True)
            vid_in, vid_metas = list(vid_in), list(vid_metas)
        else:
            vid_metas = None

        inputs = self.processor(
            text=[text],
            images=img_in,
            videos=vid_in,
            video_metadata=vid_metas,
            **vid_kw,  # ty:ignore[invalid-argument-type]
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits[0, -1, :]

    def calculate_soft_score(
        self, logits: torch.Tensor, target_tokens: list[str], target_weights: list[float]
    ) -> float:
        target_ids = [
            self.processor.tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens
        ]

        full_probs = torch.softmax(logits, dim=-1)
        target_probs = full_probs[torch.tensor(target_ids, device=self.device)]

        total_target_prob = target_probs.sum().item()
        if total_target_prob < 0.5:
            top_token_id = torch.argmax(logits).item()
            top_token_text = repr(self.processor.tokenizer.decode([top_token_id]))
            print(
                f"\n[WARNING] Soft score math breaking! Model wants {top_token_text}. Mass: {total_target_prob:.3f}"
            )

        normalized_probs = target_probs / (target_probs.sum() + 1e-9)
        score_values = torch.tensor(target_weights, device=self.device)
        return torch.sum(normalized_probs * score_values).item()


class VideoAsset:
    """Represents a video segment. Lazily generates a description only if needed."""

    def __init__(self, path: Path, engine: QwenEngine):
        self.path = path
        self.engine = engine
        self._description: str | None = None

    @property
    def description(self) -> str:
        if self._description is None:
            prompt = textwrap.dedent("""\
                You are a forensic video analyst.
                Provide a detailed, objective, chronological log of the video.

                Guidelines:
                1. Break the video down by visual changes and movements.
                2. Describe specific body parts, objects, and interactions.
                3. Do NOT interpret intent or purpose.
                4. Focus purely on visual observables.

                Output the log now.""")
            self._description = self.engine.generate(prompt, video_path=self.path)
        return self._description


class ActionAuditor(ABC):
    def __init__(self, engine: QwenEngine):
        self.engine = engine

    @abstractmethod
    def _score(self, video: VideoAsset, action: str) -> float:
        pass

    def score_pair(self, video: VideoAsset, action_a: str, action_b: str) -> float:
        score_a = self._score(video, action_a)
        score_b = self._score(video, action_b)
        return score_a - score_b

    @staticmethod
    def _extract_score(model_output: str) -> int:
        if not model_output:
            raise ValueError("Model output is empty.")
        match = re.search(r"Score\**\s*[:\-]?\s*(\d)", model_output, re.IGNORECASE)
        if match:
            return max(1, min(5, int(match.group(1))))
        match_start = re.match(r"^\s*(\d)", model_output)
        if match_start:
            return max(1, min(5, int(match_start.group(1))))
        match_fraction = re.search(r"(\d)\s*/\s*5", model_output)
        if match_fraction:
            return max(1, min(5, int(match_fraction.group(1))))
        raise ValueError(f"Could not extract score from: {model_output}")


class SoftDirectAuditor(ActionAuditor):
    def _score(self, video: VideoAsset, action: str) -> float:
        prompt1 = textwrap.dedent(f"""\
            You are a strict video auditor.
            Analyze the video content and determine if the following action occurs.

            Target Action: {action}

            Provide a step-by-step reasoning based on the visual evidence.
            Conclude by evaluating how well the video matches the action.""")

        reasoning = self.engine.generate(prompt1, video_path=video.path)

        prompt2 = textwrap.dedent("""\
            Based on your reasoning, assign a match score on a scale of 1 to 5.

            Criteria:
            5: Perfect Match (Action is clearly the main focus).
            4: Strong Match (Action occurs, minor noise).
            3: Partial Match (Action is part of a larger sequence).
            2: Weak Match (Ambiguous or hard to see).
            1: No Match (Action does not happen).

            CRITICAL INSTRUCTION: Output ONLY a single integer (1, 2, 3, 4, or 5). Do not output any words, punctuation, or spaces.""")

        logits = self.engine.get_scoring_logits(prompt1, reasoning, prompt2, video.path)
        return self.engine.calculate_soft_score(
            logits,
            target_tokens=["1", "2", "3", "4", "5"],
            target_weights=[1.0, 2.0, 3.0, 4.0, 5.0],
        )


class SoftTwoAFCAuditor(ActionAuditor):
    def _score(self, video: VideoAsset, action: str) -> float:
        raise NotImplementedError(
            "SoftTwoAFCAuditor is a pairwise evaluator and cannot score a single action. "
            "Please call `score_pair(video, action_a, action_b)` instead."
        )

    def score_pair(self, video: VideoAsset, action_a: str, action_b: str) -> float:
        prompt1 = textwrap.dedent(f"""\
            You are an expert video evaluator.
            I will provide you with a short video clip and two possible descriptions of the action occurring in the video.

            Option A: {action_a}
            Option B: {action_b}

            Your task is to determine which of the two options best describes the action and movement being performed by the character in the video.

            CRITICAL INSTRUCTIONS:
            1. Focus strictly on the action, movement, and physical interactions.
            2. Do NOT base your decision solely on the character's clothing, background, or static objects. Focus on what the character is doing.
            3. You MUST choose either Option A or Option B.

            Provide a step-by-step reasoning based on the visual evidence comparing the two options.""")

        reasoning = self.engine.generate(prompt1, video_path=video.path)

        prompt2 = textwrap.dedent("""\
            Based on your reasoning, choose the option that best matches the video.

            CRITICAL INSTRUCTION: Output ONLY a single uppercase letter ("A" or "B"). Do not output any words, punctuation, or spaces.""")

        logits = self.engine.get_scoring_logits(prompt1, reasoning, prompt2, video.path)

        prob_a = self.engine.calculate_soft_score(
            logits, target_tokens=["A", "B"], target_weights=[1.0, 0.0]
        )
        prob_b = 1.0 - prob_a

        # Return the margin.
        # If prob_a is 0.9 and prob_b is 0.1, it returns 0.8.
        return prob_a - prob_b


class SoftBlindAuditor(ActionAuditor):
    def _score(self, video: VideoAsset, action: str) -> float:
        prompt1 = textwrap.dedent(f"""\
            You are a strict action auditor.
            Your task is to rate how well the 'Target Action' matches the 'Video Description'.

            Target Action: {action}
            Video Description: {video.description}

            Reasoning Rules:
            - Rely ONLY on the Video Description provided above. 
            - Be skeptical: if a specific detail is missing from the description, assume it did not happen.
            - Check for chronological consistency.

            Output a brief step-by-step reasoning.""")

        reasoning = self.engine.generate(prompt1, None)

        prompt2 = textwrap.dedent("""\
            Based on the reasoning above, assign a match score on a scale of 1 to 5.

            Criteria:
            5: Perfect Match (Unambiguous, clearly main focus).
            4: Strong Match (Main event, minor noise).
            3: Partial Match (Action occurred but mixed with others).
            2: Weak Match (Ambiguous or minor detail).
            1: No Match (Action not found or different action).

            CRITICAL INSTRUCTION: Output ONLY a single integer (1, 2, 3, 4, or 5). Do not output any words, punctuation, or spaces.""")

        logits = self.engine.get_scoring_logits(prompt1, reasoning, prompt2, None)
        return self.engine.calculate_soft_score(
            logits,
            target_tokens=["1", "2", "3", "4", "5"],
            target_weights=[1.0, 2.0, 3.0, 4.0, 5.0],
        )


class DiscreteDirectAuditor(ActionAuditor):
    def _score(self, video: VideoAsset, action: str) -> float:
        prompt = textwrap.dedent(f"""\
            You are a strict action auditor and forensic video analyst.
            Your task is to determine if the 'Target Action' occurs in the video based on visual evidence.

            Target Action: {action}

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
            2. End your response strictly with: "Score: X" (where X is 1-5).""")

        return self._extract_score(self.engine.generate(prompt, video_path=video.path))


class DiscreteBlindAuditor(ActionAuditor):
    def _score(self, video: VideoAsset, action: str) -> float:
        prompt = textwrap.dedent(f"""\
            You are a strict action auditor.
            Your task is to rate how well the 'Target Action' matches the 'Video Description'.

            Target Action: {action}
            Video Description: {video.description}

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
            2. End your response strictly with: "Score: X" (where X is 1-5).""")

        return self._extract_score(self.engine.generate(prompt, None))


class EvaluationPipeline:
    def __init__(
        self,
        videos_dir: Path,
        output_dir: Path,
        sam_model_id: str,
        target_fps: int,
        sam_margin: float,
    ):
        self.videos_dir = videos_dir
        self.output_dir = output_dir
        self.sam_model_id = sam_model_id
        self.target_fps = target_fps
        self.sam_margin = sam_margin

        # Precompute tasks and total calls once during initialization
        self.video_tasks, self.total_calls = self._prepare_tasks()

    def _prepare_tasks(self) -> tuple[dict[Path, list[Path]], int]:
        """Scans directories, parses configs, and calculates the exact number of evaluation calls."""
        valid_dirs = [d for d in self.videos_dir.iterdir() if d.is_dir() and d.name != "debug"]
        video_pattern = re.compile(r"video_\d+\.mp4")

        video_tasks: dict[Path, list[Path]] = {}
        total_calls = 0

        for video_dir in valid_dirs:
            video_paths = [p for p in video_dir.iterdir() if video_pattern.fullmatch(p.name)]
            video_tasks[video_dir] = video_paths

            with open(video_dir / "config.json") as f:
                data = json.load(f)
                num_videos = len(video_paths)
                num_crops = len(data["prompt_data"]["bboxes"])
                num_actions = len(
                    data["prompt_data"]["action_prompts"]["split_sentences"]["segments"]
                )

                total_calls += num_videos * num_crops * num_actions

        return video_tasks, total_calls

    def run(self, auditor: ActionAuditor):
        entries = []

        pbar = tqdm(
            total=self.total_calls, desc=f"Running {auditor.__class__.__name__}", unit="call"
        )

        for video_dir, video_paths in self.video_tasks.items():
            with open(video_dir / "config.json") as f:
                json_data = json.load(f)

            margins, correct_count, total_count = [], 0, 0
            bboxes, params = json_data["prompt_data"]["bboxes"], json_data["params"]
            segments = json_data["prompt_data"]["action_prompts"]["split_sentences"]["segments"]
            action_descriptions = [s[0] for s in segments]

            current_output_dir = self.output_dir / video_dir.name
            current_output_dir.mkdir(exist_ok=True, parents=True)

            for video_path in video_paths:
                cropped_videos = run_sam2_pipeline(
                    video_path,
                    current_output_dir,
                    bboxes,
                    self.sam_model_id,
                    self.target_fps,
                    self.sam_margin,
                )

                assert len(cropped_videos) == 2
                assert len(action_descriptions) == 2
                for j, crop_path in enumerate(cropped_videos):
                    video_asset = VideoAsset(crop_path, auditor.engine)

                    true_action = action_descriptions[j]
                    false_action = action_descriptions[1 - j]

                    margin = auditor.score_pair(video_asset, true_action, false_action)

                    margins.append(margin)
                    total_count += 1

                    # Positive margin means the correct prompt scored higher
                    if margin > 0:
                        correct_count += 1

                    pbar.update(2)

            entries.append(
                {
                    "params": params,
                    "values": {
                        "margins": margins,
                        "correct_count": correct_count,
                        "total_count": total_count,
                    },
                }
            )

        pbar.close()
        return self._aggregate_results(entries)

    def _aggregate_results(self, entries: list[DataEntry]) -> list[dict]:
        grouped_entries = group_entries(entries)
        output = []
        for grouped in grouped_entries:
            params, values = grouped["params"], grouped["values"]
            margins = [m for entry in values for m in entry["margins"]]
            correct = sum(entry["correct_count"] for entry in values)
            total = sum(entry["total_count"] for entry in values)
            output.append(
                {"params": params, "statistics": self.compute_statistics(correct, total, margins)}
            )
        return output

    @staticmethod
    def compute_statistics(correct_count, total_count, margins):
        binom_res = binomtest(correct_count, total_count, p=0.5, alternative="greater")
        binom_ci = binom_res.proportion_ci(confidence_level=0.95)

        m_arr = np.array(margins)
        boot_means = [
            np.random.choice(m_arr, size=m_arr.size, replace=True).mean() for _ in range(10_000)
        ]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        return {
            "margin": {"avg": m_arr.mean(), "ci": [ci_low, ci_high]},
            "discrimination": {
                "avg": binom_res.statistic,
                "p-value": binom_res.pvalue,
                "ci": [binom_ci.low, binom_ci.high],
            },
        }


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: Forces PyTorch to use deterministic algorithms
    # Warning: May throw an error if a specific SAM2 operation doesn't have a deterministic version
    # torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate video actions using QwenEngine")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("./multi_sample_inference/debug_output/"),
        help="Path to the directory containing output videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./evaluation/output/"),
        help="Path to the directory to store the evaluation output",
    )
    args = parser.parse_args()

    set_seed(42)

    engine = QwenEngine("Qwen/Qwen3-VL-8B-Instruct")
    console = Console()

    pipeline = EvaluationPipeline(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        sam_model_id="facebook/sam2-hiera-large",
        target_fps=16,
        sam_margin=0.2,
    )

    auditors = {
        "soft_direct": SoftDirectAuditor(engine),
        # "soft_blind": SoftBlindAuditor(engine),
        # "discrete_direct": DiscreteDirectAuditor(engine),
        # "discrete_blind": DiscreteBlindAuditor(engine),
        "soft_2afc": SoftTwoAFCAuditor(engine),
    }

    results_table = Table(
        title="Evaluation Statistics", show_header=True, header_style="bold magenta"
    )
    results_table.add_column("Score Function", style="cyan")
    results_table.add_column("Parameters", style="dim")
    results_table.add_column("Discrim. Rate", justify="right", style="green")
    results_table.add_column("p-value", justify="right")
    results_table.add_column("Discrim. 95% CI", justify="center")
    results_table.add_column("Mean Margin", justify="right", style="green")
    results_table.add_column("Margin 95% CI", justify="center")

    for name, auditor in auditors.items():
        console.print(f"\n[bold blue]Evaluating:[/bold blue] {name}")
        results = pipeline.run(auditor)

        for res in results:
            s = res["statistics"]
            d, m = s["discrimination"], s["margin"]
            results_table.add_row(
                name,
                json.dumps(res["params"]),
                f"{d['avg']:.2f}",
                f"{d['p-value']:.4f}",
                f"[{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]",
                f"{m['avg']:.3f}",
                f"[{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]",
            )
        console.print(results_table)


if __name__ == "__main__":
    main()
