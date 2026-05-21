import argparse
import json
import logging
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from scipy.stats import binomtest
from tqdm import tqdm

from .auditor import (
    ActionAuditor,
    DiscreteBlindAuditor,
    DiscreteDirectAuditor,
    SoftBlindAuditor,
    SoftDirectAuditor,
    SoftTwoAFCAuditor,
    VideoAsset,
)
from .engine import QwenEngine
from .sam2_pipeline import run_sam2_pipeline
from .utils import DataEntry, group_entries


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
        valid_dirs = [
            d
            for d in self.videos_dir.iterdir()
            if d.is_dir() and d.name != "debug" and not d.name.startswith(".")
        ]
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

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Optional: Forces PyTorch to use deterministic algorithms
    # Warning: May throw an error if a specific SAM2 operation doesn't have a deterministic version
    # torch.use_deterministic_algorithms(True, warn_only=True)


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    for library in ["transformers", "torch", "sam2", "qwen_vl_utils", "urllib3"]:
        logging.getLogger(library).setLevel(logging.WARNING)


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
    setup_logging(args.output_dir / "debug_eval.log")

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
        "soft_blind": SoftBlindAuditor(engine),
        "discrete_direct": DiscreteDirectAuditor(engine),
        "discrete_blind": DiscreteBlindAuditor(engine),
        "soft_2afc": SoftTwoAFCAuditor(engine),
        # "videobench": VideoBenchAuditor(engine),
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
