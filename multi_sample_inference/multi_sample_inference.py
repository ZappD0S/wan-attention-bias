import argparse
import filecmp
import json
import os
import pickle
import queue
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import numpy as np
import torch
from diffusers.utils.loading_utils import load_image
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import ParameterGrid

from utils import create_mask_from_bbox

from . import fsdp_worker
from .utils import get_folder_name

TARGET_SIZE = (480, 832)
keep_running = True


def handle_slurm_signal(signum, _):
    global keep_running
    print(f"\n[Dispatcher] Signal {signum} received. Stopping loop...")
    keep_running = False


signal.signal(signal.SIGUSR1, handle_slurm_signal)
signal.signal(signal.SIGTERM, handle_slurm_signal)
signal.signal(signal.SIGINT, handle_slurm_signal)


def run_repeat_loop(
    tasks_list: list,
    prompt_sentences,
    img,
    single_char_imgs,
    char_segments_list,
    masks,
    config,
    action_output_path,
    repeat,
    general_prompt,
):
    global keep_running

    for repeat_idx in range(repeat):
        video_path = action_output_path / f"video_{repeat_idx}.mp4"
        if video_path.exists():
            print(f"Skipping existing: {video_path.name}")
            continue

        tasks_list.append(
            {
                "prompt_sentences": prompt_sentences,
                "img": img,
                "single_char_imgs": single_char_imgs,
                "char_segments_list": char_segments_list,
                "masks": masks,
                "config": config,
                "general_prompt": general_prompt,
                "video_path": video_path,
                "action_output_path": action_output_path,
                "repeat_idx": repeat_idx,
            }
        )
    return False


def process_action_prompts(
    tasks_list, prompt_data, param_config, img, single_char_imgs, masks, output_path, repeat
):
    allowed_prompt_types = param_config.get(
        "prompt_types", list(prompt_data["action_prompts"].keys())
    )
    for prompt_type, action_prompt_data in prompt_data["action_prompts"].items():
        if prompt_type not in allowed_prompt_types:
            continue

        config = {"params": param_config, "prompt_data": prompt_data, "prompt_type": prompt_type}
        folder_name = get_folder_name(config)
        action_output_path = output_path / folder_name
        action_output_path.mkdir(exist_ok=True)

        with (action_output_path / "config.json").open("w") as f:
            json.dump(config, f, indent=2)

        segment_lists = action_prompt_data["segments"]
        prompt_sentences = [" ".join(segments) for segments in segment_lists]
        segment_masks = action_prompt_data["mask"]
        char_segments_list = [
            [seg for is_char, seg in zip(mask_row, segs, strict=True) if is_char]
            for mask_row, segs in zip(segment_masks, segment_lists, strict=True)
        ]

        run_repeat_loop(
            tasks_list,
            prompt_sentences,
            img,
            single_char_imgs,
            char_segments_list,
            masks,
            param_config,
            action_output_path,
            repeat,
            "high quality video, background scenery",
        )
    return False


def process_parameter_grid(tasks_list, prompt_data, output_path, param_grid):
    img = load_image(prompt_data["img_paths"]["original"])
    single_char_imgs = [load_image(path) for path in prompt_data["img_paths"]["single_char"]]

    for param_config in ParameterGrid(param_grid):
        repeat = param_config.pop("repeat", 1)
        simil_masks_type = param_config["simil_masks_type"]

        if simil_masks_type == "fixed":
            bboxes = prompt_data["enlarged_bboxes"]
            masks = np.stack([create_mask_from_bbox(bbox, TARGET_SIZE) for bbox in bboxes])
        else:
            masks = np.stack(
                [
                    np.array(load_image(path).convert("L")) > 128
                    for path in prompt_data["img_paths"]["seg_masks"]
                ]
            )

        overlap_mask = masks.sum(axis=0) > 1
        masks &= ~overlap_mask
        masks = gaussian_filter(masks, sigma=5.0, axes=(1, 2))
        masks = torch.from_numpy(masks).to(torch.bool)
        masks = masks.flip(dims=(0,)) if param_config.get("invert", False) else masks

        process_action_prompts(
            tasks_list, prompt_data, param_config, img, single_char_imgs, masks, output_path, repeat
        )
    return False


def team_thread(team_id, assigned_gpus, mode, t5_cpu, task_queue):
    master_port = str(29500 + team_id)

    worker_file_path = Path(fsdp_worker.__file__).resolve()
    script_dir = worker_file_path.parent

    worker_module_name = fsdp_worker.__name__
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned_gpus))

    while keep_running:
        try:
            task_file = task_queue.get(timeout=3)
        except queue.Empty:
            break

        print(f"[Team {team_id}] Generating {task_file.name} (Mode: {mode}, T5-CPU: {t5_cpu})")

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={len(assigned_gpus)}",
            f"--master_port={master_port}",
            "-m",
            worker_module_name,
            "--task-file",
            str(task_file),
            "--mode",
            mode,
        ]

        if t5_cpu:
            cmd.append("--t5-cpu")

        try:
            subprocess.run(cmd, env=env, check=True, cwd=script_dir)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Team {team_id} failed with code {e.returncode}")

        if task_file.exists():
            task_file.unlink()
        task_queue.task_done()


def launch_workers(task_queue):
    gpus_per_team, main_mode, main_t5_cpu = auto_configure_hardware()
    num_gpus = torch.cuda.device_count()
    threads, current_gpu, worker_id = [], 0, 0

    while current_gpu + gpus_per_team <= num_gpus:
        assigned_gpus = list(range(current_gpu, current_gpu + gpus_per_team))
        t = threading.Thread(
            target=team_thread, args=(worker_id, assigned_gpus, main_mode, main_t5_cpu, task_queue)
        )
        t.start()
        threads.append(t)
        current_gpu += gpus_per_team
        worker_id += 1

    while current_gpu < num_gpus:
        # leftover gpus fallback to solo + cpu-t5 unless they are massive
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        t5_cpu = vram_gb < 65
        t = threading.Thread(
            target=team_thread, args=(worker_id, [current_gpu], "solo", t5_cpu, task_queue)
        )
        t.start()
        threads.append(t)
        current_gpu += 1
        worker_id += 1


def auto_configure_hardware():
    """Determines the best strategy based on GPU VRAM."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[Hardware] {num_gpus} GPUs detected. VRAM: {vram_gb:.1f} GB per device.")

    if vram_gb >= 65:
        print("[Strategy] Tier A (>=65GB): Running Solo workers (T5 on GPU).")
        return 1, "solo", False
    elif vram_gb >= 35:
        print("[Strategy] Tier B (35-64GB): Running FSDP Teams (T5 on GPU).")
        return 2, "fsdp", False
    else:
        print("[Strategy] Tier C (<35GB): Running FSDP Teams (T5 on CPU).")
        return 2, "fsdp", True


def sync_param_grid(src_path, output_dir):
    src, dest = Path(src_path), Path(output_dir) / Path(src_path).name

    if dest.exists():
        if not filecmp.cmp(src, dest, shallow=False):
            raise FileExistsError(f"Conflict: {dest} exists with different content.")

        return  # content is identical, nothing to do

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", required=True, type=Path)
    parser.add_argument("--param-grid-file", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()

    with open(args.param_grid_file) as f:
        param_grid = json.load(f)

    with open(args.prompts_file) as f:
        prompt_json_dict = json.load(f)

    args.output_path.mkdir(exist_ok=True, parents=True)
    sync_param_grid(args.param_grid_file, args.output_path)

    tasks_list = []
    safeguard_suffix = prompt_json_dict.get("safeguard_suffix", "")
    for prompt_data in prompt_json_dict["dataset"]:
        process_parameter_grid(tasks_list, prompt_data, args.output_path, param_grid)
        prompt_data["safeguard_suffix"] = safeguard_suffix

    if not tasks_list:
        print("No new tasks to perform.")
        return

    print(f"[Dispatcher] Serializing {len(tasks_list)} task files...")
    task_dir = args.output_path / "temp_tasks"
    task_dir.mkdir(exist_ok=True)
    task_queue = queue.Queue()

    for task_dict in tasks_list:
        task_file = task_dir / f"task_{uuid.uuid4().hex}.pkl"
        with open(task_file, "wb") as f:
            pickle.dump(task_dict, f)
        task_queue.put(task_file)

    launch_workers(task_queue)
    print("[Dispatcher] Batch processing finished.")


if __name__ == "__main__":
    main()
