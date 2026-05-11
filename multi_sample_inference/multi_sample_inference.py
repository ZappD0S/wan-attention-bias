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
import time
import uuid
from pathlib import Path

import msgspec
import numpy as np
import torch
from diffusers.utils.loading_utils import load_image
from huggingface_hub import snapshot_download
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from schema import ProcessedVideoSpecification, VideoGenerationDataset
from utils import create_mask_from_bbox

from . import fsdp_worker
from .utils import get_folder_name

TARGET_SIZE = (480, 832)
keep_running = True


def handle_slurm_signal(signum, _):
    global keep_running
    tqdm.write(f"\n[Dispatcher] Signal {signum} received. Stopping loop...")
    keep_running = False


signal.signal(signal.SIGUSR1, handle_slurm_signal)
signal.signal(signal.SIGTERM, handle_slurm_signal)
signal.signal(signal.SIGINT, handle_slurm_signal)


def run_repeat_loop(
    tasks_list: list,
    checkpoint_dir,
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
            tqdm.write(f"Skipping existing: {video_path.name}")
            continue

        tasks_list.append(
            {
                "prompt_sentences": prompt_sentences,
                "checkpoint_dir": checkpoint_dir,
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
    tasks_list,
    checkpoint_dir,
    prompt_data: ProcessedVideoSpecification,
    param_config,
    img,
    single_char_imgs,
    masks,
    output_path,
    repeat,
):
    action_prompts_dict = msgspec.to_builtins(prompt_data.action_prompts)

    allowed_prompt_types = param_config.get("prompt_types", list(action_prompts_dict.keys()))

    for prompt_type, action_prompt_data in action_prompts_dict.items():
        if prompt_type not in allowed_prompt_types:
            continue

        config = {
            "params": param_config,
            "prompt_data": msgspec.to_builtins(prompt_data),
            "prompt_type": prompt_type,
        }

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

        general_prompt = action_prompt_data.get(
            "general_prompt", "high quality video, background scenery"
        )

        run_repeat_loop(
            tasks_list,
            checkpoint_dir,
            prompt_sentences,
            img,
            single_char_imgs,
            char_segments_list,
            masks,
            param_config,
            action_output_path,
            repeat,
            general_prompt,
        )
    return False


def process_parameter_grid(
    tasks_list, checkpoint_dir, prompt_data: ProcessedVideoSpecification, output_path, param_grid
):
    img = load_image(prompt_data.img_paths.original)
    single_char_imgs = [load_image(path) for path in prompt_data.img_paths.single_char]

    for param_config in ParameterGrid(param_grid):
        repeat = param_config.pop("repeat", 1)
        simil_masks_type = param_config["simil_masks_type"]

        if simil_masks_type == "fixed":
            bboxes = prompt_data.enlarged_bboxes
            masks = np.stack([create_mask_from_bbox(bbox, TARGET_SIZE) for bbox in bboxes])
        else:
            masks = np.stack(
                [
                    np.array(load_image(path).convert("L")) > 128
                    for path in prompt_data.img_paths.seg_masks
                ]
            )

        overlap_mask = masks.sum(axis=0) > 1
        masks &= ~overlap_mask
        masks = gaussian_filter(masks, sigma=5.0, axes=(1, 2))
        masks = torch.from_numpy(masks).to(torch.bool)
        masks = masks.flip(dims=(0,)) if param_config.get("invert", False) else masks

        process_action_prompts(
            tasks_list,
            checkpoint_dir,
            prompt_data,
            param_config,
            img,
            single_char_imgs,
            masks,
            output_path,
            repeat,
        )
    return False


def team_thread(team_id, assigned_gpus, mode, t5_cpu, task_queue):
    master_port = str(29500 + team_id)

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"team_{team_id}.log"

    worker_module = fsdp_worker.__spec__.name  # ty:ignore[unresolved-attribute]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned_gpus))

    while keep_running:
        try:
            task_file = task_queue.get(timeout=3)
        except queue.Empty:
            break

        tqdm.write(f"[Team {team_id}] Generating {task_file.name} (Mode: {mode}, T5-CPU: {t5_cpu})")

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={len(assigned_gpus)}",
            f"--master_port={master_port}",
            "-m",
            worker_module,
            "--task-file",
            str(task_file),
            "--mode",
            mode,
        ]

        if t5_cpu:
            cmd.append("--t5-cpu")

        try:
            with open(log_path, "a") as log_file:
                log_file.write(f"\n--- Starting Task: {task_file.name} ---\n")
                subprocess.run(
                    cmd,
                    env=env,
                    check=True,
                    cwd=project_root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError:
            err_msg = f"[ERROR] Team {team_id} failed. Check {log_path} for details."
            tqdm.write(err_msg)

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

    return threads


def auto_configure_hardware():
    """Determines the best strategy based on GPU VRAM."""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected!")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    tqdm.write(f"[Hardware] {num_gpus} GPUs detected. VRAM: {vram_gb:.1f} GB per device.")

    if vram_gb >= 65:
        tqdm.write("[Strategy] Tier A (>=65GB): Running Solo workers (T5 on GPU).")
        return 1, "solo", False
    elif vram_gb >= 35:
        tqdm.write("[Strategy] Tier B (35-64GB): Running FSDP Teams (T5 on GPU).")
        return 2, "fsdp", False
    elif num_gpus >= 4:
        tqdm.write("[Strategy] Tier C1 (<35GB, >=4 GPUs): Running FSDP Teams of 4 (T5 on GPU).")
        return 4, "fsdp", False
    elif num_gpus >= 2:
        tqdm.write("[Strategy] Tier C2 (<35GB, 2-3 GPUs): Running FSDP Teams of 2 (T5 on CPU).")
        return 2, "fsdp", True
    else:
        tqdm.write("[Strategy] Tier C3 (<35GB, 1 GPU): Running Solo workers (T5 on CPU).")
        return 1, "solo", True


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

    with open(args.prompts_file, "rb") as f:
        dataset_root = msgspec.json.decode(
            f.read(), type=VideoGenerationDataset[ProcessedVideoSpecification]
        )

    args.output_path.mkdir(exist_ok=True, parents=True)
    sync_param_grid(args.param_grid_file, args.output_path)

    try:
        checkpoint_dir = snapshot_download("Wan-AI/Wan2.1-I2V-14B-480P", local_files_only=False)
        tqdm.write(f"[Dispatcher] Model resolved at: {checkpoint_dir}")
    except Exception as e:
        raise RuntimeError(
            "Failed to resolve model path. Ensure you have internet or the model is cached."
        ) from e

    tasks_list = []

    for prompt_data in dataset_root.dataset:
        process_parameter_grid(
            tasks_list,
            checkpoint_dir,
            prompt_data,
            args.output_path,
            param_grid,
        )

    if not tasks_list:
        print("No new tasks to perform.")
        return

    tqdm.write(f"[Dispatcher] Serializing {len(tasks_list)} task files...")
    task_dir = args.output_path / "temp_tasks"
    task_dir.mkdir(exist_ok=True)
    task_queue = queue.Queue()

    for task_dict in tasks_list:
        task_file = task_dir / f"task_{uuid.uuid4().hex}.pkl"
        with open(task_file, "wb") as f:
            pickle.dump(task_dict, f)
        task_queue.put(task_file)

    threads = launch_workers(task_queue)
    total_tasks = len(tasks_list)

    with tqdm(total=total_tasks, desc="Generating Videos", unit="task", dynamic_ncols=True) as pbar:
        while task_queue.unfinished_tasks > 0 and keep_running:
            # Update progress
            completed = total_tasks - task_queue.unfinished_tasks
            if completed > pbar.n:
                pbar.update(completed - pbar.n)

            # Calculate and display the exact finish timestamp
            rate = pbar.format_dict.get("rate")
            if rate and rate > 0:
                remaining_seconds = (pbar.total - pbar.n) / rate
                finish_timestamp = time.time() + remaining_seconds
                finish_str = time.strftime("%b %d, %H:%M:%S", time.localtime(finish_timestamp))
                pbar.set_postfix_str(f"Finish ~ {finish_str}")

            # Check for thread crashes
            if not any(t.is_alive() for t in threads):
                tqdm.write("\n[Dispatcher] All worker threads have stopped unexpectedly.")
                break

            time.sleep(1)

        # Final UI refresh to 100%
        completed = total_tasks - task_queue.unfinished_tasks
        if completed > pbar.n:
            pbar.update(completed - pbar.n)
        pbar.set_postfix_str("Done!")

    tqdm.write("[Dispatcher] Batch processing finished.")


if __name__ == "__main__":
    main()
