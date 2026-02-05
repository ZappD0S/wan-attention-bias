import argparse
import gc
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.loading_utils import load_image
from PIL import Image
from sklearn.model_selection import ParameterGrid
from wan.configs.wan_i2v_14B import i2v_14B
from wan.regional_prompt import WanI2V

from debug_utils import draw_boxes, draw_masks, unscale, write_video_masks
from utils import create_mask_from_bbox, normalize_video_tensor

from .utils import get_folder_name

SAMPLING_STEPS = 40
FRAME_NUM = 81
TARGET_SIZE = (480, 832)

keep_running = True


def handle_slurm_signal(signum, _):
    global keep_running
    print(f"Received signal {signum}. Finishing current item and exiting...")
    keep_running = False


signal.signal(signal.SIGUSR1, handle_slurm_signal)
signal.signal(signal.SIGTERM, handle_slurm_signal)


def run_inference(
    wan_i2v: WanI2V,
    prompt_sentences: list[str],
    img: Image.Image,
    char_segments_list: list[list[str]],
    masks: torch.Tensor,
    config: dict,
    general_prompt: str | None = None,
    single_char_imgs: list[Image.Image] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    num_layers = wan_i2v.model.num_layers
    timestep_bias_schedule = torch.ones(SAMPLING_STEPS, dtype=torch.bool)
    blocks_bias_schedule = torch.ones(num_layers, dtype=torch.bool)

    prompt_data_list = []
    n_char = 0

    for i, char_segments in enumerate(char_segments_list):
        single_char_img = single_char_imgs[i] if single_char_imgs is not None else None
        control_prompts = []

        for seg in char_segments:
            control_prompts.append(((n_char,), {"prompt": seg, "char_descr_list": []}))
            n_char += 1

        prompt_data_list.append(
            {"control_prompts": control_prompts, "single_char_img": single_char_img}
        )

    wlw_matrix = np.eye(n_char, n_char, dtype=bool)
    wlw_matrix = np.repeat(wlw_matrix[..., np.newaxis], FRAME_NUM, axis=-1)
    wlw_matrix = torch.from_numpy(wlw_matrix)

    bias_kwargs = {
        "general_prompt": general_prompt,
        "prompt_data_list": prompt_data_list,
        "timestep_bias_schedule": timestep_bias_schedule,
        "blocks_bias_schedule": blocks_bias_schedule,
        "face_masks": masks,
        "wlw_matrix": wlw_matrix,
    } | config

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    video, extra_data = wan_i2v.generate(
        prompt_sentences,
        img,
        bias_kwargs,
        max_area=TARGET_SIZE[0] * TARGET_SIZE[1],
        sampling_steps=SAMPLING_STEPS,
        frame_num=FRAME_NUM,
    )
    return video, extra_data


def save_outputs(video, extra_data, video_path, action_output_path, repeat_idx, max_retries=5):
    def _robust_run(path, func, *args):
        attempts = 0
        while attempts < max_retries:
            try:
                # ensure parent exists and refresh Lustre metadata
                path.parent.mkdir(parents=True, exist_ok=True)
                return func(*args)
            except FileNotFoundError:
                attempts += 1
                if attempts == max_retries:
                    raise
                time.sleep(1)

    video_norm = normalize_video_tensor(video.cpu().numpy())

    _robust_run(video_path, export_to_video, list(video_norm), video_path, 16)

    simil_masks = extra_data["simil_masks"]
    face_masks = simil_masks[0, -1].float().mean(dim=0) > 0.5
    h, w = video.shape[-2:]
    face_masks = unscale(face_masks.float(), (FRAME_NUM, h, w)).bool()

    mask_path = action_output_path / f"video_with_masks_{repeat_idx}.mp4"
    _robust_run(
        mask_path,
        write_video_masks,
        video_norm,
        mask_path,
        face_masks.transpose(0, 1).cpu().numpy(),
        16,
    )


def run_repeat_loop(
    wan_i2v,
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
        if not keep_running:
            return True

        video_path = action_output_path / f"video_{repeat_idx}.mp4"
        if video_path.exists():
            print(f"Skipping existing: {video_path.name}")
            continue

        video, extra_data = run_inference(
            wan_i2v,
            prompt_sentences,
            img,
            char_segments_list,
            masks,
            config,
            general_prompt=general_prompt,
            single_char_imgs=single_char_imgs,
        )
        save_outputs(video, extra_data, video_path, action_output_path, repeat_idx)

    return False


def process_action_prompts(
    wan_i2v, prompt_data, param_config, img, single_char_imgs, masks, output_path, repeat
):
    for prompt_type, action_prompt_data in prompt_data["action_prompts"].items():
        allowed_types = param_config.get("prompt_types")
        assert set(allowed_types) <= prompt_data["action_prompts"].keys()

        if allowed_types is not None and prompt_type not in allowed_types:
            continue

        config = {"params": param_config, "prompt_data": prompt_data, "prompt_type": prompt_type}
        folder_name = get_folder_name(config)
        action_output_path = output_path / folder_name
        action_output_path.mkdir(exist_ok=True)

        with (action_output_path / "config.json").open("w") as f:
            json.dump(config, f, indent=2)

        print(f"Working on folder {action_output_path.name}")
        segment_lists = action_prompt_data["segments"]
        prompt_sentences = [" ".join(segments) for segments in segment_lists]

        segment_masks = action_prompt_data["mask"]
        char_segments_list = [
            [seg for is_char, seg in zip(mask_row, segs) if is_char]
            for mask_row, segs in zip(segment_masks, segment_lists)
        ]

        general_prompt = action_prompt_data.get("general_prompt")

        signal_received = run_repeat_loop(
            wan_i2v,
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

        if signal_received:
            return True

    return False


def process_parameter_grid(
    wan_i2v, prompt_data, img, single_char_imgs, masks, output_path, param_grid
):
    for param_config in ParameterGrid(param_grid):
        repeat = param_config.pop("repeat", 1)

        current_masks = masks.flip(dims=(0,)) if param_config.get("invert", False) else masks

        signal_received = process_action_prompts(
            wan_i2v,
            prompt_data,
            param_config,
            img,
            single_char_imgs,
            current_masks,
            output_path,
            repeat,
        )
        if signal_received:
            return True

    return False


def process_prompt_entry(wan_i2v, prompt_data, idx, output_path, param_grid, img_dir):
    bboxes = prompt_data["bboxes"]
    masks = torch.stack(
        [torch.from_numpy(create_mask_from_bbox(bbox, TARGET_SIZE)) for bbox in bboxes]
    )

    img_path = str(img_dir / prompt_data["img_paths"]["original"])
    img = load_image(img_path)

    single_char_imgs = [
        load_image(str(img_dir / rel_path)) for rel_path in prompt_data["img_paths"]["single_char"]
    ]

    # Debug visualization
    debug_path = output_path / "debug" / f"prompt_{idx}"
    debug_path.mkdir(exist_ok=True, parents=True)

    if not (debug_path / "img_with_boxes.png").exists():
        draw_boxes(img, bboxes).save(debug_path / "img_with_boxes.png")
    if not (debug_path / "img_with_masks.png").exists():
        draw_masks(img, list(masks)).save(debug_path / "img_with_masks.png")

    return process_parameter_grid(
        wan_i2v, prompt_data, img, single_char_imgs, masks, output_path, param_grid
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", required=True, type=Path)
    parser.add_argument("--param-grid-file", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--checkpoint-path", default=Path("./weights/"), type=Path)
    parser.add_argument("--t5-cpu", action="store_true")
    args = parser.parse_args()

    with open(args.param_grid_file) as f:
        param_grid = json.load(f)
    with open(args.prompts_file) as f:
        prompt_json_dict = json.load(f)

    args.output_path.mkdir(exist_ok=True, parents=True)
    img_dir = args.prompts_file.parent

    wan_i2v = WanI2V(
        config=i2v_14B,
        checkpoint_dir=str(args.checkpoint_path / "Wan2.1-I2V-14B-480P"),
        device_id=0,
        t5_cpu=args.t5_cpu,
    )

    safeguard_suffix = prompt_json_dict["safeguard_suffix"]
    for i, prompt_data in enumerate(prompt_json_dict["dataset"]):
        prompt_data["safeguard_suffix"] = safeguard_suffix

        signal_received = process_prompt_entry(
            wan_i2v, prompt_data, i, args.output_path, param_grid, img_dir
        )
        if signal_received:
            return True

    return False


if __name__ == "__main__":
    interrupted = main()
    sys.exit(2 if interrupted else 0)
