import argparse
import gc
import json
import signal
import sys
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
    char_segment_lists: list[list[str]],
    masks: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    num_layers = wan_i2v.model.num_layers
    timestep_bias_schedule = torch.ones(SAMPLING_STEPS, dtype=torch.bool)
    blocks_bias_schedule = torch.ones(num_layers, dtype=torch.bool)

    control_prompt_lists = []
    n_char = 0
    for char_segments in char_segment_lists:
        control_prompts = []
        for seg in char_segments:
            control_prompts.append(((n_char,), {"prompt": seg, "char_descr_list": []}))
            n_char += 1
        control_prompt_lists.append(control_prompts)

    wlw_matrix = np.eye(n_char, n_char, dtype=bool)
    wlw_matrix = np.repeat(wlw_matrix[..., np.newaxis], FRAME_NUM, axis=-1)
    wlw_matrix = torch.from_numpy(wlw_matrix)

    bias_kwargs = {
        "control_prompt_lists": control_prompt_lists,
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


def save_outputs(video, extra_data, video_path, action_output_path, repeat_idx):
    video_norm = normalize_video_tensor(video.cpu().numpy())
    export_to_video(list(video_norm), video_path, fps=16)

    simil_masks = extra_data["simil_masks"]
    face_masks = simil_masks[0, -1].float().mean(dim=0) > 0.5
    h, w = video.shape[-2:]
    face_masks = unscale(face_masks.float(), (FRAME_NUM, h, w)).bool()

    write_video_masks(
        video_norm,
        action_output_path / f"video_with_masks_{repeat_idx}.mp4",
        face_masks.transpose(0, 1).cpu().numpy(),
        fps=16,
    )


def run_repeat_loop(
    wan_i2v, prompt_sentences, img, character_segments, masks, config, action_output_path, repeat
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
            wan_i2v, prompt_sentences, img, character_segments, masks, config
        )
        save_outputs(video, extra_data, video_path, action_output_path, repeat_idx)

    return False


def process_action_prompts(
    wan_i2v, prompt_data, param_config, img, masks, output_path, repeat, safeguard_suffix
):
    for prompt_type, action_prompt_data in prompt_data["action_prompts"].items():
        allowed_types = param_config.get("prompt_types")
        if allowed_types is not None and prompt_type not in allowed_types:
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
        character_segments = [
            [seg for is_char, seg in zip(mask_row, segs) if is_char]
            for mask_row, segs in zip(segment_masks, segment_lists)
        ]

        signal_received = run_repeat_loop(
            wan_i2v,
            prompt_sentences,
            img,
            character_segments,
            masks,
            param_config,
            action_output_path,
            repeat,
        )
        if signal_received:
            return True

    return False


def process_parameter_grid(
    wan_i2v, prompt_data, img, masks, output_path, param_grid, safeguard_suffix
):
    for param_config in ParameterGrid(param_grid):
        repeat = param_config.pop("repeat", 1)

        current_masks = masks[::-1] if param_config.get("invert", False) else masks

        signal_received = process_action_prompts(
            wan_i2v,
            prompt_data,
            param_config,
            img,
            current_masks,
            output_path,
            repeat,
            safeguard_suffix,
        )
        if signal_received:
            return True

    return False


def process_prompt_entry(
    wan_i2v, prompt_data, idx, output_path, param_grid, img_dir, safeguard_suffix
):
    bboxes = prompt_data["bboxes"]
    masks = torch.stack(
        [torch.from_numpy(create_mask_from_bbox(bbox, TARGET_SIZE)) for bbox in bboxes]
    )

    img_path = str(img_dir / prompt_data["img_path"])
    img = load_image(img_path)

    # Debug visualization
    debug_path = output_path / "debug" / f"prompt_{idx}"
    debug_path.mkdir(exist_ok=True, parents=True)

    if not (debug_path / "img_with_boxes.png").exists():
        draw_boxes(img, bboxes).save(debug_path / "img_with_boxes.png")
    if not (debug_path / "img_with_masks.png").exists():
        draw_masks(img, list(masks)).save(debug_path / "img_with_masks.png")

    return process_parameter_grid(
        wan_i2v, prompt_data, img, masks, output_path, param_grid, safeguard_suffix
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
        signal_received = process_prompt_entry(
            wan_i2v, prompt_data, i, args.output_path, param_grid, img_dir, safeguard_suffix
        )
        if signal_received:
            return True

    return False


if __name__ == "__main__":
    interrupted = main()
    sys.exit(2 if interrupted else 0)
