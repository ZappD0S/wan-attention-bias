import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.loading_utils import load_image
from PIL import Image
from sklearn.model_selection import ParameterGrid
from wan.configs.wan_i2v_14B import i2v_14B
from wan.regional_prompt import WanI2V

from utils import create_mask_from_bbox, normalize_video_tensor
from debug_utils import write_video_masks, draw_boxes, draw_masks, unscale

from .utils import get_folder_name

SAMPLING_STEPS = 40
FRAME_NUM = 81  # default
TARGET_SIZE = (480, 832)


def run_inference(
    wan_i2v: WanI2V,
    prompt: str,
    img: Image.Image,
    character_segments: list[str],
    masks: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    num_layers = wan_i2v.model.num_layers

    timestep_bias_schedule = torch.ones(SAMPLING_STEPS, dtype=torch.bool)
    blocks_bias_schedule = torch.ones(num_layers, dtype=torch.bool)
    control_prompts = {
        (i,): {"prompt": seg, "descr_list": []} for i, seg in enumerate(character_segments)
    }

    n_characters = len(character_segments)
    wlw_matrix = np.eye(n_characters, n_characters, dtype=bool)
    wlw_matrix = np.repeat(wlw_matrix[..., np.newaxis], FRAME_NUM, axis=-1)
    wlw_matrix = torch.from_numpy(wlw_matrix)

    bias_kwargs = {
        "control_prompts": control_prompts,
        "timestep_bias_schedule": timestep_bias_schedule,
        "blocks_bias_schedule": blocks_bias_schedule,
        "face_masks": masks,
        "wlw_matrix": wlw_matrix,
    } | config

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    video: torch.Tensor
    extra_data: dict[str, torch.Tensor]
    video, extra_data = wan_i2v.generate(  # type: ignore
        prompt,
        img,
        bias_kwargs,
        max_area=TARGET_SIZE[0] * TARGET_SIZE[1],
        sampling_steps=SAMPLING_STEPS,
        frame_num=FRAME_NUM,
    )

    return video, extra_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--checkpoint-path", default=Path("./weights/"), type=Path)
    parser.add_argument("--t5-cpu", action="store_true")

    args = parser.parse_args()

    with open("param_grid.json") as f:
        param_grid = json.load(f)

    with open(args.prompts_file) as f:
        prompts_data_list = json.load(f)

    output_path = args.output_path
    output_path.mkdir(exist_ok=True, parents=True)

    debug_path = output_path / "debug"
    debug_path.mkdir(exist_ok=True)

    wan_i2v = WanI2V(
        config=i2v_14B,
        checkpoint_dir=str(args.checkpoint_path / "Wan2.1-I2V-14B-480P"),
        device_id=0,
        t5_cpu=args.t5_cpu,
    )
    generated = False

    for i, prompt_data in enumerate(prompts_data_list):
        bboxes = prompt_data["bboxes"]
        masks_list = [
            torch.from_numpy(
                create_mask_from_bbox(
                    bbox,
                    TARGET_SIZE,
                )
            )
            for bbox in bboxes
        ]
        masks = torch.stack(masks_list)

        img_dir = args.prompts_file.parent
        img_path = str(img_dir / prompt_data["img_path"])
        img = load_image(img_path)
        assert img.size[::-1] == TARGET_SIZE

        img_with_boxes = draw_boxes(img, prompt_data["bboxes"])
        img_with_boxes.save(debug_path / "img_with_boxes.png")

        img_with_masks = draw_masks(img, list(masks))
        img_with_masks.save(debug_path / "img_with_masks.png")

        for config in ParameterGrid(param_grid):
            assert config["bias_method"] in {"none", "regional_prompting", "ediff-i"}

            # iterate over actions prompts
            for prompt_type, action_prompt_data in prompt_data["action_prompts"].items():
                assert prompt_type in {"default", "first_action", "second_action", "no_locative"}
                # generate only baseline for single action prompts
                if config["bias_method"] != "none" and prompt_type in {
                    "first_action",
                    "second_action",
                }:
                    continue

                config["prompt"] = action_prompt_data | {"type": prompt_type}

                folder_name = get_folder_name(config)
                action_output_path = output_path / folder_name / prompt_type

                segments = action_prompt_data["segments"]
                prompt = " ".join(segments)

                segment_mask = action_prompt_data["mask"]
                character_segments = [
                    seg for is_char, seg in zip(segment_mask, segments) if is_char
                ]
                assert len(bboxes) == len(character_segments), f"error in prompt #{i}"

                action_output_path = output_path / prompt_type
                action_output_path.mkdir(exist_ok=True)

                config_path = action_output_path / "config.json"

                with config_path.open("w") as f:
                    json.dump(config, f, indent=2)

                repeat = config.get("repeat", 1)
                if repeat <= 0:
                    raise ValueError("repat parma must be positive.")

                for i in range(repeat):
                    video_path = action_output_path / f"video_{i}.mp4"

                    if video_path.exists():
                        print(
                            f"The video for the prompt '{prompt}' the was already generated. Skipping..."
                        )
                        continue

                    generated = True
                    video, extra_data = run_inference(
                        wan_i2v, prompt, img, character_segments, masks, config
                    )
                    video_norm = normalize_video_tensor(video.cpu().numpy())

                    export_to_video(list(video_norm), video_path, fps=16)

                    simil_masks = extra_data["simil_masks"]

                    # TODO: is this the best way to do it?
                    face_masks = simil_masks[0, -1].float().mean(dim=0) > 0.5

                    h, w = video.shape[-2:]
                    face_masks = unscale(face_masks.float(), (FRAME_NUM, h, w)).bool()
                    write_video_masks(
                        video_norm,
                        action_output_path / "video_with_masks_{i}.mp4",
                        face_masks.transpose(0, 1).cpu().numpy(),
                        fps=16,
                    )

    return generated


if __name__ == "__main__":
    generated = main()
    if generated:
        sys.exit(0)
    else:
        sys.exit(2)
