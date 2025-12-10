import argparse
import base64
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.loading_utils import load_image
from PIL import Image
from sklearn.model_selection import ParameterGrid
from wan.configs.wan_i2v_14B import i2v_14B
from einops import rearrange
from wan.regional_prompt import WanI2V

from utils import create_mask_from_bbox

sampling_steps = 40
frame_num = 81  # default
target_size = (480, 832)


def get_folder_name(config: dict, length=6) -> str:
    encoded = json.dumps(config, sort_keys=True).encode()

    # use .digest() instead of .hexdigest() to get raw binary data
    digest = hashlib.md5(encoded).digest()
    b64_bytes = base64.urlsafe_b64encode(digest)
    folder_name = b64_bytes.decode().rstrip("=")

    return folder_name[:length]


def run_inference(
    wan_i2v: WanI2V,
    prompt: str,
    img: Image.Image,
    character_segments: list[str],
    masks: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    num_layers = wan_i2v.model.num_layers

    if config["beta"] > 0.0:
        timestep_bias_schedule = torch.ones(sampling_steps, dtype=torch.bool)
        blocks_bias_schedule = torch.ones(num_layers, dtype=torch.bool)
    else:
        timestep_bias_schedule = torch.zeros(sampling_steps, dtype=torch.bool)
        blocks_bias_schedule = torch.zeros(num_layers, dtype=torch.bool)

    control_prompts = {
        (i,): {"prompt": seg, "descr_list": []} for i, seg in enumerate(character_segments)
    }

    n_characters = len(character_segments)
    wlw_matrix = np.zeros([n_characters, n_characters], dtype=bool)
    wlw_matrix = np.eye(n_characters, n_characters)
    wlw_matrix = np.repeat(wlw_matrix[..., np.newaxis], frame_num, axis=-1)
    wlw_matrix = torch.from_numpy(wlw_matrix)

    bias_kwargs = {
        "control_prompts": control_prompts,
        "timestep_bias_schedule": timestep_bias_schedule,
        "blocks_bias_schedule": blocks_bias_schedule,
        "face_masks": masks,
        "wlw_matrix": wlw_matrix,
        "beta": config["beta"],
    }
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    video: torch.Tensor
    extra_data: dict[str, torch.Tensor]
    video, extra_data = wan_i2v.generate(  # type: ignore
        prompt,
        img,
        bias_kwargs,
        max_area=target_size[0] * target_size[1],
        sampling_steps=sampling_steps,
        frame_num=frame_num,
    )

    return video, extra_data


def generate_inference_data(prompts_data_list, img_dir):
    output = []
    for i, prompt_data in enumerate(prompts_data_list):
        bboxes = prompt_data["bboxes"]
        masks = torch.stack([create_mask_from_bbox(bbox, target_size) for bbox in bboxes])

        action_prompt_data = prompt_data["action_prompt"]
        segments = action_prompt_data["segments"]
        segment_mask = action_prompt_data["mask"]
        character_segments = [seg for is_char, seg in zip(segment_mask, segments) if is_char]
        assert len(bboxes) == len(character_segments), f"error in prompt #{i}"
        prompt = " ".join(segments)

        img_path = str(img_dir / prompt_data["img_path"])
        img = load_image(img_path)
        assert img.size[::-1] == target_size

        output.append(
            {"img": img, "masks": masks, "prompt": prompt, "character_segments": character_segments}
        )

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--checkpoint-path", default=Path("./weights/"), type=Path)
    parser.add_argument("--t5-cpu", action="store_true")

    args = parser.parse_args()

    param_grid = {"beta": np.linspace(0.0, 1.0, 5).tolist()}

    with open(args.prompts_file) as f:
        prompts_data_list = json.load(f)

    # The point of this function is just to do the necessary checks in advance, before the inference
    infer_data_list = generate_inference_data(prompts_data_list, args.prompts_file.parent)

    wan_i2v = WanI2V(
        config=i2v_14B,
        checkpoint_dir=str(args.checkpoint_path / "Wan2.1-I2V-14B-480P"),
        device_id=0,
        t5_cpu=args.t5_cpu,
    )

    for prompt_data, infer_data in zip(prompts_data_list, infer_data_list):
        img = infer_data["img"]
        masks = infer_data["masks"]
        prompt = infer_data["prompt"]
        character_segments = infer_data["character_segments"]

        for config in ParameterGrid(param_grid):
            config["prompt"] = prompt_data

            folder_name = get_folder_name(config)
            output_path = args.output_path / folder_name
            output_path.mkdir(exist_ok=True)
            config_path = output_path / "config.json"
            video_path = output_path / "video.mp4"

            if config_path.exists() and video_path.exists():
                print(f"The video for the prompt '{prompt}' the was already generated. Skipping...")
                continue

            video, extra_data = run_inference(
                wan_i2v, prompt, img, character_segments, masks, config
            )
            video = (video * 0.5 + 0.5).clamp(0, 1)
            video = rearrange(video, "C T H W -> T H W C")
            video = video.cpu().numpy()

            export_to_video(list(video), video_path, fps=16)
            with config_path.open("w") as f:
                json.dump(config, f)

            # TODO: create also debug video?
            simil_masks = extra_data["simil_masks"]


if __name__ == "__main__":
    main()
