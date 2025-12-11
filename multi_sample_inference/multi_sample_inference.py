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
from wan.regional_prompt import WanI2V

from utils import create_mask_from_bbox, normalize_video_tensor
from debug_utils import write_video_masks, draw_boxes, draw_masks, unscale

SAMPLING_STEPS = 40
FRAME_NUM = 81  # default
TARGET_SIZE = (480, 832)


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


def generate_inference_data(prompts_data_list, img_dir):
    output = []
    for i, prompt_data in enumerate(prompts_data_list):
        bboxes = prompt_data["bboxes"]
        masks = torch.stack(
            [torch.from_numpy(create_mask_from_bbox(bbox, TARGET_SIZE)) for bbox in bboxes]
        )

        action_prompt_data = prompt_data["action_prompt"]
        segments = action_prompt_data["segments"]
        segment_mask = action_prompt_data["mask"]
        character_segments = [seg for is_char, seg in zip(segment_mask, segments) if is_char]
        assert len(bboxes) == len(character_segments), f"error in prompt #{i}"
        prompt = " ".join(segments)

        img_path = str(img_dir / prompt_data["img_path"])
        img = load_image(img_path)
        assert img.size[::-1] == TARGET_SIZE

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

    param_grid = [
        {
            "bias_method": "regional_prompting",
            "beta": np.linspace(0.0, 1.0, 5).tolist(),
        },
        {
            "bias_method": "ediff-i",
            "strength": [3.0, 5.0],
        },
    ]

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
            debug_path = output_path / "debug"
            debug_path.mkdir(exist_ok=True)

            if config_path.exists() and video_path.exists():
                print(f"The video for the prompt '{prompt}' the was already generated. Skipping...")
                continue

            img_with_boxes = draw_boxes(img, prompt_data["bboxes"])
            img_with_boxes.save(debug_path / "img_with_boxes.png")

            img_with_masks = draw_masks(img, masks)
            img_with_masks.save(debug_path / "img_with_masks.png")

            video, extra_data = run_inference(
                wan_i2v, prompt, img, character_segments, masks, config
            )
            video_norm = normalize_video_tensor(video.cpu().numpy())

            export_to_video(list(video_norm), video_path, fps=16)
            with config_path.open("w") as f:
                json.dump(config, f, indent=2)

            simil_masks = extra_data["simil_masks"]

            # TODO: is this the best way to do it?
            face_masks = simil_masks[0, -1].float().mean(dim=0) > 0.5

            h, w = video.shape[-2:]
            face_masks = unscale(face_masks.float(), (FRAME_NUM, h, w)).bool()
            write_video_masks(
                video_norm,
                output_path / "video_with_masks.mp4",
                face_masks.transpose(0, 1).cpu().numpy(),
                fps=16,
            )


if __name__ == "__main__":
    main()
