# %%
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.loading_utils import load_image
from einops import rearrange
from wan.configs.wan_i2v_14B import i2v_14B
from wan.regional_prompt import WanI2V

root_path = Path.cwd().parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from debug_utils import unscale
from multi_sample_inference.multi_sample_inference import run_inference
from utils import create_mask_from_bbox, normalize_video_tensor

# %%

TARGET_SIZE = (480, 832)
IMG_DIR = Path("../image_prompt_generation/")
PROMPT_DATA: dict[str, Any] = {
    "safeguard_suffix": "The scene is filmed as a continuous shot with a static camera, ensuring no cuts and no new characters or objects entering.",
    "action_prompts": {
        "default": {
            "segments": [
                [
                    "On the left,",
                    "the golden retriever is barking loudly",
                    "while on the right,",
                    "the golden retriever is panting with its tongue out.",
                ]
            ],
            "mask": [[0, 1, 0, 1]],
        },
        "no_locative": {
            "segments": [
                [
                    "A golden retriever is barking loudly",
                    "and",
                    "a golden retriever is panting with its tongue out.",
                ]
            ],
            "mask": [[1, 0, 1]],
        },
        "split_sentences": {
            "segments": [
                ["A golden retriever is barking loudly."],
                ["A golden retriever is panting with its tongue out."],
            ],
            "mask": [[1], [1]],
        },
    },
    "bboxes": [
        [29.460861206054688, 46.01702117919922, 354.6479797363281, 462.7904968261719],
        [485.2065734863281, 30.857032775878906, 824.189453125, 455.1943054199219],
    ],
    "img_path": "images/0.png",
}

config = {
    "bias_method": "none",
}

PROMPT_TYPE = "default"

# %%
wan_i2v = WanI2V(
    config=i2v_14B,
    checkpoint_dir="../weights/Wan2.1-I2V-14B-480P/",
    device_id=0,
    t5_cpu=True,
)
# %%

action_prompt_data = PROMPT_DATA["action_prompts"][PROMPT_TYPE]


safeguard_suffix = PROMPT_DATA["safeguard_suffix"]
segment_lists = [[safeguard_suffix]] + action_prompt_data["segments"]
prompt_sentences = [" ".join(segments) for segments in segment_lists]

segment_masks = [[0]] + action_prompt_data["mask"]
character_segments = [
    [seg for is_char, seg in zip(mask_row, segs) if is_char]
    for mask_row, segs in zip(segment_masks, segment_lists)
]

img_path = IMG_DIR / PROMPT_DATA["img_path"]
img = load_image(str(img_path))

bboxes = PROMPT_DATA["bboxes"]
masks = torch.stack([torch.from_numpy(create_mask_from_bbox(bbox, TARGET_SIZE)) for bbox in bboxes])

video, extra_data = run_inference(wan_i2v, prompt_sentences, img, character_segments, masks, config)
video_norm = normalize_video_tensor(video.cpu().numpy())

simil_masks = extra_data["simil_masks"].numpy()
# B, N_denoising_steps, N_blocks, N_char, T, H, W

np.savez("output.npz", video_norm=video_norm, simil_masks=simil_masks)


# %%


data = np.load("output.npz")
video_norm = data["video_norm"]
simil_masks = data["simil_masks"]

export_to_video(list(video_norm), "video.mp4", fps=16)

video_norm = torch.from_numpy(video_norm)
simil_masks = torch.from_numpy(simil_masks)


def draw_mask_sequence(frame, masks, alpha=0.5, thickness=1, cmap_name="viridis", do_opening=False):
    frame_uint8 = (np.asarray(frame) * 255).astype(np.uint8)
    masks = np.asarray(masks)

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, len(masks))]

    overlay = frame_uint8.copy()

    for mask, color_rgba in zip(masks, colors):
        color_rgb = (int(color_rgba[0] * 255), int(color_rgba[1] * 255), int(color_rgba[2] * 255))
        mask_uint8 = mask.astype(np.uint8) * 255

        if do_opening:
            kernel = np.ones((3, 3), np.uint8)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_rgb, thickness, lineType=cv2.LINE_AA)

    output = cv2.addWeighted(overlay, alpha, frame_uint8, 1 - alpha, 0)

    return output.astype(np.float32) / 255.0


frame_num, h, w, _ = video_norm.shape

# remove batch dim
simil_masks = simil_masks.squeeze(0)

# average over blocks
# simil_masks = simil_masks.float().mean(dim=1) > 0.5

# first block
simil_masks = simil_masks[:, 0]

simil_masks = unscale(simil_masks.float(), (frame_num, h, w)).bool()

simil_masks = rearrange(simil_masks, "diff_step char T H W -> T char diff_step H W")


cmaps = ["viridis", "magma"]

output_frames = []
for frame, frame_masks in zip(video_norm, simil_masks):
    frame_with_outlines = frame.clone()

    for char_masks, cmap in zip(frame_masks, cmaps):
        # let's try to visualize only the mask for the first step
        char_masks = char_masks[:1]
        frame_with_outlines = draw_mask_sequence(
            frame_with_outlines, list(char_masks), cmap_name=cmap
        )

    output_frames.append(frame_with_outlines)

export_to_video(list(output_frames), "video_with_masks.mp4", fps=16)

# %%
