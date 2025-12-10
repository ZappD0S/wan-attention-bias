# %%
import datetime
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2 as transforms
from wan.configs.wan_i2v_14B import i2v_14B
from wan.regional_prompt import WanI2V
from wan.utils.utils import cache_video

from utils import create_mask_from_bbox, normalize_video_tensor
from debug_utils import write_video_wlw_masks

# %%

wan_i2v = WanI2V(
    config=i2v_14B,
    checkpoint_dir="./weights/Wan2.1-I2V-14B-480P/",
    device_id=0,
    t5_cpu=True,
)

# %%

PROMPT_CONFIG = "examples/dogs_no_interaction"

base_path = Path(PROMPT_CONFIG)
img_file = base_path / "original.png"
img = Image.open(img_file).convert("RGB")
original_size = img.size  # (width, height)

with open(base_path / "config.json") as f:
    config = json.load(f)

bbox_format = config["characters"]["bbox_format"]
char_data = sorted(config["characters"]["list"], key=lambda c: c["id"])

for c in char_data:
    del c["id"]

bboxes = [c["bbox"] for c in char_data]


def rescale_img_and_bboxes(img, bboxes, target_size):
    bboxes = torch.tensor(bboxes, dtype=torch.float)
    bboxes = tv_tensors.BoundingBoxes(  # type: ignore
        bboxes,
        format=bbox_format,
        canvas_size=(original_size[1], original_size[0]),  # (height, width)
    )

    format_converter = transforms.ConvertBoundingBoxFormat("XYXY")
    bboxes = format_converter(bboxes)

    transform = transforms.Compose(
        [
            transforms.Resize(min(target_size)),
            transforms.CenterCrop(target_size),
        ]
    )
    transformed_img, transformed_bboxes = transform(img, bboxes)
    transformed_bboxes = [bbox.tolist() for bbox in transformed_bboxes]
    return transformed_img, transformed_bboxes


target_size = (480, 832)
transformed_img, transformed_bboxes = rescale_img_and_bboxes(img, bboxes, target_size)
output_dir = base_path / "output"
output_dir.mkdir(exist_ok=True)
transformed_img.save(output_dir / "resized.png")

# %%

img_bgr = cv2.cvtColor(np.array(transformed_img), cv2.COLOR_RGB2BGR)

for idx, bbox in enumerate(transformed_bboxes):
    x1, y1, x2, y2 = map(round, bbox.tolist())

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.putText(
        img_bgr,
        str(idx),
        (x1 + 5, y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.9,
        color=(0, 0, 255),
        thickness=2,
    )

# Convert back to RGB for displaying with matplotlib
img_arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Display in Jupyter notebook
plt.imshow(img_arr)
plt.axis("off")
plt.show()


w, h = transformed_img.size
face_masks = torch.stack(
    [torch.from_numpy(create_mask_from_bbox(bbox, (h, w))) for bbox in transformed_bboxes]
)

# %%

COLORS = [
    [0, 0, 255],  # Red
    [0, 255, 0],  # Green
    [255, 0, 0],  # Blue
    [255, 255, 0],  # Cyan
]

img_bgr = cv2.cvtColor(np.array(transformed_img), cv2.COLOR_RGB2BGR)

overlay = img_bgr.copy()

for i, mask in enumerate(face_masks.numpy()):
    color = COLORS[i % len(COLORS)]
    overlay[mask] = color

alpha = 0.6  # Transparency factor
highlighted_image = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# %%
frame_num = 81  # default

n_characters = len(bboxes)
wlw_matrix = np.zeros([n_characters, n_characters, frame_num], dtype=bool)
descr_list = [c["descr"].strip() for c in char_data]
control_prompts = {}

for pair_data in config["wlw"]:
    inds = tuple(pair_data["pair"])
    prompt_template = pair_data["prompt_template"].strip()
    pair_descrs = tuple(descr_list[i] for i in inds)
    prompt = prompt_template.format(*pair_descrs)

    control_prompts[inds] = {
        "descr_list": pair_descrs,
        "prompt": prompt,
    }

    time_intervals = pair_data["time_intervals"]

    if not time_intervals:
        raise ValueError

    for intv in time_intervals:
        assert (0.0 <= intv[0]) and (intv[1] <= 1.0)
        intv_inds = (round(t * frame_num) for t in intv)

        idx = (inds * 2 if len(inds) == 1 else inds) + (slice(*intv_inds),)
        wlw_matrix[idx] = True

wlw_matrix = torch.from_numpy(wlw_matrix)

#           ┌───────── Observed ─────────┐
#           │         A            B     │
# ┌─────────┼────────────────────────────┤
# │         │   ┌───────────┬───────────┐│
# │         │ A │    i=0    │    i=1    ││
# │         │   │    j=0    │    j=0    ││
# │Observer │   ├───────────┼───────────┤│
# │         │ B │    i=1    │    i=1    ││
# │         │   │    j=0    │    j=1    ││
# │         │   └───────────┴───────────┘│
# └─────────┴────────────────────────────┘


# %%

negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
base_prompt = config["base_prompt"]

sampling_steps = 40

# TODO: maybe put this in the config file as well?
timestep_bias_schedule = torch.zeros(sampling_steps, dtype=torch.bool)
# timestep_bias_schedule[sampling_steps // 2 :] = True
timestep_bias_schedule[:] = True
# timestep_bias_schedule[10:30] = True

num_layers = wan_i2v.model.num_layers
blocks_bias_schedule = torch.zeros(num_layers, dtype=torch.bool)
# blocks_bias_schedule[:max(1, int(num_layers * 3 / 4))] = True
blocks_bias_schedule[:] = True

bias_kwargs = {
    "control_prompts": control_prompts,
    "timestep_bias_schedule": timestep_bias_schedule,
    "blocks_bias_schedule": blocks_bias_schedule,
    "face_masks": face_masks,
    "wlw_matrix": wlw_matrix,
    "beta": 1.0,
}

torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()
video: torch.Tensor
video, extra_data = wan_i2v.generate(  # type: ignore
    base_prompt,
    transformed_img,
    bias_kwargs,
    max_area=target_size[0] * target_size[1],
    n_prompt=negative_prompt,
    sampling_steps=sampling_steps,
    frame_num=frame_num,
)

# %%

simil_masks = extra_data["simil_masks"]
attn_weights_map = extra_data["attn_weights_map"]


# %%


video_norm = normalize_video_tensor(video.cpu().numpy())


# %%
h, w = video.shape[-2:]

face_masks = simil_masks[0, -1].float().mean(dim=0) > 0.5
# face_masks = simil_masks[0, -1].float().mean(dim=0) * 255
# face_masks = simil_masks[0, -1, -1]
# face_masks = simil_masks[0].float().mean(dim=[0,1]) > 0.5
# face_masks = simil_masks[0].float().mean(dim=[0,1]) * 255


def unscale(tensor):
    # tensor shape: (..., T, H, W)
    batch_dims = tensor.shape[:-3]

    T, H, W = tensor.shape[-3:]
    tensor = tensor.view(-1, 1, T, H, W)

    interpolated_tensor = F.interpolate(tensor, size=(frame_num, h, w), mode="nearest")

    output_shape = batch_dims + (frame_num, h, w)
    interpolated_tensor = interpolated_tensor.view(output_shape)

    return interpolated_tensor


face_masks = unscale(face_masks.float()).bool()

attn_weights_map = attn_weights_map.copy()
for inds, attn_weights in attn_weights_map.items():
    *_, last_attn_weights = attn_weights
    # remove batch dim
    last_attn_weights = last_attn_weights.squeeze(1)
    attn_weights_map[inds] = unscale(last_attn_weights)


# %%

now = datetime.datetime.now()
video_output_dir = output_dir / now.strftime(r"%Y-%m-%d_%H-%M-%S")
video_output_dir.mkdir()

write_video_wlw_masks(
    video_norm,
    video_output_dir / "people_masks.mp4",
    wlw_matrix.permute(2, 0, 1),
    face_masks.transpose(0, 1),
    fps=4,
)


# %%

video_file = cache_video(
    tensor=video.unsqueeze(0),
    save_file="example.mp4",
    fps=16,
    nrow=1,
    normalize=True,
    value_range=(-1, 1),
)
