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

# %%

wan_i2v = WanI2V(
    config=i2v_14B,
    checkpoint_dir="./weights/Wan2.1-I2V-14B-480P/",
    device_id=0,
    t5_cpu=True,
)

# %%

PROMPT_CONFIG = "examples/2animals"

base_path = Path(PROMPT_CONFIG)
img_file = base_path / "original.png"
img = Image.open(img_file).convert("RGB")
original_size = img.size  # (width, height)

with open(base_path / "config.json") as f:
    config = json.load(f)

bbox_format = config["characters"]["bbox_format"]
char_data = sorted(config["characters"]["list"], key=lambda x: x.pop("id"))

bboxes = [c["bbox"] for c in char_data]
bboxes = torch.tensor(bboxes, dtype=torch.float)
bboxes = tv_tensors.BoundingBoxes(
    bboxes,
    format=bbox_format,
    canvas_size=(original_size[1], original_size[0]),  # (height, width)
)

format_converter = transforms.ConvertBoundingBoxFormat("XYXY")
bboxes = format_converter(bboxes)

target_size = (480, 832)
transform = transforms.Compose(
    [
        transforms.Resize(min(target_size)),
        transforms.CenterCrop(target_size),
    ]
)
transformed_img, transformed_bboxes = transform(img, bboxes)
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

# %%


def create_bbox_mask(bbox, image_size):
    """
    Creates a boolean mask for a bounding box.
    """
    left, top, right, bottom = bbox
    height, width = image_size

    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    # this looks counter-inutitive, but with meshgrid the origin is the top-left corner
    mask = (y_coords >= top) & (y_coords < bottom) & (x_coords >= left) & (x_coords < right)

    return mask


w, h = transformed_img.size
face_masks = torch.stack([create_bbox_mask(bbox, (h, w)) for bbox in transformed_bboxes])

# %%

colors = [
    [0, 0, 255],  # Red
    [0, 255, 0],  # Green
    [255, 0, 0],  # Blue
    [255, 255, 0],  # Cyan
]

img_bgr = cv2.cvtColor(np.array(transformed_img), cv2.COLOR_RGB2BGR)

overlay = img_bgr.copy()

for i, mask in enumerate(face_masks.numpy()):
    color = colors[i % len(colors)]
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
    i, j = pair_data["pair"]
    prompt_template = pair_data["prompt_template"].strip()
    prompt = prompt_template.format(descr_list[i], descr_list[j])

    control_prompts[(i, j)] = {
        "descr_list": [descr_list[i], descr_list[j]],
        "prompt": prompt,
    }

    for t0, t1 in pair_data["time_intervals"]:
        assert (0 <= t0) and (t1 <= 1)
        start, end = [round(t * frame_num) for t in (t0, t1)]
        wlw_matrix[i, j, start:end] = True

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
timestep_bias_schedule = torch.zeros(sampling_steps, dtype=bool)
# timestep_bias_schedule[sampling_steps // 2 :] = True
timestep_bias_schedule[:] = True
# timestep_bias_schedule[10:30] = True

num_layers = wan_i2v.model.num_layers
blocks_bias_schedule = torch.zeros(num_layers, dtype=bool)
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
video, extra_data = wan_i2v.generate(
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
def create_mask_from_bbox(mask):
    """
    Finds the single bounding box that most closely matches a binary mask.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy().astype(np.uint8)
    elif not isinstance(mask, np.ndarray):
        raise TypeError("Input mask must be a numpy array or a torch tensor.")

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x, y, x + w, y + h)


# %%
def normalize_tensor(tensor: torch.Tensor, value_range: tuple = (-1, 1)) -> torch.Tensor:
    tensor = tensor.clamp(min(value_range), max(value_range))

    min_val, max_val = value_range
    tensor = (tensor - min_val) / (max_val - min_val)

    # (C, T, H, W) -> (T, H, W, C)
    tensor = tensor.permute(1, 2, 3, 0)

    return (tensor * 255).type(torch.uint8)


video_norm = normalize_tensor(video.cpu())


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


def write_debug_video_masks(video, save_file, wlw, face_masks, fps=16):
    _, h, w, _ = video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_file), fourcc, fps, (w, h))

    RED = (0, 0, 255)
    BLUE = (255, 0, 0)

    for frame, frame_wlw, frame_face_masks in zip(video, wlw, face_masks, strict=True):
        frame = np.ascontiguousarray(frame.numpy())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        bboxes = [create_mask_from_bbox(mask) for mask in frame_face_masks]

        for bbox, is_looking in zip(bboxes, frame_wlw):
            cv2.rectangle(
                frame,
                [bbox[0], bbox[1]],
                [bbox[2], bbox[3]],
                RED if is_looking else BLUE,
                thickness=2,
            )

        writer.write(frame)

    writer.release()


def write_debug_video_attn(video, save_file, attn_weights, fps=16):
    def _build_attn_cmap(attn):
        attn_uint8 = (attn * 255).astype(np.uint8)
        return cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)

    _, h, w, _ = video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_file), fourcc, fps, (w, h))
    alpha = 0.35  # 35% opacity

    attn_weights = attn_weights / 0.06

    for frame, attn in zip(video, attn_weights, strict=True):
        frame = np.ascontiguousarray(frame.numpy())
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        attn = np.ascontiguousarray(attn.numpy())
        attn_cmap = _build_attn_cmap(attn)
        frame = cv2.addWeighted(attn_cmap, alpha, frame, 1 - alpha, 0)

        writer.write(frame)

    writer.release()


# %%

now = datetime.datetime.now()
video_output_dir = output_dir / now.strftime(r"%Y-%m-%d_%H-%M-%S")
video_output_dir.mkdir()

write_debug_video_masks(
    video_norm,
    video_output_dir / "people_masks.mp4",
    wlw_matrix[[0, 1], [1, 0], :].transpose(0, 1),
    face_masks.transpose(0, 1),
    fps=4,
)

for ab, ab_attn_weights in attn_weights_map.items():
    a, b = ab
    for i in range(2):
        write_debug_video_attn(
            video_norm,
            video_output_dir / f"attn{ab[i]}_{a}_looks_{b}.mp4",
            ab_attn_weights[i],
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
