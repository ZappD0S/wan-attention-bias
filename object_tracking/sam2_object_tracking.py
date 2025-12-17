from pathlib import Path
import json

import cv2
import numpy as np
import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor


VIDEO_PATH = Path("input_video.mp4")
OUTPUT_PATH = Path("output/")
OUTPUT_PATH.mkdir(exist_ok=True)
INIT_BOX = [100, 200, 350, 450]  # [x_min, y_min, x_max, y_max]

MARGIN = 0.2  # 20% margin
OUTPUT_SIZE = 448  # Recommended for Qwen (multiple of 14)
TARGET_FPS = 5  # Low FPS for VLM efficiency
OVERLAY_ALPHA = 0.5

device = "cuda" if torch.cuda.is_available() else "cpu"
# predictor = Sam2VideoPredictor.from_pretrained(
#     "facebook/sam2-hiera-tiny", device_map=device, torch_dtype=torch.bfloat16
# )
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

cap = cv2.VideoCapture(VIDEO_PATH)
orig_fps = cap.get(cv2.CAP_PROP_FPS)
write_stride = max(1, int(orig_fps / TARGET_FPS))  # Calculate how many frames to skip

frames = []
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
H_orig, W_orig, _ = frames[0].shape

with open("config.json") as f:
    config = json.load(f)

bboxes = config["prompt"]["bboxes"]
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = predictor.init_state(str(VIDEO_PATH))
    for i, box in enumerate(bboxes):
        predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=0, obj_id=i, box=box
        )

    writers = {}
    for i in range(len(bboxes)):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(OUTPUT_PATH / f"obj_{i}.mp4"), fourcc, TARGET_FPS, (OUTPUT_SIZE, OUTPUT_SIZE)
        )
        writers[i] = out

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        # only write specific frames to achieve TARGET_FPS
        if frame_idx % write_stride != 0:
            continue

        for obj_id, mask_logit in zip(obj_ids, mask_logits):
            # TODO: check if squeeze() is necessary
            mask = (mask_logit > 0.0).cpu().numpy().squeeze()

            if mask.any():
                y_idx, x_idx = np.where(mask)
                y1, y2 = y_idx.min(), y_idx.max()
                x1, x2 = x_idx.min(), x_idx.max()

                pad_h, pad_w = int((y2 - y1) * MARGIN), int((x2 - x1) * MARGIN)
                y1, y2 = max(0, y1 - pad_h), min(H_orig, y2 + pad_h)
                x1, x2 = max(0, x1 - pad_w), min(W_orig, x2 + pad_w)

                crop_img = frames[frame_idx][y1:y2, x1:x2]
                crop_mask = mask[y1:y2, x1:x2]

                if crop_img.size > 0:
                    red_layer = np.zeros_like(crop_img)
                    red_layer[:] = (255, 0, 0)

                    blended_crop = cv2.addWeighted(
                        crop_img, 1 - OVERLAY_ALPHA, red_layer, OVERLAY_ALPHA, 0
                    )

                    assert crop_mask.shape == crop_img.shape[:2]
                    crop_img[crop_mask] = blended_crop[crop_mask]

                    # pad to square
                    h, w = crop_img.shape[:2]
                    dim = max(h, w)
                    square = np.zeros((dim, dim, 3), dtype=np.uint8)  # black canvas
                    # center the crop
                    ax, ay = (dim - w) // 2, (dim - h) // 2
                    square[ay : ay + h, ax : ax + w] = crop_img

                    # Resize to target size and write
                    final_frame = cv2.resize(square, (OUTPUT_SIZE, OUTPUT_SIZE))
                    writers[obj_id].write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

for w in writers.values():
    w.release()
