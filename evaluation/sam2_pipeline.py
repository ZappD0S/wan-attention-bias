import gc
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

from utils import suppress_tqdm


def load_video_frames(video_path: Path, target_fps: int) -> tuple[list[np.ndarray], int]:
    """Reads a video, returns RGB frames and the frame-skip stride."""
    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, int(orig_fps / target_fps))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    return frames, stride


def create_video_writer(path: Path, fps: int, size: int) -> cv2.VideoWriter:
    """Creates a single OpenCV video writer."""
    path.parent.mkdir(exist_ok=True, parents=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    return cv2.VideoWriter(str(path), fourcc, fps, (size, size))


def get_crop_and_mask(
    frame: np.ndarray, mask: np.ndarray, margin: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extracts the cropped image and the corresponding local mask slice."""
    if not mask.any():
        return None

    y_idx, x_idx = np.where(mask)
    y1, y2, x1, x2 = y_idx.min(), y_idx.max(), x_idx.min(), x_idx.max()

    pad_h, pad_w = int((y2 - y1) * margin), int((x2 - x1) * margin)
    y1, y2 = max(0, y1 - pad_h), min(frame.shape[0], y2 + pad_h)
    x1, x2 = max(0, x1 - pad_w), min(frame.shape[1], x2 + pad_w)

    crop = frame[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    return (crop, crop_mask) if crop.size > 0 else None


def apply_mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Applies a red tint to the masked area of an image."""
    overlay_img = image.copy()
    red_layer = np.zeros_like(image)
    red_layer[:] = (255, 0, 0)  # RGB Red

    blended = cv2.addWeighted(image, 1 - alpha, red_layer, alpha, 0)
    overlay_img[mask] = blended[mask]
    return overlay_img


def finalize_frame(image: np.ndarray, output_size: int) -> np.ndarray:
    """Pads image to square, resizes, and converts RGB to BGR for writing."""
    h, w = image.shape[:2]
    dim = max(h, w)

    square = np.zeros((dim, dim, 3), dtype=np.uint8)
    ax, ay = (dim - w) // 2, (dim - h) // 2
    square[ay : ay + h, ax : ax + w] = image

    final = cv2.resize(square, (output_size, output_size))
    return cv2.cvtColor(final, cv2.COLOR_RGB2BGR)


def get_bbox(
    mask: np.ndarray, margin: float, h_lim: int, w_lim: int
) -> tuple[int, int, int, int] | None:
    """
    Calculates a bounding box (y1, y2, x1, x2) for a boolean mask.
    Applies a margin and ensures coordinates are within (h_lim, w_lim).
    """
    if not mask.any():
        return None

    # Find the pixel indices where the mask is True
    y_idx, x_idx = np.where(mask)
    y1, y2, x1, x2 = y_idx.min(), y_idx.max(), x_idx.min(), x_idx.max()

    # Calculate padding based on the current height and width of the object
    ph = int((y2 - y1) * margin)
    pw = int((x2 - x1) * margin)

    # Apply padding and clamp to the frame boundaries (0 to height/width)
    return (max(0, y1 - ph), min(h_lim, y2 + ph), max(0, x1 - pw), min(w_lim, x2 + pw))


def run_sam2_pipeline(
    video_path: Path,
    output_dir: Path,
    bboxes: list[list[float]],
    model_id: str,
    target_fps: int,
    margin: float,
) -> list[Path]:
    output_dir.mkdir(exist_ok=True, parents=True)
    frames, stride = load_video_frames(video_path, target_fps)
    valid_frames = frames[::stride]
    h_orig, w_orig = frames[0].shape[:2]

    # 1. Inference: Get masks for ALL objects
    predictor = SAM2VideoPredictor.from_pretrained(model_id)
    saved_masks_dict = {i: [] for i in range(len(bboxes))}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16), suppress_tqdm():
        state = predictor.init_state(str(video_path))
        for i, box in enumerate(bboxes):
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=i, box=box)
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_idx % stride == 0:
                for obj_id, logit in zip(obj_ids, mask_logits, strict=True):
                    saved_masks_dict[obj_id].append((logit > 0).cpu().numpy().squeeze())

    del obj_id
    saved_masks = [saved_masks_dict[k] for k in sorted(saved_masks_dict)]
    del predictor, state
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Rendering with Cross-Object Masking
    clean_video_paths = []
    num_frames = len(valid_frames)

    for i, masks in enumerate(saved_masks):
        frame_boxes = [get_bbox(m, margin, h_orig, w_orig) for m in masks]
        valid_boxes = [b for b in frame_boxes if b is not None]
        if not valid_boxes:
            raise RuntimeError(
                f"Failed to extract any valid bounding boxes for Object {i} "
                f"in video '{video_path}'. Aborting script."
            )

        # Global BBox calculation
        gy1, gy2 = min(b[0] for b in valid_boxes), max(b[1] for b in valid_boxes)
        gx1, gx2 = min(b[2] for b in valid_boxes), max(b[3] for b in valid_boxes)
        gh, gw = (gy2 - gy1) // 2 * 2, (gx2 - gx1) // 2 * 2
        gy2, gx2 = gy1 + gh, gx1 + gw

        clean_path = output_dir / f"obj_{i}.mp4"
        clean_video_paths.append(clean_path)
        writer = cv2.VideoWriter(
            str(clean_path),
            cv2.VideoWriter_fourcc(*"mp4v"),  # ty:ignore[unresolved-attribute]
            target_fps,
            (gw, gh),
        )

        for t in range(num_frames):
            # A. Slice original frame to global dimensions
            canvas = valid_frames[t][gy1:gy2, gx1:gx2].copy()

            # B. Mask out ALL OTHER objects at this timestamp
            for j, other_masks in enumerate(saved_masks):
                if j == i:
                    continue
                # Crop the other object's mask to the current global canvas
                other_mask_cropped = other_masks[t][gy1:gy2, gx1:gx2]
                canvas[other_mask_cropped] = 0  # Black out other object

            # C. Apply the current object's visibility window
            f_box = frame_boxes[t]
            if f_box is not None:
                ly1, ly2, lx1, lx2 = [
                    c - offset for c, offset in zip(f_box, [gy1, gy1, gx1, gx1], strict=True)
                ]

                # Create visibility mask for this frame's window
                v_mask = np.zeros((gh, gw), dtype=bool)
                v_mask[max(0, ly1) : ly2, max(0, lx1) : lx2] = True
                canvas[~v_mask] = 0
            else:
                canvas[:] = 0

            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        writer.release()

    return clean_video_paths
