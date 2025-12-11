import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from utils import create_bbox_from_mask

COLORS = [
    (0, 0, 255),  # Red
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (255, 255, 0),  # Cyan
]

GREY = (128, 128, 128)

Bbox = tuple[float, float, float, float]


def draw_boxes(img: Image.Image, boxes: list[Bbox]):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        color = COLORS[i % len(COLORS)]
        box = [round(x, 2) for x in box]
        draw.rectangle(box, outline=color, width=2)

    return img


def draw_masks(img: Image.Image, masks: list[np.ndarray | torch.Tensor], alpha=0.6) -> Image.Image:
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    overlay = img_bgr.copy()

    for i, mask in enumerate(masks):
        color = COLORS[i % len(COLORS)]

        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # ensure mask is boolean
        mask = mask.astype(bool)
        overlay[mask] = color

    img_with_masks = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    img_with_masks = cv2.cvtColor(img_with_masks, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_with_masks)


def unscale(tensor: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
    # tensor shape: (..., T, H, W)
    batch_dims = tensor.shape[:-3]

    T, H, W = tensor.shape[-3:]
    tensor = tensor.view(-1, 1, T, H, W)

    interpolated_tensor = F.interpolate(tensor, size=target_size, mode="nearest")

    output_shape = batch_dims + target_size
    interpolated_tensor = interpolated_tensor.view(output_shape)

    return interpolated_tensor


def write_video_masks(video: np.ndarray, save_file, face_masks, fps=16):
    _, h, w, _ = video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(str(save_file), fourcc, fps, (w, h))

    video_frames = [(frame * 255).astype(np.uint8) for frame in video]
    for frame, frame_face_masks in zip(video_frames, face_masks, strict=True):
        frame = np.ascontiguousarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        bboxes = [create_bbox_from_mask(mask) for mask in frame_face_masks]

        for i, bbox in enumerate(bboxes):
            cv2.rectangle(
                frame,
                [bbox[0], bbox[1]],
                [bbox[2], bbox[3]],
                COLORS[i],
                thickness=2,
            )

        writer.write(frame)

    writer.release()


def write_video_wlw_masks(video: np.ndarray, save_file, wlw, face_masks, fps=16):
    _, h, w, _ = video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(str(save_file), fourcc, fps, (w, h))

    video_frames = [(frame * 255).astype(np.uint8) for frame in video]
    for frame, frame_wlw, frame_face_masks in zip(video_frames, wlw, face_masks, strict=True):
        frame = np.ascontiguousarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        bboxes = [create_bbox_from_mask(mask) for mask in frame_face_masks]

        for bbox, frame_lw in zip(bboxes, frame_wlw, strict=True):
            for i, is_looking in enumerate(frame_lw):
                if is_looking:
                    cv2.rectangle(
                        frame,
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[3]],
                        COLORS[i],
                        thickness=2,
                    )
                    break
            else:
                cv2.rectangle(
                    frame,
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[3]],
                    GREY,
                    thickness=2,
                )

        writer.write(frame)

    writer.release()


def write_debug_video_attn(video, save_file, attn_weights, fps=16):
    def _build_attn_cmap(attn):
        attn_uint8 = (attn * 255).astype(np.uint8)
        return cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)

    _, h, w, _ = video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(str(save_file), fourcc, fps, (w, h))
    alpha = 0.35  # 35% opacity

    attn_weights = attn_weights / attn_weights.max()

    for frame, attn in zip(video, attn_weights, strict=True):
        frame = np.ascontiguousarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        attn = np.ascontiguousarray(attn.numpy())
        attn_cmap = _build_attn_cmap(attn)
        frame = cv2.addWeighted(attn_cmap, alpha, frame, 1 - alpha, 0)

        writer.write(frame)

    writer.release()
