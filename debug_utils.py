import numpy as np
import cv2
from utils import create_bbox_from_mask

COLORS = [
    [0, 0, 255],  # Red
    [0, 255, 0],  # Green
    [255, 0, 0],  # Blue
    [255, 255, 0],  # Cyan
]

GREY = [128, 128, 128]


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
