import numpy as np
import cv2
import torch
from einops import rearrange


def create_mask_from_bbox(bbox, image_size):
    left, top, right, bottom = bbox
    height, width = image_size

    mask = np.zeros((height, width), dtype=bool)

    # cast to int to ensure valid slice indices
    mask[int(top) : int(bottom), int(left) : int(right)] = True

    return mask


def create_bbox_from_mask(mask):
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
        raise ValueError("No contours found!")

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x, y, x + w, y + h)


def normalize_video_tensor(video: np.ndarray, value_range: tuple = (-1, 1)) -> np.ndarray:
    min_val, max_val = value_range
    video = np.clip(video, min_val, max_val)

    video = (video - min_val) / (max_val - min_val)
    video = rearrange(video, "C T H W -> T H W C")

    return video
