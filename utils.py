import contextlib

import cv2
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm as std_tqdm


def create_mask_from_bbox(bbox, image_size):
    left, top, right, bottom = bbox
    height, width = image_size
    assert height <= width

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

    largest_contour = max(contours, key=cv2.contourArea)  # ty:ignore[no-matching-overload]

    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x, y, x + w, y + h)


def normalize_video_tensor(video: np.ndarray, value_range: tuple = (-1, 1)) -> np.ndarray:
    min_val, max_val = value_range
    assert min_val < max_val

    video = np.clip(video, min_val, max_val)

    video = (video - min_val) / (max_val - min_val)
    video = rearrange(video, "C T H W -> T H W C")

    return video


@contextlib.contextmanager
def suppress_tqdm():
    # Save the original __init__ method
    orig_init = std_tqdm.__init__

    # Define a patched version that forces disable=True
    def patched_init(self, *args, **kwargs):
        kwargs["disable"] = True
        orig_init(self, *args, **kwargs)

    # Patch the class
    std_tqdm.__init__ = patched_init  # ty:ignore[invalid-assignment]
    try:
        yield
    finally:
        # Restore the original method
        std_tqdm.__init__ = orig_init
