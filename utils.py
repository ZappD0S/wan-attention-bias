import numpy as np
import cv2
import torch


def create_mask_from_bbox(bbox, image_size):
    """
    Creates a boolean mask for a bounding box.
    """
    left, top, right, bottom = bbox
    height, width = image_size

    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    # this looks counter-inutitive, but with meshgrid the origin is the top-left corner
    mask = (y_coords >= top) & (y_coords < bottom) & (x_coords >= left) & (x_coords < right)

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
