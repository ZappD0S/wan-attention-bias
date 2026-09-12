"""Frozen mask preprocessing and provenance evidence for manifest materialization."""

import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter


def process_subject_masks(masks, processing):
    masks = np.ascontiguousarray(masks, dtype=np.bool_)
    masks &= ~(masks.sum(axis=0) > 1)
    sigma = processing["gaussian_sigma"]
    if sigma:
        if processing.get("variant") != "legacy_boolean_gaussian_v1":
            raise ValueError("dynamic masks require the frozen legacy Boolean Gaussian variant")
        masks = gaussian_filter(masks, sigma=sigma, axes=(1, 2), output=np.bool_)
    masks = np.ascontiguousarray(masks, dtype=np.bool_)
    masks &= ~(masks.sum(axis=0) > 1)
    if not masks.reshape(masks.shape[0], -1).any(axis=1).all():
        raise ValueError("every resolved subject mask must be nonempty")
    return masks


def resolved_mask_evidence(masks, actor_order, processing):
    masks = np.ascontiguousarray(masks, dtype=np.bool_)
    return {
        "shape": list(masks.shape),
        "dtype": "bool",
        "processing_variant": processing["variant"],
        "combined_sha256": hashlib.sha256(masks.tobytes()).hexdigest(),
        "actors": {
            actor_id: {
                "true_pixels": int(masks[index].sum()),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(masks[index]).tobytes()
                ).hexdigest(),
            }
            for index, actor_id in enumerate(actor_order)
        },
    }
