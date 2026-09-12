import numpy as np
import pytest

from multi_sample_inference.mask_contracts import (
    process_subject_masks,
    resolved_mask_evidence,
)


def test_legacy_boolean_gaussian_area_and_evidence_are_frozen():
    masks = np.zeros((2, 31, 31), dtype=bool)
    masks[0, 2:14, 2:14] = True
    masks[1, 17:29, 17:29] = True
    processing = {
        "variant": "legacy_boolean_gaussian_v1",
        "gaussian_sigma": 1.0,
    }
    resolved = process_subject_masks(masks, processing)
    assert resolved.reshape(2, -1).sum(axis=1).tolist() == [16, 16]
    evidence = resolved_mask_evidence(resolved, ["left", "right"], processing)
    assert evidence["processing_variant"] == "legacy_boolean_gaussian_v1"
    assert evidence["actors"]["left"]["true_pixels"] == 16
    assert len(evidence["combined_sha256"]) == 64
    assert len(evidence["actors"]["right"]["sha256"]) == 64


def test_legacy_boolean_gaussian_rejects_empty_resolved_actor():
    masks = np.zeros((1, 9, 9), dtype=bool)
    masks[0, 2:7, 2:7] = True
    with pytest.raises(ValueError, match="nonempty"):
        process_subject_masks(
            masks,
            {
                "variant": "legacy_boolean_gaussian_v1",
                "gaussian_sigma": 1.0,
            },
        )
