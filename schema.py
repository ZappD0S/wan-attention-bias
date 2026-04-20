from typing import Generic, TypeVar

import msgspec

T = TypeVar("T")


class MaskedTextPrompt(msgspec.Struct):
    """
    Text segments paired with a subject mask.
    Used to assign specific descriptions to specific entities in the video.
    """

    segments: list[list[str]]
    mask: list[list[int]]


class JointActionPrompt(msgspec.Struct):
    """
    A decomposed prompt structure used when describing multiple
    subjects performing actions simultaneously.
    """

    general_prompt: str
    segments: list[list[str]]
    mask: list[list[int]]


class ActionPromptSuite(msgspec.Struct):
    """
    Variations of motion/action descriptions used to provide
    temporal guidance for the video generation.
    """

    default: MaskedTextPrompt
    no_locative: MaskedTextPrompt
    split_sentences: JointActionPrompt


class VideoAssetPaths(msgspec.Struct):
    """
    File system references for the reference image and its
    associated segmentation/character masks.
    """

    original: str
    seg_masks: list[str]
    single_char: list[str]


class VideoSpecification(msgspec.Struct):
    """The base version (Minimal JSON)."""

    appearance_prompt: MaskedTextPrompt
    action_prompts: ActionPromptSuite


class ProcessedVideoSpecification(VideoSpecification):
    """The extended version. Inherits all fields above and adds these as REQUIRED."""

    bboxes: list[tuple[float, float, float, float]]
    img_paths: VideoAssetPaths
    enlarged_bboxes: list[tuple[float, float, float, float]]


class VideoGenerationDataset(msgspec.Struct, Generic[T]):
    """
    A generic container.
    Can be VideoGenerationDataset[VideoSpecification]
    or VideoGenerationDataset[ProcessedVideoSpecification].
    """

    safeguard_suffix: str
    dataset: list[T]
