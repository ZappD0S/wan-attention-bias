import gc
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from accelerate import Accelerator
from diffusers import FluxPipeline
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import convex_hull_image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

from debug_utils import draw_masks

from .big_lama import inpaint_image, load_lama_model

IOU_THRESHOLD = 0.5
BASE_DIR = Path("./image_prompt_generation/")
OUTPUT_FILE = BASE_DIR / "prompts_modified.json"
WEIGHTS_DIR = Path("./weights/")
IMG_DIR = BASE_DIR / "images"
MODEL_ID_DINO = "IDEA-Research/grounding-dino-base"
MODEL_ID_FLUX = "black-forest-labs/FLUX.1-dev"

os.environ["HF_HOME"] = str(WEIGHTS_DIR)


def load_dataset(directory: Path) -> tuple[list[dict], str]:
    """
    Iterates through JSON files in the directory to aggregate prompt data.
    """
    prompts_data = []
    suffix = ""

    for prompt_file_path in directory.glob("*.json"):
        # Skip the output file to avoid reading what we are writing
        if OUTPUT_FILE.exists() and OUTPUT_FILE.samefile(prompt_file_path):
            continue

        with prompt_file_path.open() as f:
            try:
                data = json.load(f)
                # Capture the safeguard suffix from the first file that has it
                if not suffix and "safeguard_suffix" in data:
                    suffix = data["safeguard_suffix"]

                if "dataset" in data:
                    prompts_data += data["dataset"]
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {prompt_file_path}")

    return prompts_data, suffix


def load_flux_model() -> FluxPipeline:
    """Initializes the Flux Image Generation Pipeline."""
    print(f"Loading Flux model: {MODEL_ID_FLUX}...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID_FLUX,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )
    return pipe


def load_dino_model() -> tuple[Any, Any, torch.device]:
    """Initializes the Grounding DINO model and processor."""
    print(f"Loading Grounding DINO: {MODEL_ID_DINO}...")
    device = Accelerator().device
    processor = AutoProcessor.from_pretrained(MODEL_ID_DINO)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID_DINO).to(device)
    return processor, model, device


def generate_image(
    pipe: FluxPipeline, prompt: str, seed: int, width: int = 832, height: int = 480
) -> Image.Image:
    """Generates an image using the Flux pipeline."""
    generator = torch.Generator("cpu").manual_seed(seed)
    image = pipe(
        prompt,
        width=width,
        height=height,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=generator,
    ).images[0]  # ty:ignore[call-non-callable]

    return image


def detect_objects(
    image: Image.Image, text_prompts: list[str], processor: Any, model: Any, device: torch.device
) -> dict[str, Any]:
    """Runs Grounding DINO to detect objects in the image based on text prompts."""
    clean_prompts = [seg.rstrip(" .") for seg in text_prompts]

    inputs = processor(images=image, text=clean_prompts, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.29,
        text_threshold=0.1,
        target_sizes=[image.size[::-1]],
    )
    return results[0]


def filter_and_sort_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: list[str],
    target_count: int,
    iou_threshold: float = 0.5,
) -> tuple[list[list[float]], list[float], list[str]]:
    """
    Applies Non-Maximum Suppression (NMS) and filters for the top-k highest scoring boxes.
    """
    # Remove overlapping boxes
    keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)

    # Keep only the target_count boxes with highest score
    # Note: sorted sorts ascending, so we slice from the end
    keep_indices = sorted(keep_indices, key=lambda i: scores[i])
    keep_indices = keep_indices[-target_count:]

    final_boxes = [boxes[i].tolist() for i in keep_indices]
    final_scores = [scores[i].item() for i in keep_indices]
    final_labels = [labels[i] for i in keep_indices]

    return final_boxes, final_scores, final_labels


def has_overlap(boxes: list[list[float]]) -> bool:
    """Check if any pair of boxes has IoU > 0 using torchvision.ops.box_iou."""
    if len(boxes) <= 1:
        return False
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    iou_matrix = torchvision.ops.box_iou(boxes_tensor, boxes_tensor)
    return bool((iou_matrix.triu(diagonal=1) > 0).any().item())


def enlarge_bboxes_adaptive(
    boxes: list[list[float]],
    image_size: tuple[int, int],
    base_factor: float = 0.5,
    min_factor: float = 0.0,
    step: float = 0.05,
) -> list[list[float]]:
    """
    Enlarges bounding boxes, skipping if already overlapping.

    1. Check if any boxes already overlap (IoU > 0)
    2. If overlapping, return unchanged
    3. If not, iteratively enlarge with factor reduction on overlap
    """
    if len(boxes) <= 1:
        return boxes

    if has_overlap(boxes):
        return boxes

    img_w, img_h = image_size
    factor = base_factor

    while factor >= min_factor:
        enlarged = []
        for x1, y1, x2, y2 in boxes:
            w = x2 - x1
            h = y2 - y1

            dx = w * factor / 2
            dy = h * factor / 2

            new_box = [
                max(0, x1 - dx),
                max(0, y1 - dy),
                min(img_w, x2 + dx),
                min(img_h, y2 + dy),
            ]
            enlarged.append(new_box)

        if not has_overlap(enlarged):
            return enlarged

        factor -= step

    return boxes


def draw_bboxes(img: Image.Image, boxes, labels, scores) -> Image.Image:
    """Visualizes bounding boxes on the image."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    for box, score, label_text in zip(boxes, scores, labels, strict=True):
        rounded_box = [round(x, 2) for x in box]
        print(
            f"Detected '{label_text}' with confidence {round(score, 3)} at location {rounded_box}"
        )

        draw.rectangle(rounded_box, outline="red", width=2)

        caption = f"{label_text}: {round(score, 2)}"
        text_bbox = draw.textbbox((rounded_box[0], rounded_box[1]), caption, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position the text
        text_location = [rounded_box[0], rounded_box[1] - text_height]

        # Ensure the text is within the image boundaries
        if text_location[1] < 0:
            text_location[1] = rounded_box[1] + 2

        draw.rectangle(
            (
                text_location[0],
                text_location[1],
                text_location[0] + text_width,
                text_location[1] + text_height,
            ),
            fill="red",
        )
        draw.text(tuple(text_location), caption, fill="white", font=font)

    return img


def process_dataset(
    prompts_list: list[dict],
    flux_pipe: FluxPipeline,
    gd_processor: Any,
    gd_model: Any,
    device,
    force=False,
):
    """Main processing loop: Generates images, detects objects, and updates dataset."""

    outputs = []

    for i, prompt_data in enumerate(prompts_list):
        appearance = prompt_data["appearance_prompt"]
        segments = appearance["segments"][0]
        mask = appearance["mask"][0]

        full_prompt = " ".join(segments)
        character_segments = [seg for is_char, seg in zip(mask, segments, strict=True) if is_char]

        img_subdir = IMG_DIR / str(i)
        img_subdir.mkdir(exist_ok=True)
        img_path = img_subdir / "original.png"

        if not img_path.exists() or force:
            print(f"\nProcessing [{i + 1}/{len(prompts_list)}]: {full_prompt[:50]}...")

            seed = 42 + i
            img = generate_image(flux_pipe, full_prompt, seed)
            img.save(img_path)
        else:
            print("Image already exists. Loading from disk...")
            img = Image.open(img_path)

        raw_results = detect_objects(
            img,
            character_segments,
            gd_processor,
            gd_model,
            device,
        )

        debug_subdir = img_subdir / "debug"
        debug_subdir.mkdir(exist_ok=True)

        debug_all_boxes = draw_bboxes(
            img,
            raw_results["boxes"].tolist(),
            raw_results["text_labels"],
            raw_results["scores"].tolist(),
        )
        debug_all_boxes.save(debug_subdir / "all_boxes_detected.png")

        boxes, scores, labels = filter_and_sort_boxes(
            raw_results["boxes"],
            raw_results["scores"],
            raw_results["text_labels"],
            target_count=len(character_segments),
            iou_threshold=IOU_THRESHOLD,
        )

        if len(boxes) != len(character_segments):
            raise Exception(
                f"ERROR: Wrong number of boxes for #{i}."
                " Expected {len(character_segments)}, got {len(boxes)}"
            )

        img_with_boxes = draw_bboxes(img, boxes, labels, scores)
        img_with_boxes.save(debug_subdir / "original_with_boxes.png")

        # sorting by horizontal center x
        boxes_sorted = sorted(boxes, key=lambda b: 0.5 * (b[2] + b[0]))

        output = prompt_data.copy()
        output["img"] = img
        output["bboxes"] = boxes_sorted

        output["img_paths"] = {}
        output["img_paths"]["original"] = str(img_path)

        outputs.append(output)

    return outputs


def get_masks_from_bboxes(image, bboxes, model, processor, device="cuda"):
    input_boxes = [bboxes]
    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

    with torch.no_grad():
        outputs = model(**inputs)

    raw_masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
    )[0][:, 0, :, :]  # Get best mask (index 0)

    # 1. Create empty black masks
    final_masks = torch.zeros_like(raw_masks)

    # 2. Only copy data inside the bbox (implicit intersection)
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)  # ensure ints
        final_masks[i, y1:y2, x1:x2] = raw_masks[i, y1:y2, x1:x2]

    return final_masks


def create_removal_mask(all_masks, keep_index, dilation_pixels=10):
    h, w = all_masks.shape[1], all_masks.shape[2]
    combined_hull = np.zeros((h, w), dtype=bool)

    for i in range(all_masks.shape[0]):
        if i == keep_index:
            continue

        mask_np = all_masks[i].cpu().numpy().astype(bool)

        if mask_np.any():
            hull = convex_hull_image(mask_np)
            combined_hull |= hull

    final_mask = combined_hull.astype(np.uint8)

    if dilation_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_pixels, dilation_pixels))
        final_mask = cv2.dilate(final_mask, kernel)

    return Image.fromarray(final_mask * 255)


def fill_out_characters(data_list, lama, device: torch.device):
    # TODO: move this somewhere else
    model_id = "facebook/sam2.1-hiera-large"
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)  # ty:ignore[invalid-argument-type]

    outputs = []

    for data in data_list:
        output = data.copy()
        raw_img = output["img"]
        bboxes = output["bboxes"]

        all_masks = get_masks_from_bboxes(raw_img, bboxes, model, processor, device)

        orig_img_path = Path(output["img_paths"]["original"])
        debug_folder = orig_img_path.parent / "debug"
        debug_folder.mkdir(exist_ok=True)

        img_with_masks = draw_masks(raw_img, list(all_masks.numpy()))
        img_with_masks.save(debug_folder / "img_with_masks.png")

        output["img_paths"]["single_char"] = []
        for j in range(len(bboxes)):
            removal_mask = create_removal_mask(all_masks, keep_index=j, dilation_pixels=100)
            removal_mask.save(debug_folder / f"mask_char_{j}.png")

            output_img = inpaint_image(lama, device, raw_img, removal_mask)
            img_path = orig_img_path.parent / f"single_char_{j}.png"
            output_img.save(str(img_path))

            output["img_paths"]["single_char"].append(str(img_path))

        outputs.append(output)

    return outputs


def main():
    IMG_DIR.mkdir(exist_ok=True)

    prompts_data_list, safeguard_suffix = load_dataset(BASE_DIR)

    flux_pipe = load_flux_model()
    gd_processor, gd_model, device = load_dino_model()

    updated_dataset = process_dataset(prompts_data_list, flux_pipe, gd_processor, gd_model, device)
    print("length:", len(updated_dataset))

    del flux_pipe
    gc.collect()
    torch.cuda.empty_cache()

    lama, _ = load_lama_model(WEIGHTS_DIR / "big-lama", device)
    updated_dataset = fill_out_characters(updated_dataset, lama, device)

    # enlarge bboxes
    for prompt_data in updated_dataset:
        raw_img = prompt_data.pop("img")
        boxes = prompt_data["bboxes"]
        boxes = enlarge_bboxes_adaptive(boxes, raw_img.size, base_factor=0.5)
        prompt_data["enlarged_bboxes"] = boxes

    # TODO: do the enlarging here!
    output_data = {"safeguard_suffix": safeguard_suffix, "dataset": updated_dataset}

    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
