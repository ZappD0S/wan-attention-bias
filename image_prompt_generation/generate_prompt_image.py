import gc
from pathlib import Path
from typing import Any

import cv2
import msgspec
import numpy as np
import torch
import torchvision
from accelerate import Accelerator
from diffusers import FluxPipeline
from msgspec.structs import asdict
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import convex_hull_image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

from schema import (
    ProcessedVideoSpecification,
    VideoAssetPaths,
    VideoGenerationDataset,
    VideoSpecification,
)
from utils import create_mask_from_bbox

from .big_lama import inpaint_image, load_lama_model

IOU_THRESHOLD = 0.5
BASE_DIR = Path("./image_prompt_generation/")
OUTPUT_FILE = BASE_DIR / "prompts_modified.json"
IMG_DIR = BASE_DIR / "images"
MODEL_ID_DINO = "IDEA-Research/grounding-dino-base"
MODEL_ID_FLUX = "black-forest-labs/FLUX.1-dev"


def load_dataset(directory: Path) -> VideoGenerationDataset[VideoSpecification]:
    """
    Decodes JSON files and aggregates them into a single VideoGenerationDataset object.
    """
    all_specs = []
    global_suffix = ""

    for prompt_file_path in directory.glob("*.json"):
        if OUTPUT_FILE.exists() and OUTPUT_FILE.samefile(prompt_file_path):
            continue

        with prompt_file_path.open("rb") as f:
            try:
                data = msgspec.json.decode(
                    f.read(), type=VideoGenerationDataset[VideoSpecification]
                )
                if not global_suffix:
                    global_suffix = data.safeguard_suffix
                all_specs.extend(data.dataset)
            except Exception as e:
                print(f"Warning: Could not decode {prompt_file_path}: {e}")

    return VideoGenerationDataset(safeguard_suffix=global_suffix, dataset=all_specs)


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

    w, h = image.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.29,
        text_threshold=0.1,
        target_sizes=[(h, w)],
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


def has_overlap(boxes: list[tuple[float, float, float, float]]) -> bool:
    """Check if any pair of boxes has IoU > 0 using torchvision.ops.box_iou."""
    if len(boxes) <= 1:
        return False
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    iou_matrix = torchvision.ops.box_iou(boxes_tensor, boxes_tensor)
    return bool((iou_matrix.triu(diagonal=1) > 0).any().item())


def enlarge_bboxes_adaptive(
    boxes: list[tuple[float, float, float, float]],
    image_size: tuple[int, int],
    base_factor: float = 0.5,
    min_factor: float = 0.0,
    step: float = 0.05,
) -> list[tuple[float, float, float, float]]:
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

    img_h, img_w = image_size
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
    dataset: VideoGenerationDataset[VideoSpecification],
    flux_pipe: FluxPipeline,
    gd_processor: AutoProcessor,
    gd_model: AutoModelForZeroShotObjectDetection,
    device: torch.device,
    force: bool = False,
) -> list[tuple[ProcessedVideoSpecification, Image.Image]]:
    """
    Main processing loop: Generates images, detects objects, saves debug visuals,
    and upgrades 'VideoSpecification' objects to 'ProcessedVideoSpecification'.
    """

    processed_pairs: list[tuple[ProcessedVideoSpecification, Image.Image]] = []

    for i, spec in enumerate(dataset.dataset):
        appearance = spec.appearance_prompt
        segments = appearance.segments[0]
        mask = appearance.mask[0]

        full_prompt = " ".join(segments)

        character_segments = [seg for is_char, seg in zip(mask, segments, strict=True) if is_char]

        img_subdir = IMG_DIR / str(i)
        img_subdir.mkdir(exist_ok=True, parents=True)
        img_path = img_subdir / "original.png"

        if not img_path.exists() or force:
            print(f"\nProcessing [{i + 1}/{len(dataset.dataset)}]: {full_prompt[:60]}...")
            seed = 42 + i
            img = generate_image(flux_pipe, full_prompt, seed)
            img.save(img_path)
        else:
            print(f"Loading existing image for item {i}...")
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

        debug_all_img = draw_bboxes(
            img,
            raw_results["boxes"].tolist(),
            raw_results["text_labels"],
            raw_results["scores"].tolist(),
        )
        debug_all_img.save(debug_subdir / "all_boxes_detected.png")

        boxes, scores, labels = filter_and_sort_boxes(
            raw_results["boxes"],
            raw_results["scores"],
            raw_results["text_labels"],
            target_count=len(character_segments),
            iou_threshold=IOU_THRESHOLD,
        )

        if len(boxes) != len(character_segments):
            raise ValueError(
                f"ERROR: Expected {len(character_segments)} objects for index {i}, "
                f"but found {len(boxes)}. Check detection thresholds."
            )

        img_with_final_boxes = draw_bboxes(img, boxes, labels, scores)
        img_with_final_boxes.save(debug_subdir / "original_with_boxes.png")

        # Sort objects by horizontal center (x) so they map correctly to 'left'/'right' prompts
        boxes_sorted = sorted(boxes, key=lambda b: 0.5 * (b[2] + b[0]))

        tuple_bboxes = [(b[0], b[1], b[2], b[3]) for b in boxes_sorted]

        processed_spec = ProcessedVideoSpecification(
            **asdict(spec),
            bboxes=tuple_bboxes,
            img_paths=VideoAssetPaths(
                original=str(img_path),
                seg_masks=[],  # To be filled in the SAM2 step
                single_char=[],  # To be filled in the Inpainting step
            ),
            enlarged_bboxes=[],  # To be filled at the end of the pipeline
        )

        processed_pairs.append((processed_spec, img))

    return processed_pairs


def compute_segmentation_masks(
    image: Image.Image,
    bboxes: list[tuple[float, float, float, float]],
    model: Sam2Model,
    processor: Sam2Processor,
    device="cuda",
) -> np.ndarray:
    input_boxes = [bboxes]
    inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

    with torch.no_grad():
        outputs = model(**inputs)

    raw_seg_masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
    )[0][:, 0, :, :]  # get best mask (index 0)

    # create empty black masks
    seg_masks = torch.zeros_like(raw_seg_masks)

    # only copy data inside the bbox (implicit intersection)
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)  # ensure ints
        seg_masks[i, y1:y2, x1:x2] = raw_seg_masks[i, y1:y2, x1:x2]

    # TODO: don't hardcode this...
    dilation_pixels = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_pixels, dilation_pixels))

    convex_masks_list = []
    for seg_mask in seg_masks:
        mask_np = seg_mask.cpu().numpy().astype(bool)

        convex_hull = convex_hull_image(mask_np).astype(np.uint8) * 255
        dilated_mask = cv2.dilate(convex_hull, kernel)

        convex_masks_list.append(dilated_mask.astype(bool))

    convex_masks = np.stack(convex_masks_list, axis=0)

    return convex_masks


def create_removal_mask(
    seg_masks: np.ndarray, keep_index: int, dilation_pixels: int
) -> Image.Image:
    _, h, w = seg_masks.shape

    # this is the complement of the mask with index `keep_index`
    compl_mask = np.zeros((h, w), dtype=bool)
    for i, seg_mask in enumerate(seg_masks):
        if i == keep_index:
            continue

        compl_mask |= seg_mask

    final_mask = compl_mask.astype(np.uint8) * 255

    if dilation_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_pixels, dilation_pixels))
        final_mask = cv2.dilate(final_mask, kernel)

    return Image.fromarray(final_mask)


def fill_out_characters(
    raw_img: Image.Image, seg_masks: np.ndarray, lama_model, device: torch.device | str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for i in range(len(seg_masks)):
        removal_mask = create_removal_mask(seg_masks, keep_index=i, dilation_pixels=100)
        single_char_img = inpaint_image(lama_model, device, raw_img, removal_mask)

        output.append({"removal_mask": removal_mask, "single_char_img": single_char_img})

    return output


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True, parents=True)

    dataset_root = load_dataset(BASE_DIR)

    flux_pipe = load_flux_model()
    gd_processor, gd_model, device = load_dino_model()

    updated_pairs = process_dataset(dataset_root, flux_pipe, gd_processor, gd_model, device)

    del flux_pipe
    gc.collect()
    torch.cuda.empty_cache()

    model_id = "facebook/sam2.1-hiera-large"
    sam2_processor = Sam2Processor.from_pretrained(model_id)
    sam2_model = Sam2Model.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)  # ty:ignore[invalid-argument-type]

    lama_model, _ = load_lama_model("./weights/big-lama", device)

    for spec, raw_img in updated_pairs:
        bboxes = spec.bboxes
        w, h = raw_img.size

        seg_masks = compute_segmentation_masks(
            raw_img, bboxes, sam2_model, sam2_processor, device=device
        )
        save_dir = Path(spec.img_paths.original).parent

        for j, bbox in enumerate(bboxes):
            mask_np = create_mask_from_bbox(bbox, (h, w))
            mask_img = Image.fromarray(mask_np)

            seg_mask_path = save_dir / f"segmentaion_mask_{j}.png"
            mask_img.save(seg_mask_path)
            spec.img_paths.seg_masks.append(str(seg_mask_path))

        single_char_out = fill_out_characters(raw_img, seg_masks, lama_model, device=device)
        for i, data in enumerate(single_char_out):
            single_char_img = data["single_char_img"]
            single_char_img_path = save_dir / f"single_char_{i}.png"
            single_char_img.save(str(single_char_img_path))
            spec.img_paths.single_char.append(str(single_char_img_path))

        raw_enlarged = enlarge_bboxes_adaptive(bboxes, (h, w), base_factor=0.5)
        spec.enlarged_bboxes = [(b[0], b[1], b[2], b[3]) for b in raw_enlarged]

    processed_dataset_root = VideoGenerationDataset[ProcessedVideoSpecification](
        safeguard_suffix=dataset_root.safeguard_suffix, dataset=[pair[0] for pair in updated_pairs]
    )

    print(f"Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        f.write(msgspec.json.encode(processed_dataset_root))

    print("Done.")


if __name__ == "__main__":
    main()
