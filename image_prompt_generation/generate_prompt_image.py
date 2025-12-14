import os
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

import torch
from diffusers import FluxPipeline  # type: ignore
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator

import torchvision

# %%
IOU_THRESHOLD = 0.5

os.environ["HF_HOME"] = "../weights/"

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)

with open("prompts.json") as f:
    prompts_data_list = json.load(f)

model_id = "IDEA-Research/grounding-dino-base"
device = Accelerator().device

gd_processor = AutoProcessor.from_pretrained(model_id)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
img_dir = Path("./images")
img_dir.mkdir(exist_ok=True)
debug_img_dir = img_dir / "debug"
debug_img_dir.mkdir(exist_ok=True)


def draw_bboxes(img: Image.Image, boxes, labels, scores):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, score, labels in zip(boxes, scores, labels):
        box = [round(x, 2) for x in box]
        print(f"Detected '{labels}' with confidence {round(score, 3)} at location {box}")

        draw.rectangle(box, outline="red", width=2)

        label = f"{labels}: {round(score, 2)}"
        text_bbox = draw.textbbox((box[0], box[1]), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position the text
        text_location = [box[0], box[1] - text_height]

        # Ensure the text is within the image boundaries
        if text_location[1] < 0:
            text_location[1] = box[1] + 2  # Position inside the box if it goes off-screen

        draw.rectangle(
            (
                text_location[0],
                text_location[1],
                text_location[0] + text_width,
                text_location[1] + text_height,
            ),
            fill="red",
        )
        draw.text(tuple(text_location), label, fill="white", font=font)

    return img


for i, prompt_data in enumerate(prompts_data_list):
    appearance_prompt_data = prompt_data["appearance_prompt"]
    segments = appearance_prompt_data["segments"]
    prompt = " ".join(segments)

    mask = appearance_prompt_data["mask"]
    character_segments = [seg for is_char_seg, seg in zip(mask, segments) if is_char_seg]

    seed = 42 + i
    generator = torch.Generator("cpu").manual_seed(seed)

    image = pipe(
        prompt,
        width=832,
        height=480,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=generator,
    ).images[0]  # type: ignore

    img_path = img_dir / f"{i}.png"
    image.save(img_path)

    inputs = gd_processor(
        images=image, text=[seg.rstrip(" .") for seg in character_segments], return_tensors="pt"
    ).to(gd_model.device)

    with torch.no_grad():
        outputs = gd_model(**inputs)

    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.29,
        text_threshold=0.15,
        target_sizes=[image.size[::-1]],
    )
    result = results[0]

    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["text_labels"]

    # remove overlapping boxes
    keep_indices = torchvision.ops.nms(boxes, scores, IOU_THRESHOLD)

    # keep only the len(character_segments) boxes with highest score
    keep_indices = sorted(keep_indices, key=lambda i: scores[i])
    keep_indices = keep_indices[-len(character_segments) :]

    boxes = [boxes[i].tolist() for i in keep_indices]
    scores = [scores[i].item() for i in keep_indices]
    labels = [labels[i] for i in keep_indices]

    img_with_boxes = draw_bboxes(image, boxes, labels, scores)

    assert len(boxes) == len(character_segments), f"wrong number of boxes for #{i}"
    img_with_boxes.save(debug_img_dir / f"output_with_boxes_{i}.png")

    def center_y_coord(box):
        return 0.5 * (box[2] + box[0])

    boxes = sorted(boxes, key=center_y_coord)
    prompt_data["bboxes"] = boxes
    prompt_data["img_path"] = str(img_path)


with open("prompts_modified.json", "w") as f:
    prompts_data_list = json.dump(prompts_data_list, f, indent=2)
