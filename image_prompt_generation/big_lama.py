import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint


def load_lama_model(checkpoint_path="big-lama", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config_path = f"{checkpoint_path}/config.yaml"
    with open(config_path) as f:
        config = OmegaConf.create(yaml.safe_load(f))

    # Set to inference mode (this is the key!)
    config.training_model.predict_only = True
    config.visualizer.kind = "noop"

    # Load checkpoint
    model_path = f"{checkpoint_path}/models/best.ckpt"
    model = load_checkpoint(config, model_path, strict=False, map_location=device)
    model.freeze()
    model.to(device)

    print(f"LaMa model loaded successfully on {device}")
    return model, device


def inpaint_image(model, device, image, mask):
    # Ensure inputs are PIL Images
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image")
    if not isinstance(mask, Image.Image):
        raise ValueError("mask must be a PIL Image")

    # Convert to RGB and L if needed
    image = image.convert("RGB")
    mask = mask.convert("L")

    # Convert to tensors
    image_array = np.array(image) / 255.0
    mask_array = np.array(mask) / 255.0

    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).float()

    # Move to device
    batch = {"image": image_tensor, "mask": mask_tensor}
    batch = move_to_device(batch, device)

    # Inpaint
    with torch.no_grad():
        result = model(batch)
        inpainted = result["inpainted"][0].permute(1, 2, 0).cpu().numpy()
        inpainted = (inpainted * 255).astype(np.uint8)

    # Convert back to PIL Image
    inpainted_image = Image.fromarray(inpainted)

    return inpainted_image
