import argparse
import os
import pickle
import time
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils.export_utils import export_to_video
from wan.configs.wan_i2v_14B import i2v_14B
from wan.regional_prompt import WanI2V

from debug_utils import unscale, write_video_soft_masks
from utils import normalize_video_tensor

SAMPLING_STEPS, FRAME_NUM, TARGET_SIZE = 40, 81, (480, 832)


@torch.inference_mode()
def run_inference(wan_i2v, task):
    num_layers = wan_i2v.model.num_layers
    bias_kwargs = {
        "general_prompt": task["general_prompt"],
        "prompt_data_list": [
            {
                "control_prompts": [
                    ((j,), {"prompt": seg, "char_descr_list": []}) for j, seg in enumerate(chars)
                ],
                "single_char_img": task["single_char_imgs"][i]
                if task["single_char_imgs"]
                else None,
            }
            for i, chars in enumerate(task["char_segments_list"])
        ],
        "timestep_bias_schedule": torch.ones(SAMPLING_STEPS, dtype=torch.bool),
        "blocks_bias_schedule": torch.ones(num_layers, dtype=torch.bool),
        "face_masks": task["masks"].to(wan_i2v.device),
        "wlw_matrix": torch.from_numpy(
            np.repeat(
                np.eye(sum(len(c) for c in task["char_segments_list"]))[..., np.newaxis],
                FRAME_NUM,
                axis=-1,
            )
        ).to(wan_i2v.param_dtype),
    } | task["config"]

    return wan_i2v.generate(
        task["prompt_sentences"],
        task["img"],
        bias_kwargs,
        max_area=TARGET_SIZE[0] * TARGET_SIZE[1],
        sampling_steps=SAMPLING_STEPS,
        frame_num=FRAME_NUM,
        offload_model=False,
    )


def save_outputs(video, extra_data, task, max_retries=5):
    """Restored logic to save both the video and the attention masks."""

    def _robust_run(path, func, *args):
        attempts = 0
        while attempts < max_retries:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                return func(*args)
            except FileNotFoundError:
                attempts += 1
                if attempts == max_retries:
                    raise
                time.sleep(1)

    # 1. Save the main Video
    video_norm = normalize_video_tensor(video.cpu().numpy())
    _robust_run(task["video_path"], export_to_video, list(video_norm), task["video_path"], 16)

    # 2. Extract and Save Soft Attention Masks (from extra_data)
    if "simil_masks" in extra_data:
        simil_masks = extra_data["simil_masks"]

        assert simil_masks.shape[0] == 1
        simil_masks = simil_masks[0]

        h, w = video.shape[-2:]
        action_output_path = task["action_output_path"]
        repeat_idx = task["repeat_idx"]

        for step_idx in [0, -1]:
            # average the masks across all dit blocks
            # Result shape: [num_char, T, H, W]
            step_simil_masks = simil_masks[step_idx].float().mean(dim=0)

            # upscale latent-space masks to match the real video resolution
            step_simil_masks = unscale(step_simil_masks, (FRAME_NUM, h, w))

            for char_idx, mask in enumerate(step_simil_masks):
                mask_filename = f"video_{repeat_idx}_soft_masks_char_{char_idx}_step_{step_idx}.mp4"
                mask_path = action_output_path / mask_filename

                _robust_run(
                    mask_path,
                    write_video_soft_masks,
                    video_norm,
                    mask_path,
                    mask.cpu().numpy(),
                    16,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--mode", choices=["fsdp", "solo"], required=True)
    parser.add_argument("--t5-cpu", action="store_true")
    args = parser.parse_args()

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError as e:
        raise RuntimeError("Must launch via torchrun.") from e

    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    with open(args.task_file, "rb") as f:
        task = pickle.load(f)

    is_fsdp = args.mode == "fsdp"
    # If we are sharding, we only shard T5 if it's NOT on the CPU
    t5_fsdp = is_fsdp and not args.t5_cpu

    checkpoint_dir = task["checkpoint_dir"]
    wan_i2v = WanI2V(
        config=i2v_14B,
        checkpoint_dir=checkpoint_dir,
        device_id=local_rank,
        rank=dist.get_rank(),
        t5_fsdp=t5_fsdp,
        dit_fsdp=is_fsdp,
        use_usp=False,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
    )

    outputs = run_inference(wan_i2v, task)

    if dist.get_rank() == 0:
        assert outputs is not None
        video, extra_data = outputs
        save_outputs(video, extra_data, task)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
