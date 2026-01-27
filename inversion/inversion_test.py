import gc

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video, load_image, load_video
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

# ==========================================
# 1. EMBEDDING HELPERS
# ==========================================


def build_wan_pipeline(model_path, shift=3.0, dtype=torch.bfloat16, device="cuda"):
    """
    Loads the Transformer and VAE, detects mode, and sets up the scheduler.
    """
    print(f"--- Building Wan 2.1 Pipeline (Shift: {shift}) ---")

    transformer = WanTransformer3DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype
    ).to(device)  # ty:ignore[invalid-argument-type]

    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(
        device  # ty:ignore[invalid-argument-type]
    )

    # Auto-detect I2V based on transformer input channels
    is_i2v = transformer.config.in_channels == 36
    PipelineClass = WanImageToVideoPipeline if is_i2v else WanPipeline

    pipe = PipelineClass.from_pretrained(
        model_path,
        transformer=transformer,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        image_encoder=None,
        torch_dtype=dtype,
    )

    # Configure Scheduler with the specific Shift
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config, num_train_timesteps=1000, shift=shift
    )

    # Optimization for VAE (Slicing helps with memory)
    pipe.vae.disable_tiling()
    pipe.vae.enable_slicing()

    return pipe, is_i2v


# ==========================================
# 2. RAM-OPTIMIZED EMBEDDING HELPERS
# ==========================================


@torch.no_grad()
def get_prompt_embeddings(model_path, prompt, dtype=torch.bfloat16, device="cuda"):
    print(f"--- Encoding Prompt: '{prompt}' ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", load_in_4bit=True, device_map="cpu"
    )

    temp_pipe = WanPipeline.from_pretrained(
        model_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=None,
        vae=None,
        torch_dtype=dtype,
    )

    prompt_embeds, _ = temp_pipe.encode_prompt(
        prompt=prompt,
        device=torch.device("cpu"),
        num_videos_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    prompt_embeds = prompt_embeds.to(device, dtype=dtype)
    del text_encoder, tokenizer, temp_pipe
    gc.collect()
    torch.cuda.empty_cache()
    return prompt_embeds


@torch.no_grad()
def get_image_context_embeddings(model_path, image, dtype=torch.bfloat16, device="cuda"):
    print("--- Encoding Image Context (CLIP) ---")
    image_processor = CLIPImageProcessor.from_pretrained(model_path, subfolder="image_processor")
    image_encoder = CLIPVisionModel.from_pretrained(
        model_path, subfolder="image_encoder", torch_dtype=dtype
    ).to(device)

    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(
        device=device, dtype=dtype
    )
    image_embeds = image_encoder(pixel_values).last_hidden_state

    del image_encoder, image_processor
    gc.collect()
    torch.cuda.empty_cache()
    return image_embeds


# ==========================================
# 3. VAE & LATENT HELPERS
# ==========================================


def get_vae_norm_params(pipeline):
    mean = torch.tensor(pipeline.vae.config.latents_mean, device="cuda", dtype=pipeline.dtype).view(
        1, -1, 1, 1, 1
    )
    std = torch.tensor(pipeline.vae.config.latents_std, device="cuda", dtype=pipeline.dtype).view(
        1, -1, 1, 1, 1
    )
    return mean, std


@torch.no_grad()
def encode_video(pipeline, video_path, num_frames=81):
    print("--- Encoding Video to Latents ---")
    video = load_video(video_path)[:num_frames]
    video_tensor = pipeline.video_processor.preprocess(video)
    if video_tensor.ndim == 4:
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
    video_tensor = video_tensor.to(device="cuda", dtype=pipeline.dtype)

    latents = pipeline.vae.encode(video_tensor).latent_dist.sample()
    mean, std = get_vae_norm_params(pipeline)
    latents = (latents - mean) / std
    return latents


@torch.no_grad()
def get_i2v_conditioning(pipeline, image, num_frames, latent_shape):
    print("--- Creating I2V Concatenated Conditioning (Zero-Padded) ---")
    device, dtype = pipeline.device, pipeline.dtype

    img_tensor = pipeline.video_processor.preprocess([image]).to(device, dtype=dtype)
    if img_tensor.ndim == 4:
        img_tensor = img_tensor.unsqueeze(2)

    img_latents = pipeline.vae.encode(img_tensor).latent_dist.sample()
    mean, std = get_vae_norm_params(pipeline)
    img_latents = (img_latents - mean) / std

    cond_latents = torch.zeros(
        1, 16, num_frames, latent_shape[3], latent_shape[4], device=device, dtype=dtype
    )
    cond_latents[:, :, :1, :, :] = img_latents

    mask = torch.zeros(
        1, 4, num_frames, latent_shape[3], latent_shape[4], device=device, dtype=dtype
    )
    mask[:, :, :1, :, :] = 1.0

    return cond_latents, mask


# ==========================================
# 4. CORE ODE SOLVER (Unified)
# ==========================================


@torch.no_grad()
def run_ode(
    pipeline,
    z,
    prompt_embeds,
    clip_context=None,
    cond_latents=None,
    mask=None,
    num_steps=50,
    reverse=False,
):
    pipeline.scheduler.set_timesteps(num_steps, device="cuda")
    timesteps = pipeline.scheduler.timesteps
    if reverse:
        timesteps = timesteps.flip(0)

    is_i2v_input = pipeline.transformer.config.in_channels == 36

    for i, t in tqdm(enumerate(timesteps[:-1]), total=len(timesteps) - 1, desc="ODE Step"):
        t_batch = t.expand(z.shape[0])

        if is_i2v_input:
            if cond_latents is not None and mask is not None:
                model_input = torch.cat([z, cond_latents, mask], dim=1)
            else:
                raise ValueError(
                    "The loaded Wan model is I2V, but 'cond_latents' or 'mask' was not provided."
                )
        else:
            model_input = z

        noise_pred = pipeline.transformer(
            hidden_states=model_input,
            timestep=t_batch,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_image=clip_context,
            return_dict=False,
        )[0]

        dt = (timesteps[i + 1] - t) / pipeline.scheduler.config.num_train_timesteps
        z = z + noise_pred * dt

    return z


@torch.no_grad()
def decode_latents_to_video(pipeline, latents, output_filename, fps=15):
    print(f"--- Decoding Latents to {output_filename} ---")
    pipeline.vae.enable_tiling()
    mean, std = get_vae_norm_params(pipeline)
    latents = (latents * std) + mean

    video_out = pipeline.vae.decode(latents.to(pipeline.device), return_dict=False)[0]
    frames = pipeline.video_processor.postprocess_video(video_out, output_type="pil")[0]
    export_to_video(frames, output_filename, fps=fps)


# ==========================================
# 5. EXECUTION
# ==========================================

# MODEL_PATH = "../weights/Wan2.1-I2V-14B-480P-Diffusers/"
MODEL_PATH = "../weights/Wan2.1-T2V-14B-Diffusers/"
VIDEO_IN = "./video.mp4"
PROMPT = "A video of two golden retrievers."
FIRST_FRAME = "../image_prompt_generation/images/0.png"
SHIFT = 3.0

# 1. Build
pipe, is_i2v = build_wan_pipeline(MODEL_PATH)
mode_suffix = "i2v" if is_i2v else "t2v"

# 3. Get Embeddings & Conditionals
prompt_embeds = get_prompt_embeddings(MODEL_PATH, PROMPT)
orig_latents = encode_video(pipe, VIDEO_IN)

clip_context = None
cond_latents = None
mask = None

if is_i2v:
    print(f"Loading separate conditioning image: {FIRST_FRAME}")
    input_image = load_image(FIRST_FRAME)

    # Resize image to match video dimensions (Latent Dim * 8)
    target_h = orig_latents.shape[3] * 8
    target_w = orig_latents.shape[4] * 8
    input_image = input_image.resize((target_w, target_h))

    # CLIP Attention context
    clip_context = get_image_context_embeddings(MODEL_PATH, input_image)

    # VAE Concatenation latents and mask
    cond_latents, mask = get_i2v_conditioning(
        pipe, input_image, num_frames=orig_latents.shape[2], latent_shape=orig_latents.shape
    )

inverted_noise = run_ode(
    pipe, orig_latents, prompt_embeds, clip_context, cond_latents, mask, reverse=True
)

recon_latents = run_ode(
    pipe, inverted_noise, prompt_embeds, clip_context, cond_latents, mask, reverse=True
)

# 6. Memory Cleanup
# Essential to offload the 14B transformer before the VAE decoder kicks in
print("--- Offloading Transformer to free VRAM for VAE ---")
pipe.transformer.to("cpu")
del pipe.transformer
gc.collect()
torch.cuda.empty_cache()

# 7. Decode and Save with Mode-Specific Name
output_filename = f"reconstruction_result_{mode_suffix}.mp4"
decode_latents_to_video(pipe, recon_latents, output_filename)

print(f"Process complete. Output saved to {output_filename}")
