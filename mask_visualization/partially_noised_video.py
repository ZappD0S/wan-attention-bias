# %%
import torch
from diffusers.utils import load_video
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from wan.configs.wan_i2v_14B import i2v_14B
from wan.regional_prompt import WanI2V

DEVICE = "cuda"
DTYPE = torch.bfloat16
WIDTH, HEIGHT = 832, 480


# %%
wan_i2v = WanI2V(
    config=i2v_14B,
    checkpoint_dir="../weights/Wan2.1-I2V-14B-480P/",
    device_id=0,
    t5_cpu=True,
)

# %%
scheduler, _ = wan_i2v._get_scheduler("unipc", sampling_steps=40, shift=5.0)
num_train_timesteps = wan_i2v.num_train_timesteps

VIDEO_PATH = "./video.mp4"
raw_frames = load_video(VIDEO_PATH)

# Wan2.1 frame alignment (4n + 1 Rule)
# We crop the list to ensure the VAE is happy
target_len = ((len(raw_frames) - 1) // 4) * 4 + 1
raw_frames = raw_frames[:target_len]

# 3. Process into Tensor
processor = VideoProcessor(do_resize=True, do_normalize=True)

# preprocess_video returns [Batch, Frames, Channels, Height, Width]
video_tensor = processor.preprocess_video(video=raw_frames, height=HEIGHT, width=WIDTH).to(
    DEVICE, dtype=DTYPE
)

video_tensor = rearrange(video_tensor, "B T C H W -> B C T H W")

# align frames for Wan (4n+1 rule)
num_frames = len(raw_frames)
target_num_frames = ((num_frames - 1) // 4) * 4 + 1
raw_frames = raw_frames[:target_num_frames]

# process into [B, C, F, H, W] tensor
# preprocess() returns a tensor in [B, C, F, H, W] normalized to [-1, 1]
video_tensor = processor.preprocess_video(raw_frames).to(DEVICE, dtype=DTYPE)


vae = wan_i2v.vae
with torch.no_grad():
    latents = vae.encode(video_tensor)

# encode() returns a list
latents = torch.stack(latents)

noise = torch.randn_like(latents)


target_t = 0.3 * num_train_timesteps

timesteps_tensor = scheduler.timesteps.to(DEVICE)
diffs = torch.abs(timesteps_tensor - target_t)
closest_index = torch.argmin(diffs)
actual_timestep = timesteps_tensor[closest_index].unsqueeze(0)

noisy_latents = scheduler.add_noise(
    original_samples=latents, noise=noise, timesteps=actual_timestep
)

normalized_t = actual_timestep.item() / num_train_timesteps

# TODO: we should save the image used for generating the initial video (let's say the first frame of the video) and the actual_timestep
torch.save(noisy_latents, f"video_latents_{normalized_t:.2f}_noise.pt")

# %%
