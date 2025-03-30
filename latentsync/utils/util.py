# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import json
from typing import Optional, Union
import matplotlib.pyplot as plt

from latentsync.utils.darken_restore import enhance_face_brightness
import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
from torchvision import transforms

from einops import rearrange
import cv2
from decord import AudioReader, VideoReader
import shutil
import subprocess
from pathlib import Path


# Machine epsilon for a float32 (single precision)
eps = np.finfo(np.float32).eps


def read_json(filepath: str):
    with open(filepath) as f:
        json_dict = json.load(f)
    return json_dict


def read_video(video_path: str, change_fps=True, use_decord=True):
    if change_fps:
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        command = (
            f"ffmpeg -loglevel error -y -nostdin -i {video_path} -r 25 -crf 18 {os.path.join(temp_dir, 'video.mp4')}"
        )
        subprocess.run(command, shell=True)
        target_video_path = os.path.join(temp_dir, "video.mp4")
    else:
        target_video_path = video_path

    if use_decord:
        return read_video_decord(target_video_path)
    else:
        return read_video_cv2(target_video_path)


def read_video_decord(video_path: str):
    vr = VideoReader(video_path)
    video_frames = vr[:].asnumpy()
    vr.seek(0)
    return video_frames


def read_video_cv2(video_path: str):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return np.array([])

    frames = []

    while True:
        # Read a frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame_rgb)

    # Release the video capture object
    cap.release()

    return np.array(frames)


def read_audio(audio_path: str, audio_sample_rate: int = 16000):
    if audio_path is None:
        raise ValueError("Audio path is required.")
    ar = AudioReader(audio_path, sample_rate=audio_sample_rate, mono=True)

    # To access the audio samples
    audio_samples = torch.from_numpy(ar[:].asnumpy())
    audio_samples = audio_samples.squeeze(0)

    return audio_samples


def write_video(
        video_output_path: str,
        video_frames: np.ndarray,
        fps: int, is_high_quality: bool = True,
        use_darken: Optional[bool] = False,
        brightness_factor: Optional[float] = 1.0):
    """
    Write video frames to a high-quality MP4 file using FFmpeg.
    
    Args:
        video_output_path: Path to save the video
        video_frames: Numpy array of frames with shape (n_frames, height, width, channels)
        fps: Frames per second
        is_high_quality: If True, use settings for maximum visual quality (larger file size)
    
    Returns:
        Path to the output video file
    """
    # Ensure file has .mp4 extension
    if not video_output_path.lower().endswith(".mp4"):
        video_output_path = video_output_path.rsplit(".", 1)[0] + ".mp4"
    
    # Create a temporary directory for frames
    temp_dir = Path(os.path.dirname(video_output_path)) / "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save frames as PNG (lossless)
    for i, frame in enumerate(video_frames):
        # OpenCV expects BGR, but our frames might be RGB
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Use maximum PNG quality
        cv2.imwrite(str(temp_dir / f"frame_{i:05d}.png"), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    if use_darken and brightness_factor:
        enhance_face_brightness(temp_dir, 0, len(video_frames), brightness_factor=brightness_factor)

    # FFmpeg quality settings
    if is_high_quality:
        # Ultra high quality settings
        crf = "10"        # Very low CRF for near-lossless quality (0-51, lower is better)
        preset = "veryslow"  # Slowest preset = best compression efficiency
        additional_args = [
            "-tune", "film",  # Tune for high-detail content
            "-color_primaries", "bt709",  # Ensure proper color space
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-movflags", "+faststart",   # For web playback optimization
        ]
    else:
        # Default high quality settings
        crf = "17"
        preset = "slow"
        additional_args = []
    
    # Base FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-r", str(fps),  # Frame rate
        "-i", str(temp_dir / "frame_%05d.png"),  # Input pattern
        "-c:v", "libx264",  # H.264 codec
    ]
    
    # Add quality settings
    cmd.extend(["-crf", crf, "-preset", preset])
    
    # Add any additional arguments
    cmd.extend(additional_args)
    
    # Common settings
    cmd.extend([
        "-pix_fmt", "yuv420p",  # Pixel format for compatibility
        video_output_path
    ])

    cmd.extend([
        "-c:v", "libx265",
        "-crf", "15",
        "-preset", "veryslow",
        "-x265-params", "lossless=1"  # For truly lossless output
    ])
    
    # Run FFmpeg
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg failed: {e.stderr.decode()}")
    finally:
        # Clean up temporary files
        for file in temp_dir.glob("*.png"):
            os.remove(file)
        os.rmdir(temp_dir)
    
    return video_output_path


def init_dist(backend="nccl", **kwargs):
    """Initializes distributed environment."""
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for training.")
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)

    return local_rank


def zero_rank_print(s):
    if dist.is_initialized() and dist.get_rank() == 0:
        print("### " + s)


def zero_rank_log(logger, message: str):
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info(message)


def check_video_fps(video_path: str):
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        raise ValueError(f"Video FPS is not 25, it is {fps}. Please convert the video to 25 FPS.")


def one_step_sampling(ddim_scheduler, pred_noise, timesteps, x_t):
    # Compute alphas, betas
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timesteps].to(dtype=pred_noise.dtype)
    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/abs/2010.02502
    if ddim_scheduler.config.prediction_type == "epsilon":
        beta_prod_t = beta_prod_t[:, None, None, None, None]
        alpha_prod_t = alpha_prod_t[:, None, None, None, None]
        pred_original_sample = (x_t - beta_prod_t ** (0.5) * pred_noise) / alpha_prod_t ** (0.5)
    else:
        raise NotImplementedError("This prediction type is not implemented yet")

    # Clip "predicted x_0"
    if ddim_scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
    return pred_original_sample


def plot_loss_chart(save_path: str, *args):
    # Creating the plot
    plt.figure()
    for loss_line in args:
        plt.plot(loss_line[1], loss_line[2], label=loss_line[0])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()

    # Save the figure to a file
    plt.savefig(save_path)

    # Close the figure to free memory
    plt.close()


CRED = "\033[91m"
CEND = "\033[0m"


def red_text(text: str):
    return f"{CRED}{text}{CEND}"


log_loss = nn.BCELoss(reduction="none")


def cosine_loss(vision_embeds, audio_embeds, y):
    sims = nn.functional.cosine_similarity(vision_embeds, audio_embeds)
    # sims[sims!=sims] = 0 # remove nan
    # sims = sims.clamp(0, 1)
    loss = log_loss(sims.unsqueeze(1), y).squeeze()
    return loss


def save_image(image, save_path):
    # input size (C, H, W)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255).to(torch.uint8)
    image = transforms.ToPILImage()(image)
    # Save the image copy
    image.save(save_path)

    # Close the image file
    image.close()


def gather_loss(loss, device):
    # Sum the local loss across all processes
    local_loss = loss.item()
    global_loss = torch.tensor(local_loss, dtype=torch.float32).to(device)
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)

    # Calculate the average loss across all processes
    global_average_loss = global_loss.item() / dist.get_world_size()
    return global_average_loss


def gather_video_paths_recursively(input_dir):
    print(f"Recursively gathering video paths of {input_dir} ...")
    paths = []
    gather_video_paths(input_dir, paths)
    return paths


def gather_video_paths(input_dir, paths):
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".mp4"):
            filepath = os.path.join(input_dir, file)
            paths.append(filepath)
        elif os.path.isdir(os.path.join(input_dir, file)):
            gather_video_paths(os.path.join(input_dir, file), paths)


def count_video_time(video_path):
    video = cv2.VideoCapture(video_path)

    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    return frame_count / fps


def check_ffmpeg_installed():
    # Run the ffmpeg command with the -version argument to check if it's installed
    result = subprocess.run("ffmpeg -version", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    if not result.returncode == 0:
        raise FileNotFoundError("ffmpeg not found, please install it by:\n    $ conda install -c conda-forge ffmpeg")
