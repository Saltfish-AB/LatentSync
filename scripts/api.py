from typing import Optional
from fastapi import FastAPI, HTTPException
from latentsync.utils.darken_restore import calculate_inverse_factor, enhance_face_brightness
from latentsync.utils.thumbnail import create_video_thumbnail_gif
from pydantic import BaseModel
import asyncio
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.utils.gcs import upload_video_to_gcs
from latentsync.utils.download import cleanup_folder, download_file
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
import time
import os
import uuid

app = FastAPI()

# Define a queue to hold incoming requests
request_queue = asyncio.Queue(maxsize=10)  # Adjust `maxsize` as needed

# Semaphore to ensure only one request is processed at a time
processing_semaphore = asyncio.Semaphore(1)


class RequestPayload(BaseModel):
    id: str
    video_id: str
    audio_url: str
    start_from_backwards: Optional[bool] = None
    force_video_length: Optional[bool] = None
    is_dynamic_clip: Optional[bool] = None
    text: Optional[str] = None
    use_darken: Optional[bool] = None
    brightness_factor: Optional[float] = 1


@app.on_event("startup")
async def startup_event():
    """
    Initialize shared variables and start the background worker.
    """
    config = OmegaConf.load("configs/unet/stage2.yaml")
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32
    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")
    
    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    denoising_unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        "checkpoints/latentsync_unet.pt",
        device="cpu",
    )

    denoising_unet = denoising_unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        denoising_unet=denoising_unet,
        scheduler=scheduler,
    ).to("cuda")

    torch.seed()
    print(f"Initial seed: {torch.initial_seed()}")

    app.state.shared_variable = {"pipeline": pipeline, "config": config, "dtype": dtype}  # Example shared variable
    asyncio.create_task(process_requests())
    print("Startup: Shared variable initialized and background task started.")


async def process_requests():
    """
    Background task to handle queued requests.
    """
    while True:
        payload, task = await request_queue.get()
        try:
            async with processing_semaphore:
                start_time = time.time()
                id = payload["id"]
                video_id = payload["video_id"]
                audio_url = payload["audio_url"]
                start_from_backwards = payload["start_from_backwards"] or False
                force_video_length = payload["force_video_length"] or False
                is_dynamic_clip = payload.get("is_dynamic_clip", False)
                use_darken = payload.get("use_darken", False)
                brightness_factor = payload.get("brightness_factor", 1)
                text = payload.get("text", None)
                print("payload", payload)

                video_path = "/latent-sync-data/{}.mp4".format(video_id)
                data_path = "/latent-sync-data/{}.pth".format(video_id)
                audio_path = "/latent-sync-data/{}.wav".format(id)

                if is_dynamic_clip and os.path.exists("/latent-sync-data/{}_rotated.pth".format(video_id))  and os.path.exists("/latent-sync-data/{}_rotated.mp4".format(video_id)):
                    data_path = "/latent-sync-data/{}_rotated.pth".format(video_id)
                    video_path = "/latent-sync-data/{}_rotated.mp4".format(video_id)
                    if use_darken:
                        data_path = "/latent-sync-data/{}_darken_rotated.pth".format(video_id)
                        video_path = "/latent-sync-data/{}_darken_rotated.mp4".format(video_id)
                elif use_darken:
                    data_path = "/latent-sync-data/{}_darken.pth".format(video_id)
                    video_path = "/latent-sync-data/{}_darken.mp4".format(video_id)

                if not os.path.exists(video_path):
                    raise HTTPException(status_code=400, detail="Video file not found.")
                if not os.path.exists(data_path):
                    raise HTTPException(status_code=400, detail="Data file not found.")
                if not os.path.exists(audio_path):
                    download_file(audio_url, audio_path)

                print("data_path", data_path)
                print("video_path", video_path)

                video_out_path = "results/{}.mp4".format(id)
                config = app.state.shared_variable["config"]
                dtype = app.state.shared_variable["dtype"]

                calculated_factor = calculate_inverse_factor(brightness_factor)

                app.state.shared_variable["pipeline"](
                    video_path=video_path,
                    audio_path=audio_path,
                    video_out_path=video_out_path,
                    video_mask_path=video_out_path.replace(".mp4", "_mask.mp4"),
                    num_frames=16,
                    num_inference_steps=20,
                    guidance_scale=1.5,
                    weight_dtype=dtype,
                    width=config.data.resolution,
                    height=config.data.resolution,
                    data_path=data_path,
                    start_from_backwards=start_from_backwards,
                    force_video_length=force_video_length,
                    use_darken=use_darken,
                    brightness_factor=calculated_factor,
                )

                output_id = uuid.uuid4()
                gcs_path = "videos/{}.mp4".format(output_id)
                upload_video_to_gcs(
                    bucket_name="saltfish-public",
                    source_file_path=f"{video_out_path}",
                    destination_blob_name=gcs_path
                )

                gif_id = None
                if is_dynamic_clip and text:
                    gif_output = "results/thumbnail.gif"
                    gif_id = uuid.uuid4()
                    create_video_thumbnail_gif(
                        video_path=video_out_path,
                        output_path=gif_output,
                        duration=6,
                        fps=3,
                        subtitle_text=text
                    )
                    upload_video_to_gcs(
                        bucket_name="saltfish-public",
                        source_file_path=gif_output,
                        destination_blob_name="gifs/{}.gif".format(gif_id)
                    )

                end_time = time.time()
                elapsed_time = end_time - start_time

                cleanup_folder("./results")
                
                task.set_result({
                    "message": "Request processed successfully",
                    "output_url": "https://storage.saltfish.ai/{}".format(gcs_path),
                    "gif_url": "https://storage.saltfish.ai/gifs/{}.gif".format(gif_id) if gif_id else None,
                    "elapsed_time": elapsed_time
                })
        except Exception as e:
            task.set_exception(e)
        finally:
            request_queue.task_done()


@app.post("/process")
async def process(payload: RequestPayload):
    """
    POST endpoint to queue requests. Ensures only one request is processed at a time.
    """
    if request_queue.full():
        return {"error": "Queue is full, try again later."}

    # Use a Future to hold the result of the request
    task = asyncio.get_event_loop().create_future()
    await request_queue.put((payload.dict(), task))  # Add the payload to the queue

    # Wait for the task to be completed
    result = await task
    return result

@app.get("/ping")
async def ping():
    """
    A simple ping endpoint to check if the server is running.
    """
    return {"message": "pong"}