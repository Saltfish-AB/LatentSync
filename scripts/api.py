from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.utils.gcs import upload_video_to_gcs
from latentsync.utils.download import download_file
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.utils.mouth_enhancer import MouthEnhancer
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


@app.on_event("startup")
async def startup_event():
    """
    Initialize shared variables and start the background worker.
    """
    config = OmegaConf.load("configs/unet/second_stage.yaml")
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

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        "checkpoints/latentsync_unet.pt",  # load checkpoint
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    # set xformers
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    mouth_enhancer = MouthEnhancer(debug_mode=True)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
        mouth_enhancer=mouth_enhancer,
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

                video_path = "/latent-sync-data/{}.mp4".format(video_id)
                data_path = "/latent-sync-data/{}.pth".format(video_id)
                audio_path = "/latent-sync-data/{}.wav".format(id)
                if not os.path.exists(video_path):
                    raise HTTPException(status_code=400, detail="Video file not found.")
                if not os.path.exists(data_path):
                    raise HTTPException(status_code=400, detail="Data file not found.")
                if not os.path.exists(audio_path):
                    download_file(audio_url, audio_path)

                video_out_path = "results/{}.mp4".format(id)
                config = app.state.shared_variable["config"]
                dtype = app.state.shared_variable["dtype"]
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
                    force_video_length=force_video_length
                )

                output_id = uuid.uuid4()
                gcs_path = "videos/{}.mp4".format(output_id)
                upload_video_to_gcs(
                    bucket_name="saltfish-public",
                    source_file_path=f"{video_out_path}",
                    destination_blob_name=gcs_path
                )
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                task.set_result({
                    "message": "Request processed successfully",
                    "output_url": "https://storage.saltfish.ai/{}".format(gcs_path),
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