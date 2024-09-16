from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from typing import Literal

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_video

from PIL import Image
from io import BytesIO
import tempfile
import os
import asyncio

app = FastAPI()

# Cache to store loaded pipelines
pipelines = {}


async def get_pipeline(
    generate_type: str,
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Retrieves or loads the pipeline for the specified generation type and model path.
    """
    key = (generate_type, model_path, dtype)
    if key in pipelines:
        return pipelines[key]

    # Load the appropriate pipeline based on the generation type
    if generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    elif generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    elif generate_type == "v2v":
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported generate_type: {generate_type}")

    # Set the scheduler and other configurations
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

    # Cache the pipeline
    pipelines[key] = pipe
    return pipe


@app.post("/generate_video")
async def generate_video_endpoint(
    background_tasks: BackgroundTasks,
    prompt: str = Form(..., description="The description of the video to be generated."),
    model_path: str = Form("THUDM/CogVideoX-5b", description="The path of the pre-trained model to be used."),
    image_or_video: UploadFile = File(
        None, description="The image or video file to be used as input for 'i2v' or 'v2v' generation types."
    ),
    guidance_scale: float = Form(6.0, description="The scale for classifier-free guidance."),
    num_inference_steps: int = Form(50, description="Number of steps for the inference process."),
    num_videos_per_prompt: int = Form(1, description="Number of videos to generate per prompt."),
    generate_type: Literal["t2v", "i2v", "v2v"] = Form(
        "t2v", description="The type of video generation: 't2v', 'i2v', or 'v2v'."
    ),
    dtype: Literal["float16", "bfloat16"] = Form(
        "bfloat16", description="The data type for computation: 'float16' or 'bfloat16'."
    ),
    seed: int = Form(42, description="The seed for reproducibility."),
):
    """
    Endpoint to generate video based on a prompt and other parameters.

    Returns:
    - The generated video file.
    """
    try:
        # Convert dtype string to torch.dtype
        dtype_torch = torch.float16 if dtype == "float16" else torch.bfloat16

        # Asynchronously get or load the pipeline
        pipe = await get_pipeline(generate_type, model_path, dtype_torch)

        image = None
        video = None

        if generate_type == "i2v":
            if image_or_video is None:
                raise HTTPException(status_code=400, detail="An image file must be provided for 'i2v' generation type.")
            # Read the uploaded image
            image_bytes = await image_or_video.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        elif generate_type == "v2v":
            if image_or_video is None:
                raise HTTPException(status_code=400, detail="A video file must be provided for 'v2v' generation type.")
            # Read the uploaded video
            video_bytes = await image_or_video.read()
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video_file:
                tmp_video_file.write(video_bytes)
                tmp_video_file_path = tmp_video_file.name
            video = load_video(tmp_video_file_path)
            # Schedule the temporary file to be deleted after processing
            background_tasks.add_task(os.remove, tmp_video_file_path)

        # Generate the video frames based on the prompt.
        generator = torch.Generator().manual_seed(seed)
        if generate_type == "i2v":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipe(
                    prompt=prompt,
                    image=image,
                    num_videos_per_prompt=num_videos_per_prompt,
                    num_inference_steps=num_inference_steps,
                    num_frames=49,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ),
            )
            video_generate = result.frames[0]
        elif generate_type == "t2v":
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipe(
                    prompt=prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    num_inference_steps=num_inference_steps,
                    num_frames=49,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ),
            )
            video_generate = result.frames[0]
        else:  # v2v
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pipe(
                    prompt=prompt,
                    video=video,
                    num_videos_per_prompt=num_videos_per_prompt,
                    num_inference_steps=num_inference_steps,
                    num_frames=49,
                    use_dynamic_cfg=True,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ),
            )
            video_generate = result.frames[0]

        # Export the generated frames to a video file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output_file:
            output_path = tmp_output_file.name
            export_to_video(video_generate, output_path, fps=8)

        # Schedule the output file to be deleted after sending response
        background_tasks.add_task(os.remove, output_path)

        # Return the video file
        return FileResponse(output_path, media_type="video/mp4", filename="output.mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
