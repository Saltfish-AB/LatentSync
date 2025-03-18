#!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/stage2.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 20 \
    --guidance_scale 1.5 \
    --video_path "assets/arvid2_cf.mp4" \
    --audio_path "assets/ElevenLabs_2025-03-18T15_12_11_qura_ivc_s100_sb75_se0_b_m2.mp3" \
    --video_out_path "video_out.mp4"
