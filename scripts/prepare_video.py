import argparse
from omegaconf import OmegaConf
from latentsync.pipelines.affine_transform_video import generate_affine_transforms
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/second_stage.yaml")
    parser.add_argument("--video_path", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    generate_affine_transforms(args.video_path, height=config.data.resolution)
