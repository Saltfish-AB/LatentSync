import argparse
import sys
from urllib.parse import urlparse
from omegaconf import OmegaConf
from latentsync.utils.persist_data import save_on_persistent_disk
from latentsync.utils.gcs import upload_video_to_gcs
from latentsync.utils.download import cleanup_file, download_file
from latentsync.pipelines.affine_transform_video import generate_affine_transforms


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--unet_config_path", type=str, default="configs/unet/second_stage.yaml")
        parser.add_argument("--video_id", type=str, required=True)
        parser.add_argument("--video_url", type=str, required=True)
        args = parser.parse_args()

        video_id = args.video_id
        config = OmegaConf.load(args.unet_config_path)

        filename = urlparse(args.video_url).path.split('/')[-1]
        video_path = f"assets/{filename}"

        # Download the video
        download_file(args.video_url, video_path)

        output_data_filename = f"{video_id}.pth"
        output_data_path = f"results/{output_data_filename}"

        # Generate affine transforms
        generate_affine_transforms(video_path, output_data_path, height=config.data.resolution)

        # Upload results
        upload_video_to_gcs("saltfish-public", output_data_path, f"latentsync_data/{output_data_filename}")
        save_on_persistent_disk(output_data_path)
        save_on_persistent_disk(video_path, f"{video_id}.mp4")

        # Cleanup
        cleanup_file(video_path)

        print("✅ Process completed successfully.")
        sys.exit(0)  # Return 0 on success

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)  # Return 1 on failure


if __name__ == "__main__":
    main()
