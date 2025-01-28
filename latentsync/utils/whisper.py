import subprocess
import json
import os
import ffmpeg

def process_video_with_whisper(video_path: str, output_json_path: str, trimmed_video_path: str):
    """
    Process a video using Whisper to extract word-level timestamps and trim the video 
    one second after the last word.

    Args:
        video_path (str): Path to the input video file.
        output_json_path (str): Path to save the Whisper output JSON.
        trimmed_video_path (str): Path to save the trimmed video.

    Returns:
        dict: JSON data containing transcription and timestamps.
    """
    try:
        # Step 1: Run Whisper to generate JSON output
        whisper_cmd = [
            "whisper",
            video_path,
            "--model", "turbo",
            "--output_format", "json",
            "--output_dir", os.path.dirname(output_json_path)
        ]
        print(f"Running Whisper with command: {' '.join(whisper_cmd)}")
        subprocess.run(whisper_cmd, check=True)

        # Step 2: Load the JSON output
        with open(output_json_path, "r") as f:
            whisper_data = json.load(f)

        # Step 3: Get the last word's end timestamp
        last_word_end = 0
        for segment in whisper_data.get("segments", []):
            for word in segment.get("words", []):
                last_word_end = max(last_word_end, word.get("end", 0))

        # Add 1 second buffer to the end timestamp
        trim_end = last_word_end + 0.5

        # Step 4: Trim the video
        print(f"Trimming video to {trim_end} seconds")
        print(trim_end)
        ffmpeg.input(video_path, ss=0).output(trimmed_video_path, t=trim_end).run(overwrite_output=True)

        print(f"Trimmed video saved to {trimmed_video_path}")

        # Return the JSON data for further inspection
        return whisper_data

    except subprocess.CalledProcessError as e:
        print(f"Error running Whisper: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise