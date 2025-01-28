import subprocess

def add_silence_to_audio(input_audio: str, output_audio: str, silence_duration: float = 0.5):
    """
    Adds silence to the end of an audio file using FFmpeg.

    Args:
        input_audio (str): Path to the input audio file.
        output_audio (str): Path to save the output audio file with added silence.
        silence_duration (float): Duration of the silence to add in seconds (default is 0.5s).

    Raises:
        Exception: If FFmpeg encounters an error.
    """
    try:
        # FFmpeg command to add silence
        command = [
            "ffmpeg",
            "-i", input_audio,  # Input audio file
            "-f", "lavfi",      # Generate silence
            "-t", str(silence_duration),
            "-i", "anullsrc=r=44100:cl=stereo",
            "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
            "-map", "[out]",
            "-y", output_audio  # Overwrite the output file if it exists
        ]
        
        # Run the FFmpeg command
        subprocess.run(command, check=True)
        print(f"Silence added successfully. Saved as: {output_audio}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
