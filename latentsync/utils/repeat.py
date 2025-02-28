import torch
import numpy as np

def repeat_to_length(array, target_length):
    """Repeats an array (torch.Tensor, list, or np.ndarray) to match the target length."""
    current_length = len(array)

    print(f"Original length: {current_length}, Target length: {target_length}")

    if current_length >= target_length:
        print("No repetition needed, returning original array.")
        return array[:target_length]  # Truncate if already long enough

    repeat_factor = -(-target_length // current_length)  # Ceiling division
    print(f"Repeating with factor: {repeat_factor}")

    if isinstance(array, torch.Tensor):
        new_array = array.repeat((repeat_factor, *[1] * (array.dim() - 1)))[:target_length]
    elif isinstance(array, np.ndarray):
        new_array = np.tile(array, (repeat_factor, *[1] * (array.ndim - 1)))[:target_length]
    elif isinstance(array, list):
        new_array = (array * repeat_factor)[:target_length]
    else:
        raise TypeError("Unsupported type for repetition")

    print(f"New length: {len(new_array)}")
    return new_array


def truncate_to_length(array, target_length):
    """Truncates an array (torch.Tensor, list, or np.ndarray) from the beginning to match the target length."""
    current_length = len(array)
    
    print(f"Original length: {current_length}, Target length: {target_length}")
    
    if current_length <= target_length:
        print("No truncation needed, returning original array.")
        return array  # Return as is if already within target length
    
    start_index = current_length - target_length  # Calculate the starting index
    print(f"Truncating from index: {start_index}")
    
    if isinstance(array, torch.Tensor):
        new_array = array[start_index:]
    elif isinstance(array, np.ndarray):
        new_array = array[start_index:]
    elif isinstance(array, list):
        new_array = array[start_index:]
    else:
        raise TypeError("Unsupported type for truncation")
    
    print(f"New length: {len(new_array)}")
    return new_array

def prepend_zero_tensors(whisper_chunks, num_prepend, tensor_shape):
    """
    Prepends a specified number of zero tensors to the beginning of whisper_chunks.

    Args:
        whisper_chunks (list of torch.Tensor): The original list of tensors.
        num_prepend (int): The number of zero tensors to prepend.
        tensor_shape (tuple): The shape of the zero tensors to create.

    Returns:
        list of torch.Tensor: The modified list with zero tensors prepended.
    """
    if num_prepend <= 0:
        return whisper_chunks  # No change if num_prepend is 0 or negative

    # Create the zero tensors with the given shape
    zero_tensors = [torch.zeros(tensor_shape) for _ in range(num_prepend)]

    # Prepend them to the whisper_chunks list
    return zero_tensors + whisper_chunks

import torch

def pad_whisper_chunks(whisper_chunks, tensor_shape, audio_samples, audio_sample_rate, fps=25):
    """
    Pads whisper_chunks with zero tensors to make its length divisible by 16.
    Also pads audio_samples with zeros to align with whisper_chunks in terms of time.

    Args:
        whisper_chunks (list of torch.Tensor): The original list of tensors (video frames at `fps`).
        tensor_shape (tuple): The shape of the zero tensors to create.
        audio_samples (torch.Tensor): The original audio samples.
        audio_sample_rate (int): The sample rate of the audio.
        fps (int): The frames per second of whisper_chunks (default: 25).

    Returns:
        tuple: Modified whisper_chunks, padded audio_samples, padding duration (sec).
    """
    # Pad whisper_chunks
    current_length = len(whisper_chunks)
    num_to_add = (16 - (current_length % 16)) % 16  # Calculate how many to add
    
    padding_duration = num_to_add / fps  # Time added due to chunk padding

    if num_to_add > 0:
        zero_tensors = [torch.zeros(tensor_shape) for _ in range(num_to_add)]
        whisper_chunks = zero_tensors + whisper_chunks  # Prepend zero tensors
    
    # Compute expected total audio length (using updated whisper_chunks duration)
    total_duration = len(whisper_chunks) / fps  # New total duration in seconds
    expected_audio_length = int(total_duration * audio_sample_rate)  # Expected audio samples
    
    # Pad audio_samples to match the new duration
    audio_length = audio_samples.shape[0]
    pad_amount = int(padding_duration * audio_sample_rate)  # Ensure same padding duration
    
    if pad_amount > 0:
        zero_padding = torch.zeros(pad_amount, dtype=audio_samples.dtype)
        audio_samples = torch.cat([zero_padding, audio_samples], dim=0)

    return whisper_chunks, audio_samples, padding_duration

def pad_whisper_chunks_end(whisper_chunks, tensor_shape, audio_samples, audio_sample_rate, fps=25, divisible_by=16):
    """
    Pads whisper_chunks with zero tensors to make its length divisible by a specified number.
    Also pads audio_samples with zeros to align with whisper_chunks in terms of time.

    This version pads at the end instead of the beginning.

    Args:
        whisper_chunks (list of torch.Tensor): The original list of tensors (video frames at `fps`).
        tensor_shape (tuple): The shape of the zero tensors to create.
        audio_samples (torch.Tensor): The original audio samples.
        audio_sample_rate (int): The sample rate of the audio.
        fps (int): The frames per second of whisper_chunks (default: 25).
        divisible_by (int): Make the number of chunks divisible by this value (default: 16).

    Returns:
        tuple: Modified whisper_chunks, padded audio_samples, padding duration (sec).
    """
    # Create a fresh copy of whisper_chunks to ensure we don't have reference issues
    whisper_chunks_padded = whisper_chunks.copy()
    
    # Check if input is already a list to avoid errors
    if not isinstance(whisper_chunks_padded, list):
        whisper_chunks_padded = list(whisper_chunks_padded)
    
    # Pad whisper_chunks
    current_length = len(whisper_chunks_padded)
    num_to_add = (divisible_by - (current_length % divisible_by)) % divisible_by  # Calculate how many to add
    
    padding_duration = num_to_add / fps  # Time added due to chunk padding

    if num_to_add > 0:
        zero_tensors = [torch.zeros(tensor_shape) for _ in range(num_to_add)]
        whisper_chunks_padded = whisper_chunks_padded + zero_tensors
    
    # Create a copy of audio_samples to ensure we don't have reference issues
    audio_samples_padded = audio_samples.clone() if isinstance(audio_samples, torch.Tensor) else audio_samples.copy()
    
    # Pad audio_samples to match the new duration
    pad_amount = int(padding_duration * audio_sample_rate)  # Ensure same padding duration
    
    if pad_amount > 0:
        zero_padding = torch.zeros(pad_amount, dtype=audio_samples_padded.dtype)
        audio_samples_padded = torch.cat([audio_samples_padded, zero_padding], dim=0)
    
    return whisper_chunks_padded, audio_samples_padded, padding_duration

def pad_whisper_chunks_to_target(whisper_chunks, tensor_shape, audio_samples, audio_sample_rate, target_frames, fps=25):
    """
    Pads whisper_chunks with zero tensors to reach a target number of frames.
    Also pads audio_samples with zeros to align with whisper_chunks in terms of time.

    Args:
        whisper_chunks (list of torch.Tensor): The original list of tensors (video frames at `fps`).
        tensor_shape (tuple): The shape of the zero tensors to create.
        audio_samples (torch.Tensor): The original audio samples.
        audio_sample_rate (int): The sample rate of the audio.
        target_frames (int): The target number of frames to pad to.
        fps (int): The frames per second of whisper_chunks (default: 25).

    Returns:
        tuple: Modified whisper_chunks, padded audio_samples, padding duration (sec).
    """
    # Create a fresh copy of whisper_chunks to ensure we don't have reference issues
    whisper_chunks_padded = whisper_chunks.copy()
    
    # Check if input is already a list to avoid errors
    if not isinstance(whisper_chunks_padded, list):
        whisper_chunks_padded = list(whisper_chunks_padded)
    
    # Calculate how many frames to add
    current_length = len(whisper_chunks_padded)
    
    # Ensure target_frames is at least as large as current_length
    if target_frames < current_length:
        raise ValueError(f"Target frames ({target_frames}) must be greater than or equal to current length ({current_length})")
    
    num_to_add = target_frames - current_length
    padding_duration = num_to_add / fps  # Time added due to frame padding

    if num_to_add > 0:
        zero_tensors = [torch.zeros(tensor_shape) for _ in range(num_to_add)]
        whisper_chunks_padded = whisper_chunks_padded + zero_tensors
    
    # Create a copy of audio_samples to ensure we don't have reference issues
    audio_samples_padded = audio_samples.clone() if isinstance(audio_samples, torch.Tensor) else audio_samples.copy()
    
    # Pad audio_samples to match the new duration
    pad_amount = int(padding_duration * audio_sample_rate)  # Ensure same padding duration
    
    if pad_amount > 0:
        zero_padding = torch.zeros(pad_amount, dtype=audio_samples_padded.dtype)
        audio_samples_padded = torch.cat([audio_samples_padded, zero_padding], dim=0)
    
    return whisper_chunks_padded, audio_samples_padded, padding_duration

def add_start_silence(audio_samples, audio_sample_rate, silence_duration=0.25):
    """
    Adds a specified duration of silence at the beginning of audio samples.
    
    Args:
        audio_samples (torch.Tensor): The original audio samples.
        audio_sample_rate (int): The sample rate of the audio.
        silence_duration (float): The duration of silence to add in seconds (default: 0.25).
        
    Returns:
        torch.Tensor: The audio samples with silence added at the beginning.
    """
    import torch
    
    # Calculate the number of samples to add based on the sample rate and silence duration
    num_silence_samples = int(silence_duration * audio_sample_rate)
    
    print(f"Adding {silence_duration} seconds of silence ({num_silence_samples} samples) at the start")
    
    # Create a tensor of zeros with the same dtype as the original audio
    silence = torch.zeros(num_silence_samples, dtype=audio_samples.dtype)
    
    # Concatenate the silence with the original audio
    padded_audio = torch.cat([silence, audio_samples], dim=0)
    
    print(f"Original audio length: {len(audio_samples)}, New audio length: {len(padded_audio)}")
    
    return padded_audio