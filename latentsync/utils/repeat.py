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

def pad_whisper_chunks_end(whisper_chunks, tensor_shape, audio_samples, audio_sample_rate, fps=25):
    """
    Pads whisper_chunks with zero tensors to make its length divisible by 16.
    Also pads audio_samples with zeros to align with whisper_chunks in terms of time.

    This version pads at the end instead of the beginning.

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
        whisper_chunks = whisper_chunks + zero_tensors  # Append zero tensors
    
    # Compute expected total audio length (using updated whisper_chunks duration)
    total_duration = len(whisper_chunks) / fps  # New total duration in seconds
    expected_audio_length = int(total_duration * audio_sample_rate)  # Expected audio samples
    
    # Pad audio_samples to match the new duration
    audio_length = audio_samples.shape[0]
    pad_amount = int(padding_duration * audio_sample_rate)  # Ensure same padding duration
    
    if pad_amount > 0:
        zero_padding = torch.zeros(pad_amount, dtype=audio_samples.dtype)
        audio_samples = torch.cat([audio_samples, zero_padding], dim=0)
    
    return whisper_chunks, audio_samples, padding_duration
