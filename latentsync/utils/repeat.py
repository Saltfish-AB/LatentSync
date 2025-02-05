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
