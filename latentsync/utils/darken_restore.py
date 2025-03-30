import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path


def create_natural_face_mask(face_landmarks, image_shape, expansion_factor=1.1, feather_amount=40):
    """
    Create an anatomically correct face mask based on facial landmarks.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        image_shape: Shape of the image (height, width)
        expansion_factor: Factor to expand the convex hull (1.1 = 10% expansion)
        feather_amount: Pixels to feather the mask
        
    Returns:
        Feathered mask as a float32 array (0.0-1.0)
    """
    
    height, width = image_shape[:2]
    
    # Create a blank mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Collect facial landmark points
    points = []
    for landmark in face_landmarks.landmark:
        x, y = int(landmark.x * width), int(landmark.y * height)
        points.append((x, y))
    
    # Convert to numpy array
    points = np.array(points, dtype=np.int32)
    
    # Find the center of the face
    face_center = np.mean(points, axis=0).astype(np.int32)
    
    # Expand points outward from center for a larger coverage area
    if expansion_factor > 1.0:
        vectors = points - face_center
        expanded_points = face_center + (vectors * expansion_factor).astype(np.int32)
        
        # Make sure all points are within the image
        expanded_points[:, 0] = np.clip(expanded_points[:, 0], 0, width - 1)
        expanded_points[:, 1] = np.clip(expanded_points[:, 1], 0, height - 1)
        points = expanded_points
    
    # Compute the convex hull to get a natural face shape
    hull = cv2.convexHull(points)
    
    # Draw the convex hull on the mask
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Add critical facial landmark points that might not be inside the convex hull
    # (ears, hair, neck, etc.)
    boundary_indices = [
        # Jawline and chin
        *range(0, 17),  # Entire jawline
        # Ears
        *range(100, 110), *range(127, 137),  
        # Edges of eyebrows
        46, 55, 276, 285,
        # Top of forehead
        *range(10, 108, 8)  # Sample points across forehead
    ]
    
    # Create a secondary mask for these points
    secondary_mask = np.zeros_like(mask)
    boundary_points = []
    
    for idx in boundary_indices:
        if idx < len(face_landmarks.landmark):
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * width), int(lm.y * height)
            boundary_points.append((x, y))
    
    # If we have enough boundary points
    if len(boundary_points) > 3:
        boundary_hull = cv2.convexHull(np.array(boundary_points))
        cv2.fillConvexPoly(secondary_mask, boundary_hull, 255)
        
        # Combine masks
        mask = cv2.bitwise_or(mask, secondary_mask)
    
    # Expand the mask slightly with dilation to ensure full coverage
    kernel_size = max(1, int(min(width, height) * 0.01))  # 1% of the smaller dimension
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Apply feathering to the mask
    mask_float = mask.astype(np.float32) / 255.0
    if feather_amount > 0:
        mask_float = cv2.GaussianBlur(mask_float, (feather_amount*2+1, feather_amount*2+1), 0)
    
    return mask_float


def enhance_face_brightness(temp_dir, frame_start, frame_end, brightness_factor=1.3, 
                           feather_amount=40, max_value=235, verbose=True, 
                           preserve_alpha=True, preserve_quality=True):
    """
    Enhance brightness in the face region with a natural, anatomically correct mask.
    Works on individual frames stored in a directory, reading and overwriting each frame.
    
    Args:
        temp_dir: Path to directory containing frames
        frame_start: Starting frame number
        frame_end: Ending frame number (inclusive)
        brightness_factor: Factor to increase brightness (>1.0 increases brightness)
        feather_amount: Pixels to feather for smooth transition
        max_value: Maximum pixel value to prevent overexposure (0-255)
        verbose: Whether to print detailed information
        preserve_alpha: Whether to preserve alpha channel if present
        preserve_quality: Whether to use lossless compression settings
    """
    
    if verbose:
        print(f"\nEnhancing frames in directory: {temp_dir}")
        print(f"Processing frames {frame_start} to {frame_end}")
        print(f"Brightness factor: {brightness_factor}")
        print(f"Quality preservation: {'ON' if preserve_quality else 'OFF'}")
        print(f"Alpha channel preservation: {'ON' if preserve_alpha else 'OFF'}")
        
        if brightness_factor > 1.0:
            print(f"Mode: Brightening faces")
        elif brightness_factor < 1.0:
            print(f"Mode: Darkening faces")
        else:
            print(f"Mode: No brightness change (factor = 1.0)")
            
        print(f"Feather amount: {feather_amount} pixels")
        print(f"Maximum pixel value: {max_value} (prevents overexposure)")
    
    # Ensure directory exists
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        print(f"ERROR: Directory does not exist: {temp_dir}")
        return False
    
    # Initialize face mesh with more landmarks for better accuracy
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True  # Get more accurate landmarks
    )
    
    try:
        # For tracking face position to smooth between frames
        last_face_mask = None
        alpha_smooth = 0.7  # Smoothing factor
        first_valid_mask = None
        
        face_detected_count = 0
        frame_count = 0
        
        # Create a progress bar if verbose
        total_frames = frame_end - frame_start + 1
        pbar = tqdm(total=total_frames, desc="Processing frames") if verbose else None
        
        # Store original image format for each frame to preserve it
        original_formats = {}
        
        # First pass to find a valid face mask
        if verbose:
            print("First pass: Finding first valid face mask...")
        
        # Try to find a valid face in the first few frames
        for i in range(frame_start, min(frame_start + 100, frame_end + 1), 5):
            frame_path = str(temp_dir / f"frame_{i:05d}.png")
            if not Path(frame_path).exists():
                continue
                
            # Read frame with unchanged flag to preserve all channels
            frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                continue
            
            # Store original format
            original_formats[i] = frame.shape
                
            # Check if the image has an alpha channel
            has_alpha = frame.shape[2] == 4 if len(frame.shape) > 2 else False
            
            # Extract BGR channels for face detection
            if has_alpha:
                # If image has alpha channel, separate it
                bgr_frame = frame[:, :, 0:3]
            else:
                bgr_frame = frame
                
            # Try to find a face
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Found a face, create mask
                first_valid_mask = create_natural_face_mask(
                    results.multi_face_landmarks[0], 
                    rgb_frame.shape,
                    expansion_factor=1.2,
                    feather_amount=feather_amount
                )
                if verbose:
                    print(f"Found first valid face at frame {i}")
                break
        
        # Process each frame
        for i in range(frame_start, frame_end + 1):
            frame_path = str(temp_dir / f"frame_{i:05d}.png")
            
            # Check if the frame exists
            if not Path(frame_path).exists():
                if verbose:
                    print(f"WARNING: Frame does not exist: {frame_path}")
                if pbar:
                    pbar.update(1)
                continue
            
            # Read the frame with unchanged flag to preserve all channels
            frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                if verbose:
                    print(f"WARNING: Could not read frame: {frame_path}")
                if pbar:
                    pbar.update(1)
                continue
            
            # Store original format if not already stored
            if i not in original_formats:
                original_formats[i] = frame.shape
            
            # Check if the image has an alpha channel
            has_alpha = frame.shape[2] == 4 if len(frame.shape) > 2 else False
            
            # Extract BGR and alpha channels if present
            if has_alpha and preserve_alpha:
                # If image has alpha channel, separate it
                bgr_frame = frame[:, :, 0:3]
                alpha_channel = frame[:, :, 3]
            else:
                bgr_frame = frame
                alpha_channel = None
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            # Create a copy of the original BGR frame
            processed_frame = bgr_frame.copy()
            current_mask = None
            
            if results.multi_face_landmarks:
                face_detected_count += 1
                
                # Create natural face mask
                current_mask = create_natural_face_mask(
                    results.multi_face_landmarks[0], 
                    rgb_frame.shape,
                    expansion_factor=1.2,
                    feather_amount=feather_amount
                )
                
                # Smooth mask between frames
                if last_face_mask is not None:
                    current_mask = alpha_smooth * last_face_mask + (1 - alpha_smooth) * current_mask
                
                # Save for next frame
                last_face_mask = current_mask.copy()
            elif last_face_mask is not None:
                # If no face detected but we have a previous mask, use it
                current_mask = last_face_mask.copy()
                # Fade out gradually
                last_face_mask = last_face_mask * 0.8
            
            # Apply brightness adjustment if we have a mask
            if current_mask is not None:
                # Convert to HSV for brightness adjustment
                hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                h, s, v = cv2.split(hsv_frame)
                
                # Calculate brightness adjustment
                if brightness_factor > 1.0:
                    # Brightening: Increase V channel based on factor
                    # But limit maximum value to prevent overexposure
                    v_enhanced = v * (1.0 + current_mask * (brightness_factor - 1.0))
                    v_enhanced = np.clip(v_enhanced, 0, max_value)
                else:
                    # Darkening: Decrease V channel
                    v_enhanced = v * (1.0 - current_mask * (1.0 - brightness_factor))
                    v_enhanced = np.clip(v_enhanced, 0, 255)
                
                # Merge channels back
                hsv_adjusted = cv2.merge([h, s, v_enhanced])
                hsv_adjusted = hsv_adjusted.astype(np.uint8)
                
                # Convert back to BGR
                processed_frame = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
            else:
                # Use first valid mask for frames without detected faces
                if first_valid_mask is not None:
                    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                    h, s, v = cv2.split(hsv_frame)
                    
                    if brightness_factor > 1.0:
                        v_enhanced = v * (1.0 + first_valid_mask * (brightness_factor - 1.0) * 0.8)
                        v_enhanced = np.clip(v_enhanced, 0, max_value)
                    else:
                        v_enhanced = v * (1.0 - first_valid_mask * (1.0 - brightness_factor) * 0.8)
                        v_enhanced = np.clip(v_enhanced, 0, 255)
                    
                    hsv_adjusted = cv2.merge([h, s, v_enhanced])
                    hsv_adjusted = hsv_adjusted.astype(np.uint8)
                    
                    processed_frame = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
            
            # Recombine with alpha channel if it exists
            if has_alpha and preserve_alpha and alpha_channel is not None:
                # Merge BGR with original alpha
                final_frame = np.zeros((processed_frame.shape[0], processed_frame.shape[1], 4), dtype=np.uint8)
                final_frame[:, :, 0:3] = processed_frame
                final_frame[:, :, 3] = alpha_channel
            else:
                final_frame = processed_frame
            
            # Preserve original bit depth and other parameters
            original_shape = original_formats[i]
            if len(original_shape) > 2 and original_shape[2] != final_frame.shape[2]:
                # If channel counts don't match, adjust
                if original_shape[2] == 3 and final_frame.shape[2] == 4:
                    # Original was 3 channels but we have 4, drop alpha
                    final_frame = final_frame[:, :, 0:3]
                elif original_shape[2] == 4 and final_frame.shape[2] == 3:
                    # Original was 4 channels but we have 3, add alpha
                    alpha = np.full((original_shape[0], original_shape[1]), 255, dtype=np.uint8)
                    final_frame = np.dstack((final_frame, alpha))
            
            # Write the frame back with appropriate quality settings
            if preserve_quality:
                cv2.imwrite(frame_path, final_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                cv2.imwrite(frame_path, final_frame)
            
            frame_count += 1
            if pbar:
                pbar.update(1)
        
        # Clean up
        if pbar:
            pbar.close()
        
        if verbose:
            print(f"Processed {frame_count} frames, detected faces in {face_detected_count} frames")
            if frame_count > 0:
                print(f"Face detection rate: {face_detected_count/frame_count*100:.1f}%")
        
        # If no faces detected, warn the user
        if face_detected_count == 0 and verbose:
            print("WARNING: No faces were detected in any frames.")
            return False
        
        return True
    
    except Exception as e:
        if verbose:
            print(f"Error processing frames: {e}")
            import traceback
            traceback.print_exc()
        return False
    finally:
        # Release resources
        face_mesh.close()
    
    return False


def calculate_inverse_factor(original_factor):
    """
    Calculate the inverse factor needed to restore brightness.
    
    Args:
        original_factor: The factor used to reduce brightness (e.g., 0.7)
    
    Returns:
        Inverse factor needed to restore original brightness
    """
    if original_factor >= 1.0:
        return 1.0  # No inversion needed if original didn't darken
    
    # Apply a more conservative restoration approach
    # Instead of direct mathematical inverse (1/factor), use a dampened approach
    
    # Calculate how much darkening was applied (e.g., for 0.8, it's 0.2 or 20%)
    darkening_amount = 1.0 - original_factor
    
    # Apply a dampened restoration (70-80% of what would be mathematically "perfect")
    # This prevents over-brightening while still improving the image
    restoration_strength = 1  # Adjust this value (0.6-0.8) to control restoration strength
    
    # Calculate a dampened inverse factor
    inverse = 1.0 + (darkening_amount / original_factor) * restoration_strength
    print("brightness factor", inverse)
    return round(inverse, 2)