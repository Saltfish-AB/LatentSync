import cv2
import numpy as np
import os
import subprocess
import tempfile
import sys
import mediapipe as mp
from tqdm import tqdm
import re


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

def enhance_face_brightness(input_path, output_path, brightness_factor=1.3, 
                           feather_amount=40, max_value=235, verbose=True):
    """
    Enhance brightness in the face region with a natural, anatomically correct mask.
    For previously darkened videos, use 1/original_factor to restore.
    For example, if a video was darkened with factor 0.7, use 1/0.7 = 1.43 to restore.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        brightness_factor: Factor to increase brightness (>1.0 increases brightness)
        feather_amount: Pixels to feather for smooth transition
        max_value: Maximum pixel value to prevent overexposure (0-255)
        verbose: Whether to print detailed information
    """
    if verbose:
        print(f"\nEnhancing video: {input_path}")
        print(f"Output will be saved to: {output_path}")
        print(f"Brightness factor: {brightness_factor}")
        
        if brightness_factor > 1.0:
            print(f"Mode: Brightening faces")
        elif brightness_factor < 1.0:
            print(f"Mode: Darkening faces")
        else:
            print(f"Mode: No brightness change (factor = 1.0)")
            
        print(f"Feather amount: {feather_amount} pixels")
        print(f"Maximum pixel value: {max_value} (prevents overexposure)")
    
    # Ensure input file exists
    if not os.path.exists(input_path):
        print(f"ERROR: Input file does not exist: {input_path}")
        return False
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Initialize face mesh with more landmarks for better accuracy
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True  # Get more accurate landmarks
    )
    
    try:
        # Open the video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video: {input_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if verbose:
            print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Process and save frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            if verbose:
                print(f"Temporary frames directory: {frames_dir}")
            
            frame_count = 0
            face_detected_count = 0
            pbar = tqdm(total=total_frames, desc="Processing frames") if verbose else None
            
            # For tracking face position to smooth between frames
            last_face_mask = None
            alpha_smooth = 0.7  # Smoothing factor
            
            # First pass to find first valid face mask
            if verbose:
                print("First pass: Finding first valid face mask...")
            
            # Skip ahead in video to find a face faster
            first_valid_mask = None
            original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            for skip_i in range(0, min(300, total_frames), 5):  # Check first 300 frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, skip_i)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Try to find a face
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    # Found a face, create mask
                    first_valid_mask = create_natural_face_mask(
                        results.multi_face_landmarks[0], 
                        frame.shape,
                        expansion_factor=1.2,
                        feather_amount=feather_amount
                    )
                    if verbose:
                        print(f"Found first valid face at frame {skip_i}")
                    break
            
            # Reset video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
            last_face_mask = first_valid_mask
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                # Create a copy of the original frame
                processed_frame = frame.copy()
                current_mask = None
                
                if results.multi_face_landmarks:
                    face_detected_count += 1
                    
                    # Create natural face mask
                    current_mask = create_natural_face_mask(
                        results.multi_face_landmarks[0], 
                        frame.shape,
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
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                    h, s, v = cv2.split(hsv_frame)
                    
                    # Calculate brightness adjustment
                    if brightness_factor > 1.0:
                        # Brightening: Increase V channel based on factor
                        # But limit maximum value to prevent overexposure
                        v_enhanced = v * (1.0 + current_mask * (brightness_factor - 1.0))
                        v_enhanced = np.clip(v_enhanced, 0, max_value)
                    else:
                        # Darkening: Decrease V channel (same as original darkening)
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
                        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
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
                
                # Save the processed frame
                frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(frame_path, processed_frame)
                
                frame_count += 1
                if pbar:
                    pbar.update(1)
            
            # Clean up
            cap.release()
            if pbar:
                pbar.close()
            
            if verbose:
                print(f"Processed {frame_count} frames, detected faces in {face_detected_count} frames")
                print(f"Face detection rate: {face_detected_count/frame_count*100:.1f}%")
            
            # If no faces detected, warn the user
            if face_detected_count == 0:
                print("WARNING: No faces were detected in the video.")
                return False
            
            # Create final video with FFmpeg
            if verbose:
                print("Creating final video with FFmpeg...")
            
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(frames_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-profile:v', 'high',
                '-level', '4.1',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ]
            
            # Add output path
            cmd.append(output_path)
            
            # Try the simple version first (no audio)
            if verbose:
                print(f"Running command (video only): {' '.join(cmd)}")
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # If successful and original has audio, add audio in a separate step
            if result.returncode == 0:
                # Check if original has audio
                audio_check = subprocess.run(
                    ['ffmpeg', '-i', input_path, '-c', 'copy', '-f', 'null', '-'],
                    stderr=subprocess.PIPE, stdout=subprocess.PIPE
                )
                
                has_audio = "Audio: " in audio_check.stderr.decode()
                
                if has_audio:
                    if verbose:
                        print("Original video has audio. Adding audio track...")
                    
                    # Create a temporary file for the video-only version
                    temp_video = output_path + ".temp.mp4"
                    os.rename(output_path, temp_video)
                    
                    # Combine video and audio
                    audio_cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_video,        # Processed video
                        '-i', input_path,        # Original with audio
                        '-c:v', 'copy',          # Copy video stream as is
                        '-c:a', 'copy',          # Copy audio stream as is
                        '-map', '0:v',           # Use video from first input
                        '-map', '1:a',           # Use audio from second input
                        '-shortest',             # Match durations
                        output_path
                    ]
                    
                    if verbose:
                        print(f"Running command (adding audio): {' '.join(audio_cmd)}")
                    
                    audio_result = subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_video)
                    except:
                        pass
                    
                    if audio_result.returncode == 0:
                        if verbose:
                            print("Successfully added audio track.")
                    else:
                        if verbose:
                            print(f"Failed to add audio. Using video-only version.")
                            # If adding audio failed, copy back the video-only version
                            if os.path.exists(temp_video):
                                os.rename(temp_video, output_path)
                
                return True
            else:
                if verbose:
                    print(f"FFmpeg failed with error code {result.returncode}")
                    print(f"Error message: {result.stderr.decode()}")
                return False
    
    except Exception as e:
        if verbose:
            print(f"Error processing video: {e}")
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

def try_ffmpeg_version():
    """Check FFmpeg version and capabilities"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("FFmpeg found:")
            # Extract the first line
            version_line = result.stdout.decode().split('\n')[0]
            print(f"  {version_line}")
            return True
        else:
            print("FFmpeg found but returned an error.")
            return False
    except Exception as e:
        print(f"FFmpeg not found or error checking version: {e}")
        return False

# Main script
if __name__ == "__main__":
    print("\n--- Face Brightness Enhancement ---\n")
    
    # Check FFmpeg
    if not try_ffmpeg_version():
        print("ERROR: FFmpeg is required for this script to work.")
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Face Brightness Enhancement")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", nargs="?", help="Output video path (default: input_brightened.mp4)")
    parser.add_argument("--factor", type=float, help="Brightness factor (>1.0 to brighten, <1.0 to darken)")
    parser.add_argument("--restore", type=float, help="Original darkening factor to restore from (e.g. 0.7)")
    parser.add_argument("--feather", type=int, default=40, help="Feather amount in pixels")
    parser.add_argument("--max-value", type=int, default=235, 
                       help="Maximum pixel value to prevent overexposure (0-255)")
    
    args = parser.parse_args()
    
    input_path = args.input
    
    # Set default output path if not provided
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_brightened{ext}"
    
    # Determine brightness factor
    brightness_factor = 1.25  # Default enhancement factor
    
    if args.restore and args.factor:
        print("ERROR: Cannot use both --factor and --restore. Please use only one.")
        sys.exit(1)
    elif args.restore:
        # Calculate inverse factor to restore original brightness
        brightness_factor = calculate_inverse_factor(args.restore)
        print(f"Restoring from original factor {args.restore} with inverse factor {brightness_factor}")
    elif args.factor:
        brightness_factor = args.factor
    
    # Process the video
    success = enhance_face_brightness(
        input_path, 
        output_path, 
        brightness_factor=brightness_factor,
        feather_amount=args.feather,
        max_value=args.max_value,
        verbose=True
    )
    
    if success:
        print("\n--- Processing complete ---")
        print(f"Enhanced video saved to: {output_path}")
    else:
        print("\n--- Processing failed ---")
        sys.exit(1)