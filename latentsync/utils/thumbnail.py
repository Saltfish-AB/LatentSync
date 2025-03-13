import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def get_text_dimensions(text, font):
    """
    Get text dimensions (width, height) using the appropriate method
    for the current Pillow version.
    
    Args:
        text (str): The text to measure
        font (ImageFont): The font to use
        
    Returns:
        tuple: (width, height) of the text in pixels
    """
    # For Pillow >= 10.0.0
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    # For Pillow >= 8.0.0
    elif hasattr(font, "getlength"):
        return int(font.getlength(text)), font.size
    # For older Pillow versions
    else:
        # Create a temporary image and draw object
        img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(img)
        return draw.textsize(text, font=font)

def truncate_text(text, font, max_width):
    """
    Truncate text to fit within a given width and add ellipsis if needed.
    
    Args:
        text (str): The text to truncate
        font (ImageFont): The font used to render the text
        max_width (int): Maximum width in pixels
        
    Returns:
        str: Truncated text with ellipsis if needed
    """
    # Check if the text already fits
    text_width, _ = get_text_dimensions(text, font)
    if text_width <= max_width:
        return text
    
    # Add characters one by one until we exceed max_width
    ellipsis = "..."
    ellipsis_width, _ = get_text_dimensions(ellipsis, font)
    available_width = max_width - ellipsis_width
    
    result = ""
    for char in text:
        if get_text_dimensions(result + char, font)[0] <= available_width:
            result += char
        else:
            break
    
    return result + ellipsis

def create_video_thumbnail_gif(video_path, output_path, duration=3, fps=5, subtitle_text="Hello there Henrik, I wanted to show you...", max_width=640, max_size_mb=2):
    """
    Create a GIF thumbnail from the first few seconds of a video with a play button overlay
    and subtitle text.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path where the output GIF will be saved
        duration (int): Duration in seconds of video to capture for the GIF
        fps (int): Frames per second for the output GIF
        subtitle_text (str): Text for the subtitle overlay
        max_width (int): Maximum width of the output GIF
        max_size_mb (float): Target maximum file size in MB
    """
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    original_fps = video.get(cv2.CAP_PROP_FPS)
    orig_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate aspect ratio and resize dimensions
    aspect_ratio = orig_width / orig_height
    
    # Resize to fit within max_width while maintaining aspect ratio
    if orig_width > max_width:
        width = max_width
        height = int(width / aspect_ratio)
    else:
        width = orig_width
        height = orig_height
    
    # Calculate how many frames to extract
    total_frames = int(duration * fps)
    frame_interval = int(original_fps / fps)
    
    frames = []
    frame_count = 0
    
    # Create play button triangle - increased size by 50%
    play_button_size = min(width, height) // 3  # Larger play button (was 1/4 before)
    play_button_color = (255, 255, 255, 180)  # White with some transparency
    
    # Create font for subtitle
    # Increased font size (was height // 20, now height // 14)
    font_size = max(18, height // 14)  # Minimum 18pt, scales with video height
    
    try:
        # Try to use a standard font that should be available on most systems
        font = ImageFont.truetype("Arial.ttf", size=font_size)
    except IOError:
        try:
            # Try some other common fonts
            font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        except IOError:
            try:
                # Try system fonts on macOS
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=font_size)
            except IOError:
                try:
                    # Try system fonts on Linux
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=font_size)
                except IOError:
                    # Fallback to default font
                    font = ImageFont.load_default()
    
    while frame_count < total_frames:
        ret, frame = video.read()
        
        if not ret:
            break
        
        # Only process every nth frame based on our desired output fps
        if frame_count % frame_interval == 0:
            # Convert from BGR (OpenCV format) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize the frame if needed
            if orig_width != width or orig_height != height:
                frame_rgb = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
            
            # Convert to PIL Image for easier overlay drawing
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img, 'RGBA')
            
            # Draw play button (triangle) in the center
            center_x, center_y = width // 2, height // 2
            
            # Semi-transparent circle background for play button
            draw.ellipse(
                [(center_x - play_button_size//2, center_y - play_button_size//2),
                 (center_x + play_button_size//2, center_y + play_button_size//2)],
                fill=(0, 0, 0, 128)  # Black with transparency
            )
            
            # Play triangle
            play_triangle = [
                (center_x - play_button_size//4, center_y - play_button_size//3),
                (center_x + play_button_size//3, center_y),
                (center_x - play_button_size//4, center_y + play_button_size//3)
            ]
            draw.polygon(play_triangle, fill=play_button_color)
            
            # Calculate maximum text width (leave some padding on the sides)
            max_text_width = width - 40  # 20px padding on each side
            
            # Truncate text if it's too long
            display_text = truncate_text(subtitle_text, font, max_text_width)
            
            # Draw subtitle at the bottom with black outline and white fill
            # Get text dimensions using our helper function
            text_width, text_height = get_text_dimensions(display_text, font)
            
            # Add padding for the text background - increased for larger text
            text_padding_x, text_padding_y = 20, 10
            
            # Draw black background for subtitle text
            text_bg_left = (width - text_width) // 2 - text_padding_x
            text_bg_top = height - text_height - 20 - text_padding_y
            text_bg_right = text_bg_left + text_width + 2 * text_padding_x
            text_bg_bottom = text_bg_top + text_height + 2 * text_padding_y
            
            # Draw rounded rectangle background for text
            draw.rectangle(
                [text_bg_left, text_bg_top, text_bg_right, text_bg_bottom],
                fill=(0, 0, 0, 180)  # Semi-transparent black
            )
            
            # Position text on top of background
            text_position = ((width - text_width) // 2, text_bg_top + text_padding_y)
            
            # Draw text outline (black) and text (white)
            # Handle both old and new Pillow versions
            if hasattr(draw, "text") and "font" in draw.text.__code__.co_varnames:
                # Older Pillow version
                for offset_x, offset_y in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    draw.text(
                        (text_position[0] + offset_x, text_position[1] + offset_y),
                        display_text,
                        font=font,
                        fill=(0, 0, 0, 255)
                    )
                
                # Draw text (white)
                draw.text(
                    text_position,
                    display_text,
                    font=font,
                    fill=(255, 255, 255, 255)
                )
            else:
                # Newer Pillow version
                for offset_x, offset_y in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    draw.text(
                        xy=(text_position[0] + offset_x, text_position[1] + offset_y),
                        text=display_text,
                        font=font,
                        fill=(0, 0, 0, 255)
                    )
                
                # Draw text (white)
                draw.text(
                    xy=text_position,
                    text=display_text,
                    font=font,
                    fill=(255, 255, 255, 255)
                )
            
            # Convert back to numpy array
            frame_with_overlay = np.array(pil_img)
            frames.append(frame_with_overlay)
        
        frame_count += 1
    
    video.release()
    
    # Save as animated GIF
    # Set loop=0 for infinite looping
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    
    # Check file size and compress if needed
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        compress_gif(output_path, target_size_mb=max_size_mb)
        
    print(f"GIF thumbnail created at: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

def compress_gif(gif_path, target_size_mb=2):
    """
    Compress a GIF to target a specific file size.
    
    Args:
        gif_path (str): Path to the GIF file
        target_size_mb (float): Target file size in MB
    """
    current_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
    if current_size_mb <= target_size_mb:
        return
    
    # Read the GIF
    gif = imageio.mimread(gif_path)
    
    # Calculate compression ratio
    compression_ratio = target_size_mb / current_size_mb
    
    # Options to reduce file size:
    # 1. Reduce color palette
    # 2. Further reduce resolution 
    # 3. Increase optimization
    
    # If compression needed is severe, reduce resolution more
    new_frames = []
    if compression_ratio < 0.5:
        # Reduce resolution by 30%
        scale_factor = 0.7
        for frame in gif:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            new_frames.append(resized)
    else:
        new_frames = gif
    
    # Save with more aggressive optimization
    imageio.mimsave(
        gif_path, 
        new_frames, 
        loop=0,  # Infinite loop
        optimize=True,
        quantizer='nq'  # Neural-net quantizer for better compression
    )

# Example usage
if __name__ == "__main__":
    video_path = "input_video.mp4"  # Replace with your video path
    output_path = "thumbnail.gif"
    subtitle_text = "Hi Henrik! Alexandra here, Head of Customer success at AndSend. I noticed you signed up"
    
    create_video_thumbnail_gif(
        video_path=video_path,
        output_path=output_path,
        duration=3,  # First 3 seconds
        fps=5,       # Low FPS to keep file size small
        subtitle_text=subtitle_text
    )