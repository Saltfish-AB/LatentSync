import os
import sys
import cv2
import torch
import numpy as np
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.detection import init_detection_model
from facelib.utils.face_restoration_helper import FaceRestoreHelper

class MouthEnhancer:
    """
    A class for enhancing mouth regions in facial images using CodeFormer.
    """
    
    def __init__(self, model_path=None, device=None, debug_mode=False):
        """
        Initialize the MouthEnhancer.
        
        Args:
            model_path: Optional path to a pre-downloaded CodeFormer model.
                        If None, the model will be downloaded automatically.
            device: Optional torch device. If None, will use CUDA if available, otherwise CPU.
            debug_mode: Whether to save debug images showing the detected mouth regions.
        """
        # Add current directory to Python path
        sys.path.append(os.getcwd())
        
        # Set up device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set debug mode
        self.debug_mode = debug_mode
        self.debug_dir = "temp"
        
        # Initialize models
        self._init_codeformer_model(model_path)
        
        # Initialize face helper
        self.face_helper = FaceRestoreHelper(
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
            upscale_factor=1
        )
    
    def _init_codeformer_model(self, model_path):
        """Initialize the CodeFormer model."""
        # Create the model
        self.net = ARCH_REGISTRY.get('CodeFormer')(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=['32', '64', '128', '256']
        ).to(self.device)
        
        # Load model weights
        if model_path is None:
            # Download pre-trained model
            pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
            model_path = load_file_from_url(
                url=pretrain_model_url,
                model_dir='weights/CodeFormer',
                progress=True,
                file_name=None
            )
        
        checkpoint = torch.load(model_path)['params_ema']
        self.net.load_state_dict(checkpoint)
        self.net.eval()
    
    def process(self, img, fidelity_weight=0.5, debug_dir=None, scale_factor=0.25):
        """
        Process an image to enhance the mouth region with optimizations for speed.
        
        Args:
            img: Input image (numpy array) or path to an image file
            fidelity_weight: Balance between quality and fidelity (0-1)
            debug_dir: Directory to save debug images (if debug_mode is True)
                    If None, debug images will be saved in the current directory
            scale_factor: Factor to scale down the image for faster processing (0-1)
        
        Returns:
            Enhanced image as a numpy array
        """
        # Handle both image paths and numpy arrays
        if isinstance(img, str):
            # Read the image from file
            input_img = cv2.imread(img, cv2.IMREAD_COLOR)
            if input_img is None:
                raise ValueError(f"Could not read image file: {img}")
        else:
            # Assume it's already a numpy array
            input_img = img.copy()
        
        # Store original dimensions
        original_img = input_img.copy()
        original_height, original_width = input_img.shape[:2]
        
        # OPTIMIZATION: Resize input image for faster processing
        if scale_factor < 1.0:
            process_height = int(original_height * scale_factor)
            process_width = int(original_width * scale_factor)
            input_img = cv2.resize(input_img, (process_width, process_height), interpolation=cv2.INTER_AREA)
            print(f"Resized for processing: {process_width}x{process_height}")
        
        # Reset the face helper for new image
        self.face_helper.clean_all()
        
        # OPTIMIZATION: Use a smaller minimum face size for detection since we've scaled down
        self.face_helper.read_image(input_img)
        # Detect face - set only_center_face=True assuming we're focusing on a single face
        num_faces = self.face_helper.get_face_landmarks_5(only_center_face=True, resize=None, eye_dist_threshold=5)
        
        if num_faces == 0:
            print("No face detected! Returning original image.")
            return original_img
        
        # Align and warp the faces
        self.face_helper.align_warp_face()
        
        # Process each face (usually just one with only_center_face=True)
        for idx, aligned_face in enumerate(self.face_helper.cropped_faces):
            # Get landmarks for this face
            landmarks = self.face_helper.all_landmarks_5[idx]
            
            # Store original face
            original_face = aligned_face.copy()
            h, w = aligned_face.shape[:2]
            
            # Get mouth landmarks (last 2 points are mouth corners)
            mouth_left = landmarks[3]
            mouth_right = landmarks[4]
            
            # Calculate mouth center and size
            mouth_center_x = (mouth_left[0] + mouth_right[0]) / 2
            mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            
            # Define mouth region with dynamic size based on landmarks
            # OPTIMIZATION: Slightly expanded padding to ensure we get the full mouth
            mouth_x1 = max(0, int(mouth_center_x - mouth_width * 0.6))
            mouth_x2 = min(w, int(mouth_center_x + mouth_width * 0.6))
            mouth_y1 = max(0, int(mouth_center_y - mouth_width * 0.4))
            mouth_y2 = min(h, int(mouth_center_y + mouth_width * 0.3))
            
            # OPTIMIZATION: Process only the mouth region instead of the whole face
            mouth_region = aligned_face[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
            
            # Process the mouth region with CodeFormer
            try:
                # Convert mouth region to tensor
                mouth_region_t = img2tensor(mouth_region / 255., bgr2rgb=True, float32=True)
                normalize(mouth_region_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                mouth_region_t = mouth_region_t.unsqueeze(0).to(self.device)
                
                # OPTIMIZATION: Use torch.cuda.amp for mixed precision if available
                if hasattr(torch.cuda, 'amp') and self.device != torch.device('cpu'):
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            output = self.net(mouth_region_t, w=fidelity_weight, adain=True)[0]
                else:
                    with torch.no_grad():
                        output = self.net(mouth_region_t, w=fidelity_weight, adain=True)[0]
                        
                restored_mouth = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
                
            except Exception as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_mouth = tensor2img(mouth_region_t, rgb2bgr=True, min_max=(-1, 1))
            
            restored_mouth = restored_mouth.astype('uint8')
            
            # Ensure restored mouth has the same dimensions as the original mouth region
            if restored_mouth.shape[:2] != mouth_region.shape[:2]:
                print(f"Resizing restored mouth to match original dimensions")
                restored_mouth = cv2.resize(restored_mouth, (mouth_region.shape[1], mouth_region.shape[0]), 
                                        interpolation=cv2.INTER_LINEAR)
            
            # Create a smooth blending mask for the mouth region
            blend_mask = np.zeros((mouth_y2-mouth_y1, mouth_x2-mouth_x1), dtype=np.float32)
            # Create an inner rectangle with full opacity
            inner_padding = int(min(mouth_region.shape[0], mouth_region.shape[1]) * 0.1)
            if inner_padding > 0:
                cv2.rectangle(blend_mask, 
                            (inner_padding, inner_padding), 
                            (blend_mask.shape[1]-inner_padding, blend_mask.shape[0]-inner_padding), 
                            1.0, -1)
                # Blur the edges for smooth transition
                blend_mask = cv2.GaussianBlur(blend_mask, (inner_padding*2+1, inner_padding*2+1), 0)
            else:
                # If region is too small for padding, use full opacity
                blend_mask.fill(1.0)
                
            # Apply blending mask
            blend_mask_3d = np.stack([blend_mask] * 3, axis=2)
            blended_mouth = restored_mouth * blend_mask_3d + mouth_region * (1 - blend_mask_3d)
            blended_mouth = blended_mouth.astype('uint8')
            
            # Place the blended mouth back into the aligned face
            aligned_face[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = blended_mouth
            
            # Save debug images if debug mode is enabled
            if self.debug_mode and debug_dir is not None:
                os.makedirs(debug_dir, exist_ok=True)
                
                # Create debug image showing the mouth region on aligned face
                debug_face = original_face.copy()
                cv2.rectangle(debug_face, (mouth_x1, mouth_y1), (mouth_x2, mouth_y2), (0, 255, 0), 2)
                # Mark landmarks
                for lm in landmarks:
                    cv2.circle(debug_face, (int(lm[0]), int(lm[1])), 3, (0, 0, 255), -1)
                
                # Save debug images
                debug_path = os.path.join(debug_dir, f"debug_aligned_face_{idx}.png")
                mouth_path = os.path.join(debug_dir, f"debug_mouth_region_{idx}.png")
                restored_path = os.path.join(debug_dir, f"debug_restored_mouth_{idx}.png")
                
                imwrite(debug_face, debug_path)
                imwrite(mouth_region, mouth_path)
                imwrite(restored_mouth, restored_path)
                print(f"Debug images saved to: {debug_dir}")
            
            # Add the enhanced face to the helper
            self.face_helper.add_restored_face(aligned_face)
        
        # Get the inverse affine transform
        self.face_helper.get_inverse_affine(None)
        
        # Paste the enhanced face back into the input image
        result_img = self.face_helper.paste_faces_to_input_image(upsample_img=input_img)
        
        # If we resized earlier, resize back to original dimensions
        if scale_factor < 1.0:
            result_img = cv2.resize(result_img, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        
        return result_img
    
    def process_and_save(self, input_path, output_path, fidelity_weight=0.5):
        """
        Process an image file and save the result to a new file.
        
        Args:
            input_path: Path to the input image file
            output_path: Path to save the enhanced image
            fidelity_weight: Balance between quality and fidelity (0-1)
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Process the image
        debug_dir = os.path.dirname(output_path) if self.debug_mode else None
        result_img = self.process(input_path, fidelity_weight, debug_dir)
        
        # Save the result
        imwrite(result_img, output_path)
        print(f"Enhanced image saved to: {output_path}")
        
        return result_img

    def process_warped_frame(self, face, inverse_affine, w_up, h_up, fidelity_weight=0.5, debug_dir=None):
        """
        Process a frame that has already been warped with warpAffine.
        
        Args:
            face: Input face image (numpy array)
            inverse_affine: Inverse affine transformation matrix
            w_up: Target width
            h_up: Target height
            fidelity_weight: Balance between quality and fidelity (0-1)
            debug_dir: Directory to save debug images (if debug_mode is True)
        
        Returns:
            Enhanced frame as a numpy array
        """
        # Store original dimensions
        original_face = face.copy()
        h, w = face.shape[:2]
        print(f"Processing warped frame with dimensions: {w}x{h}")
        
        # Reset the face helper for new frame
        self.face_helper.clean_all()
        
        # Use face as aligned face directly (since it's already warped)
        aligned_face = face.copy()
        
        # Since this is a pre-warped face, we need to estimate landmarks
        # For a warped face, we can use approximate positions based on facial proportions
        # These are approximate mouth corner positions for a typical aligned face
        landmark_left = np.array([w * 0.3, h * 0.7])  # Left mouth corner
        landmark_right = np.array([w * 0.7, h * 0.7]) # Right mouth corner
        
        # Calculate mouth center and size
        mouth_center_x = (landmark_left[0] + landmark_right[0]) / 2
        mouth_center_y = (landmark_left[1] + landmark_right[1]) / 2
        mouth_width = np.linalg.norm(landmark_right - landmark_left)
        
        # Define mouth region with dynamic size based on estimated landmarks
        mouth_x1 = max(0, int(mouth_center_x - mouth_width * 0.5))
        mouth_x2 = min(w, int(mouth_center_x + mouth_width * 0.5))
        mouth_y1 = max(0, int(mouth_center_y - mouth_width * 0.3))
        mouth_y2 = min(h, int(mouth_center_y + mouth_width * 0.1))
        
        # Create a mouth mask with blurred edges for smooth blending
        mouth_mask = np.zeros((h, w), dtype=np.float32)
        cv2.rectangle(mouth_mask, (mouth_x1, mouth_y1), (mouth_x2, mouth_y2), 1, -1)
        mouth_mask = cv2.GaussianBlur(mouth_mask, (31, 31), 0)
        
        # Process the face with CodeFormer
        aligned_face_t = img2tensor(aligned_face / 255., bgr2rgb=True, float32=True)
        normalize(aligned_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        aligned_face_t = aligned_face_t.unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                output = self.net(aligned_face_t, w=fidelity_weight, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f"Failed inference for CodeFormer: {error}")
            restored_face = tensor2img(aligned_face_t, rgb2bgr=True, min_max=(-1, 1))
        
        restored_face = restored_face.astype('uint8')
        
        # Ensure restored face has the same dimensions as the original face
        if restored_face.shape[:2] != (h, w):
            print(f"Resizing restored face from {restored_face.shape[1]}x{restored_face.shape[0]} to {w}x{h}")
            restored_face = cv2.resize(restored_face, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Apply mouth mask for smooth blending
        mouth_mask_3d = np.stack([mouth_mask] * 3, axis=2)
        blended_face = restored_face * mouth_mask_3d + original_face * (1 - mouth_mask_3d)
        blended_face = blended_face.astype('uint8')
        
        # Save debug images if debug mode is enabled
        if self.debug_mode and debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create debug image showing the mouth region on aligned face
            debug_face = original_face.copy()
            cv2.rectangle(debug_face, (mouth_x1, mouth_y1), (mouth_x2, mouth_y2), (0, 255, 0), 2)
            
            # Save debug face
            debug_path = os.path.join(debug_dir, f"debug_warped_frame.png")
            imwrite(debug_face, debug_path)
            print(f"Debug warped frame with mouth region saved to: {debug_path}")
        
        # Apply inverse affine transformation to get back to original coordinates
        result_frame = cv2.warpAffine(blended_face, inverse_affine, (w_up, h_up), flags=cv2.INTER_LANCZOS4)
        
        return result_frame


# Example usage
if __name__ == "__main__":
    # Initialize the enhancer
    enhancer = MouthEnhancer(debug_mode=False)
    
    # Process an image
    enhancer.process_and_save(
        input_path="temp_faces/face_0001.jpg",
        output_path="temp_faces/enhanced_mouth.jpg",
        fidelity_weight=0.5
    )
