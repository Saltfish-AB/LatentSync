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
        self.debug_dir = "temp"
    
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
    
    def process(self, img, fidelity_weight=0.85):
        """
        Process an image to enhance the mouth region.
        
        Args:
            img: Input image (numpy array) or path to an image file
            fidelity_weight: Balance between quality and fidelity (0-1)
        
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
        print(f"Processing image with dimensions: {original_width}x{original_height}")
        
        # Reset the face helper for new image
        self.face_helper.clean_all()
        
        # Detect face and landmarks
        self.face_helper.read_image(input_img)
        num_faces = self.face_helper.get_face_landmarks_5(only_center_face=False, resize=None, eye_dist_threshold=5)
        print(f"Detected {num_faces} faces")
        
        if num_faces == 0:
            print("No face detected! Returning original image.")
            return original_img
        
        # Align and warp the faces
        self.face_helper.align_warp_face()
        
        # Process each face
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
            mouth_x1 = max(0, int(mouth_center_x - mouth_width * 0.5))
            mouth_x2 = min(w, int(mouth_center_x + mouth_width * 0.5))
            mouth_y1 = max(0, int(mouth_center_y - mouth_width * 0.3))
            mouth_y2 = min(h, int(mouth_center_y + mouth_width * 0.1))
            
            # Create a mouth mask with blurred edges for smooth blending
            mouth_mask = np.zeros((h, w), dtype=np.float32)
            cv2.rectangle(mouth_mask, (mouth_x1, mouth_y1), (mouth_x2, mouth_y2), 1, -1)
            mouth_mask = cv2.GaussianBlur(mouth_mask, (31, 31), 0)
            
            # Process the aligned face with CodeFormer
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
                
            # Add the enhanced face to the helper
            self.face_helper.add_restored_face(blended_face)
        
        # Get the inverse affine transform
        self.face_helper.get_inverse_affine(None)
        
        # Paste the enhanced face back into the original image
        result_img = self.face_helper.paste_faces_to_input_image(upsample_img=original_img)
        
        # Double-check dimensions
        if result_img.shape[:2] != (original_height, original_width):
            print(f"Warning: Result dimensions changed from {original_width}x{original_height} to {result_img.shape[1]}x{result_img.shape[0]}")
            print("Resizing result to match original dimensions")
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
        
        result_img = self.process(input_path, fidelity_weight)
        
        # Save the result
        imwrite(result_img, output_path)
        print(f"Enhanced image saved to: {output_path}")
        
        return result_img

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
