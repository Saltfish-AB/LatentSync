from typing import Optional
import torch
import tqdm

from ..utils.image_processor import ImageProcessor
from ..utils.util import read_video

def affine_transform_video(image_processor, video_path):
    video_frames = read_video(video_path, use_decord=False)
    faces = []
    boxes = []
    affine_matrices = []
    print(f"Affine transforming {len(video_frames)} faces...")
    for frame in tqdm.tqdm(video_frames):
        face, box, affine_matrix = image_processor.affine_transform(frame)
        faces.append(face)
        boxes.append(box)
        affine_matrices.append(affine_matrix)

    faces = torch.stack(faces)
    return faces, video_frames, boxes, affine_matrices

def generate_affine_transforms(video_path, output_path, height: int = 512, mask: str = "fix_mask"):
    image_processor = ImageProcessor(height, mask=mask, device="cuda")

    faces, video_frames, boxes, affine_matrices = affine_transform_video(
        image_processor=image_processor,
        video_path=video_path)

    data_to_save = {
        "faces": faces,
        "boxes": boxes,
        "affine_matrices": affine_matrices
    }
    torch.save(data_to_save, output_path)
