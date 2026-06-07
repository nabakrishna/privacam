import cv2
import numpy as np
import torch
from ultralytics import YOLO
from config import SENSITIVE_CLASS_IDS, DETECTION_CONFIDENCE

def resolve_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(device):
    model = YOLO("yolov8s-seg.pt")
    model.to(device)
    return model

def build_combined_mask(result, height, width):
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    detections_count = 0
    
    if result.masks is not None and result.boxes is not None:
        for idx, mask_tensor in enumerate(result.masks.data):
            cls_id = int(result.boxes.cls[idx].item())
            conf = float(result.boxes.conf[idx].item())
            
            # The critical check to prevent blurring people/wrong classes
            if cls_id in SENSITIVE_CLASS_IDS and conf >= DETECTION_CONFIDENCE:
                detections_count += 1
                mask_np = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.bitwise_or(combined_mask, (mask_resized * 255).astype(np.uint8))
                
    return combined_mask, detections_count

def apply_privacy_blur(frame, combined_mask, kernel_tuple, sigma):
    output_frame = frame.copy()
    
    if np.any(combined_mask > 0):
        kx, ky = kernel_tuple
        kx = kx if kx % 2 != 0 else kx + 1
        ky = ky if ky % 2 != 0 else ky + 1
        
        blurred = cv2.GaussianBlur(frame, (kx, ky), sigma)
        mask_bool = combined_mask > 0
        output_frame[mask_bool] = blurred[mask_bool]
        
    return output_frame