import cv2
import time
from config import *
import numpy as np

def _corner_brackets(img, color):
    h, w = img.shape[:2]
    arm = 22
    pad = 9
    anchors = [(pad, pad), (w - pad, pad), (pad, h - pad), (w - pad, h - pad)]
    for ax, ay in anchors:
        dx = 1 if ax < w // 2 else -1
        dy = 1 if ay < h // 2 else -1
        cv2.line(img, (ax, ay), (ax + dx * arm, ay), color, 2, cv2.LINE_AA)
        cv2.line(img, (ax, ay), (ax, ay + dy * arm), color, 2, cv2.LINE_AA)

def _scanlines(img, strength=0.06):
    darkened = img.copy()
    darkened[1::2] = (darkened[1::2] * (1.0 - strength)).astype(np.uint8)
    cv2.addWeighted(darkened, strength, img, 1.0 - strength, 0, img)

def draw_detection_labels(frame, result):
    if result.boxes is None:
        return 0
    count = 0
    for idx, cls_tensor in enumerate(result.boxes.cls):
        cls_id = int(cls_tensor.item())
        conf = float(result.boxes.conf[idx].item())
        if cls_id not in SENSITIVE_CLASS_IDS or conf < DETECTION_CONFIDENCE:
            continue
        count += 1
        x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int)
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_ALERT, 1, cv2.LINE_AA)
        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SM, FONT_W)
        badge_y1 = max(y1 - th - baseline - 6, 0)
        cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, y1), C_ALERT, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - baseline - 2), FONT, FONT_SM, C_WHITE, FONT_W, cv2.LINE_AA)
    return count

def draw_hud(frame, fps, detections, device, frame_idx, kernel_size):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (6, 6), (w - 6, h - 6), C_DIM, 1)
    cv2.line(frame, (6, 30), (w - 6, 30), C_DIM, 1)
    cv2.line(frame, (6, h - 28), (w - 6, h - 28), C_DIM, 1)
    _corner_brackets(frame, C_PRIMARY)
    cv2.putText(frame, "PRIVACAM  v1.0", (16, 22), FONT, FONT_LG, C_PRIMARY, FONT_W, cv2.LINE_AA)
    ts = time.strftime("%Y-%m-%d    %H:%M:%S")
    (ts_w, _), _ = cv2.getTextSize(ts, FONT, FONT_SM, FONT_W)
    cv2.putText(frame, ts, ((w - ts_w) // 2, 22), FONT, FONT_SM, C_SECONDARY, FONT_W, cv2.LINE_AA)
    fps_label = f"FPS  {fps:05.1f}"
    (fl_w, _), _ = cv2.getTextSize(fps_label, FONT, FONT_SM, FONT_W)
    cv2.putText(frame, fps_label, (w - fl_w - 14, 22), FONT, FONT_SM, C_SECONDARY, FONT_W, cv2.LINE_AA)
    if detections > 0 and int(time.time() * 2) % 2 == 0:
        banner = "[ PRIVACY SHIELD ACTIVE ]"
        (bw, _), _ = cv2.getTextSize(banner, FONT, FONT_LG, FONT_W + 1)
        cv2.putText(frame, banner, ((w - bw) // 2, 56), FONT, FONT_LG, C_ALERT, FONT_W + 1, cv2.LINE_AA)
    footer_y = h - 10
    dev_label = f"DEV:{device.upper()}"
    cv2.putText(frame, dev_label, (16, footer_y), FONT, FONT_SM, C_DIM, FONT_W, cv2.LINE_AA)
    blur_label = f"BLUR  {kernel_size}x{kernel_size}"
    (dw, _), _ = cv2.getTextSize(dev_label, FONT, FONT_SM, FONT_W)
    cv2.putText(frame, blur_label, (16 + dw + 16, footer_y), FONT, FONT_SM, C_DIM, FONT_W, cv2.LINE_AA)
    det_color = C_ALERT if detections > 0 else C_SECONDARY
    det_label = f"MASKED  {detections:02d}"
    (det_w, _), _ = cv2.getTextSize(det_label, FONT, FONT_SM, FONT_W)
    cv2.putText(frame, det_label, ((w - det_w) // 2, footer_y), FONT, FONT_SM, det_color, FONT_W, cv2.LINE_AA)
    fc_label = f"FRAME  #{frame_idx:07d}"
    (fc_w, _), _ = cv2.getTextSize(fc_label, FONT, FONT_SM, FONT_W)
    cv2.putText(frame, fc_label, (w - fc_w - 14, footer_y), FONT, FONT_SM, C_DIM, FONT_W, cv2.LINE_AA)
    _scanlines(frame)