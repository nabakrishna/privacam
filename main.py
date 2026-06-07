import cv2
import time
from pathlib import Path
from collections import deque
from config import *
from camera import CameraStream
from vision import resolve_device, load_model, build_combined_mask, apply_privacy_blur
from ui import draw_detection_labels, draw_hud

def run():
    device = resolve_device()
    model = load_model(device)
    cam = CameraStream().start()
    win_title = "PrivaCam  -  [Q] Quit    [S] Screenshot    [+/-] Blur Intensity"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, CAPTURE_WIDTH, CAPTURE_HEIGHT)
    frame_times = deque(maxlen=FPS_WINDOW)
    fps = 0.0
    frame_idx = 0
    blur_kernel = BLUR_KERNEL_SIZE.copy()

    while True:
        t_start = time.perf_counter()
        ret, frame = cam.read()
        if not ret or frame is None:
            continue
        frame_idx += 1
        results = model.predict(
            frame,
            imgsz=INFERENCE_IMGSZ,
            conf=DETECTION_CONFIDENCE,
            device=device,
            retina_masks=True,
            verbose=False,
            half=(device == "cuda")
        )
        result = results[0]
        # mask = build_combined_mask(result, CAPTURE_HEIGHT, CAPTURE_WIDTH)
        height, width = frame.shape[:2]
        mask, n_detections = build_combined_mask(result, height, width)
        output_frame = apply_privacy_blur(frame, mask, (blur_kernel[0], blur_kernel[1]), BLUR_SIGMA)
        # output_frame = apply_privacy_blur(frame, mask, (blur_kernel[0], blur_kernel[1]), BLUR_SIGMA)
        n_detections = draw_detection_labels(output_frame, result)
        draw_hud(output_frame, fps, n_detections, device, frame_idx, blur_kernel[0])
        cv2.imshow(win_title, output_frame)
        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if frame_times:
            fps = len(frame_times) / sum(frame_times)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            outpath = Path(f"privacam_{ts_str}.png")
            cv2.imwrite(str(outpath), output_frame)
        elif key in (ord("+"), ord("=")):
            new_size = min(blur_kernel[0] + BLUR_KERNEL_STEP, 201)
            blur_kernel = [new_size, new_size]
        elif key == ord("-"):
            new_size = max(blur_kernel[0] - BLUR_KERNEL_STEP, 11)
            blur_kernel = [new_size, new_size]

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()