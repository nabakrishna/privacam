# privacam

A real-time privacy masking system that uses instance segmentation to detect and blur sensitive objects (like phones, documents, or bottles) on a live camera feed. It includes a custom retro green-phosphor surveillance HUD. 

## How it works

1. Grabs your webcam feed using a background thread (to prevent IO bottlenecks).
2. Runs YOLOv8s-seg to find specific object classes.
3. Generates a combined pixel-perfect mask of those objects.
4. Applies a Gaussian blur exclusively to the masked areas, leaving the rest of the room visible.

## Setup

Make sure you have Python installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU, it is highly recommended to install the CUDA version of PyTorch directly from their website before running the above command. It will automatically detect CUDA and use FP16 half-precision to max out your framerate.

## Usage

Start the app by running the main file:

```bash
py -3.12 main.py (use py 3.12 version for better compatablity)
```

### Key-bindings

* **Q** — Quit the application
* **S** — Save a screenshot to the current directory
* **+ / =** — Increase the blur intensity 
* **-** — Decrease the blur intensity 

## Project Structure

The code is split into specific modules so it's easy to tweak:

* `config.py` — Edit this to change camera resolution, blur strength, HUD colors, or the YOLO class IDs you want to hide.
* `camera.py` — Handles the multi-threaded video stream.
* `vision.py` — Manages PyTorch, the YOLOv8 model, and the image processing math.
* `ui.py` — Renders the CRT scanlines, corner brackets, and detection labels.
* `main.py` — The core loop that ties it all together.

## Customization Notes

By default, the system targets COCO classes for Bottles (39), Cell Phones (67), and Books (73 - used as a proxy for documents/paper). If you want to detect different items, just update the `SENSITIVE_CLASS_IDS` in `config.py`. 
