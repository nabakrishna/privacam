from ultralytics import YOLO

# This automatically downloads the weights for you
model = YOLO("yolo26s-seg.pt")