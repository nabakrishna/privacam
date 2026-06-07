from ultralytics import YOLO
model = YOLO("yolov8s-seg.pt") 
results = model.predict(source="test_debit_card.jpg", conf=0.25)

for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        confidence = box.conf[0].item()
        #map model ids etc
        print(f"Detected Class ID: {cls_id} with {confidence:.2f} confidence")