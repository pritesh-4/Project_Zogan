from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model("elephant.jpg")

for result in results:
    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = result.names[class_id]

        if class_name == "elephant" and confidence > 0.70:
            print("🚨 ELEPHANT DETECTED!")