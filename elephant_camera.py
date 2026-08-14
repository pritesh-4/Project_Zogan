import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    results = model(frame)

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = result.names[class_id]

            if class_name == "elephant" and confidence > 0.70:
                print("🚨 ELEPHANT DETECTED!")

    annotated_frame = results[0].plot()

    cv2.imshow("Elephant Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()