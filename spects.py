from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model (for object detection)
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Open webcam (0 for default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        break  # Stop if no frame is captured

    # Run YOLO detection on the frame
    results = model(frame)

    spectacles_detected = False  # Flag to check if spectacles are detected

    # Process detected objects
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]  # Get object label

            # If the detected object is glasses/spectacles
            if label in ["glasses", "sunglasses"]:
                spectacles_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
                cv2.putText(frame, "Spectacles Detected!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # If spectacles are detected, show a warning message
    if spectacles_detected:
        cv2.putText(frame, "REMOVE SPECTACLES WHILE DRIVING!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the output
    cv2.imshow("Live Spectacles Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
