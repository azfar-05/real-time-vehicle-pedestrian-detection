import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video file
cap = cv2.VideoCapture("videos/traffic.mp4")

# Safety check
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(
    'outputs/output.avi',
    fourcc,
    20.0,
    (frame_width, frame_height)
)

# Target classes
target_classes = ["person", "car", "bus", "truck", "motorcycle"]

# FPS calculation
prev_time = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Finished processing video.")
        break

    # Run YOLO inference
    results = model(frame)

    # Counters
    people_count = 0
    vehicle_count = 0

    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Class ID
            cls_id = int(box.cls[0])

            # Class name
            class_name = model.names[cls_id]

            # Filter unwanted classes
            if class_name not in target_classes:
                continue

            # Confidence score
            confidence = float(box.conf[0])

            # Confidence threshold
            if confidence < 0.6:
                continue

            # Count objects + set colors
            if class_name == "person":
                people_count += 1
                color = (0, 255, 0)   # Green
            else:
                vehicle_count += 1
                color = (255, 0, 0)   # Blue

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Label text
            label = f"{class_name} {confidence:.2f}"

            # Draw label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # FPS calculation
    current_time = time.time()

    fps = 1 / max(current_time - prev_time, 0.0001)

    prev_time = current_time

    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Display people count
    cv2.putText(
        frame,
        f"People: {people_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display vehicle count
    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Save processed frame
    out.write(frame)

    # Show output
    cv2.imshow(
        "Real-Time Vehicle and Pedestrian Detection",
        frame
    )

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()