import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------------------
# 🔧 CONFIGURATION
# ----------------------------------------
VIDEO_PATH = "dog_cat_child.mp4"              # Input video file
OUTPUT_PATH = "output.mp4"            # Output video file (with boxes)
YOLO_MODEL = "yolov8m.pt"             # More accurate than yolov8n
SELECTED_CLASSES = ['person', 'dog', 'cat']  # Choose classes to detect
CONF_THRESHOLD = 0.25

model = YOLO(YOLO_MODEL)
COCO_CLASSES = model.names


cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("❌ Cannot read video. Exiting.")
    cap.release()
    exit()

h, w = frame.shape[:2]
frame_center = np.array([w / 2, h / 2])

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (w, h))

print(f"🚀 Starting detection on {VIDEO_PATH}...")
print(f"📦 Saving to {OUTPUT_PATH}")
print(f"📏 Frame size: {w}x{h}\n")

def get_relative_position(box_center, frame_center):
    x_rel = 2 * (box_center[0] - frame_center[0]) / frame_center[0]
    y_rel = 2 * (box_center[1] - frame_center[1]) / frame_center[1]
    return round(x_rel, 2), round(y_rel, 2)

frame_idx = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF_THRESHOLD)[0]
        annotated_frame = frame.copy()

        for i, det in enumerate(results.boxes):
            cls_id = int(det.cls[0])
            cls_name = COCO_CLASSES[cls_id]
            conf = float(det.conf[0])

            if cls_name not in SELECTED_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, det.xyxy[0])
            box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            rel_x, rel_y = get_relative_position(box_center, frame_center)

            label = f"{cls_name} {conf:.2f} [{rel_x}, {rel_y}]"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            print(f"[Frame {frame_idx}] {label}")

        # Show streaming video
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # Save to output video
        out_writer.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Interrupted by user.")
            break

        frame_idx += 1

except KeyboardInterrupt:
    print("🛑 Keyboard Interrupt. Exiting.")

finally:
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print("✅ Video processing complete.")
