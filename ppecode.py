from ultralytics import YOLO
import cv2
import cvzone
import math
import os

# Load YOLO model from the uploaded file
model_path = "D:/ppe final/myenv/ppe.pt"

if not os.path.exists(model_path):
    print("Error: Model file not found!")
    exit()

model = YOLO(model_path)

# Open webcam
cap = cv2.VideoCapture(0)  # Change to 1 if multiple cameras are connected
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Class names
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

while True:
    success, img = cap.read()

    # 🛑 Ensure frame is captured properly
    if not success or img is None or img.size == 0:
        print("Error: Failed to capture image from webcam")
        continue  # Skip this frame

    results = model(img, stream=True)

    for r in results:
        for box in r.boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = round(float(box.conf[0]), 2)

            # Class Name (Ensure index is valid)
            cls = int(box.cls[0])
            if 0 <= cls < len(classNames):
                currentClass = classNames[cls]
            else:
                print(f"Warning: Invalid class index {cls}. Skipping...")
                continue

            # Set color based on detected class
            if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                myColor = (0, 0, 255)  # Red for safety violations
            elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                myColor = (0, 255, 0)  # Green for compliance
            else:
                myColor = (255, 0, 0)  # Blue for other detections

            # Draw Bounding Box and Label
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=myColor)
            cvzone.putTextRect(img, f'{currentClass} {conf}', 
                               (max(0, x1), max(35, y1)), scale=1, thickness=1, 
                               colorB=myColor, colorT=(255, 255, 255), 
                               colorR=myColor, offset=5)

    # 🛑 Ensure img is valid before showing
    if img is not None and img.size > 0:
        cv2.imshow("YOLO PPE Detection", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
