import cv2
import os
import shutil
from ultralytics import YOLO

# Load YOLOv8 model (pretrained on COCO dataset which includes "person")
model = YOLO("yolov8n.pt")  # use 'n' (nano) for speed

# Input and output folders
input_folder = "photos"
output_folders = ["blur", "faceless", "excellent"]

# Create output folders if not exist
for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

def is_blurry(image, threshold=100.0):
    """Return True if image is blurry"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness_score < threshold

# Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        filepath = os.path.join(input_folder, filename)
        image = cv2.imread(filepath)

        # 1. Blur check
        if is_blurry(image):
            shutil.move(filepath, os.path.join("blur", filename))
            continue

        # 2. Face detection with YOLOv8
        results = model(image)
        face_detected = False

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # COCO class 0 = "person"
                    face_detected = True
                    break

        if face_detected:
            shutil.move(filepath, os.path.join("excellent", filename))
        else:
            shutil.move(filepath, os.path.join("faceless", filename))

print("✅ Sorting    complete. Check 'blur', 'faceless', 'excellent' folders.")
