import cv2
import os
import shutil
import argparse
from ultralytics import YOLO

def is_blurry(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness_score < threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="photos")
    parser.add_argument("--threshold", type=float, default=100.0)
    args = parser.parse_args()

    input_folder = args.input
    output_folders = ["blur", "faceless", "excellent"]

    if not os.path.exists(input_folder):
        print(f"Error: Directory '{input_folder}' not found.")
        return

    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)

    model = YOLO("yolov8n.pt")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath)

            if image is None:
                print(f"Warning: Could not read {filename}, skipping.")
                continue

            if is_blurry(image, args.threshold):
                shutil.move(filepath, os.path.join("blur", filename))
                continue

            results = model(image)
            face_detected = False

            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:
                        face_detected = True
                        break
                if face_detected:
                    break

            if face_detected:
                shutil.move(filepath, os.path.join("excellent", filename))
            else:
                shutil.move(filepath, os.path.join("faceless", filename))

    print("Sorting complete.")

if __name__ == "__main__":
    main()
