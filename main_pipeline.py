# main_pipeline.py
# Integrated Deepfake + Microexpression Detection
# Author: Anushka

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

def run_deepfake_detection(input_path):
    print("\n[Step 1] Running Deepfake Detection...")

    try:
        model = tf.keras.models.load_model("deepfake_model.keras")
    except Exception as e:
        print(f"❌ Error loading deepfake model: {e}")
        return None

    if input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("❌ Failed to read video.")
            return None
        img = cv2.resize(frame, (224, 224))
    else:
        img = cv2.imread(input_path)
        if img is None:
            print("❌ Failed to read image.")
            return None
        img = cv2.resize(img, (224, 224))

    img = np.expand_dims(img / 255.0, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]

    if prediction < 0.5:
        print("Real input detected.")
        return "real"
    else:
        print("Deepfake detected — microexpression analysis skipped.")
        return "fake"

def run_microexpression_analysis():
    print("\n[Step 2] Running Microexpression Detection...")

    try:
        from micropart import fuzzy_integ
        result = fuzzy_integ.run_analysis()
        print(f"\n[Result] {result}")
    except Exception as e:
        print(f"❌ Error running microexpression analysis: {e}")

if __name__ == "__main__":
    print("===Deepfake + Microexpression Pipeline ===")

    input_path = input("Enter the path to your image or video: ").strip()

    if not os.path.exists(input_path):
        print("❌ File not found. Please check the path.")
        sys.exit(1)

    result = run_deepfake_detection(input_path)

    if result == "real":
        run_microexpression_analysis()

    print("\nProcess completed.")

