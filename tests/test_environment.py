import mediapipe as mp
import cv2
import numpy as np

def test_imports():
    print("MediaPipe version:", mp.__version__)
    print("OpenCV version:", cv2.__version__)
    print("NumPy version:", np.__version__)
    assert mp.__version__ == "0.10.14", "MediaPipe version mismatch"
    assert cv2.__version__ == "4.11.0", "OpenCV version mismatch"
    assert np.__version__ == "1.26.4", "NumPy version mismatch"
    print("All dependencies imported successfully!")

if __name__ == "__main__":
    test_imports()
