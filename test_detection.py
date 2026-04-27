import cv2
import numpy as np
import os
from enhancement import enhance_low_light
from detector import detect_plate

def create_dummy_image():
    """Creates a dark image with a simulated number plate for testing."""
    # Create a dark background
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img = img + 30 # dark gray

    # Add a "plate" - light rectangle (simulating retroreflective plate in low light)
    plate_x, plate_y, plate_w, plate_h = 300, 400, 200, 50
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (200, 200, 200), -1)

    # Add some "text" to the plate
    cv2.putText(img, "ABC 123", (plate_x + 10, plate_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    cv2.imwrite("test_low_light.jpg", img)
    print("Created test_low_light.jpg")

def test_pipeline():
    create_dummy_image()

    img = cv2.imread("test_low_light.jpg")
    if img is None:
        print("Failed to load test image.")
        return

    print("Testing enhancement...")
    enhanced = enhance_low_light(img)
    cv2.imwrite("test_enhanced.jpg", enhanced)
    print("Saved test_enhanced.jpg")

    print("Testing detection...")
    plates = detect_plate(enhanced)
    print(f"Detected {len(plates)} potential plates.")

    if len(plates) > 0:
        print("Success: At least one plate detected.")
    else:
        print("Failure: No plates detected.")

    # Cleanup (commented out for debugging if needed)
    # if os.path.exists("test_low_light.jpg"):
    #     os.remove("test_low_light.jpg")
    # if os.path.exists("test_enhanced.jpg"):
    #     os.remove("test_enhanced.jpg")

if __name__ == "__main__":
    test_pipeline()
