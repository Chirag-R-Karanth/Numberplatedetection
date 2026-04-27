import cv2
import sys
import easyocr
from enhancement import enhance_low_light
from detector import detect_plate, preprocess_for_ocr

def process_image(image_path, reader):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return

    print("Step 1: Enhancing low-light image...")
    enhanced_img = enhance_low_light(img)

    # Save enhanced image for debugging
    cv2.imwrite("enhanced_debug.jpg", enhanced_img)

    print("Step 2: Detecting potential number plates...")
    potential_plates = detect_plate(enhanced_img)

    if not potential_plates:
        # Try detection on original image if enhancement failed to find any
        print("No plates found in enhanced image, trying original image...")
        potential_plates = detect_plate(img)

    if not potential_plates:
        print("No potential number plates detected.")
        return

    print(f"Found {len(potential_plates)} potential regions. Running OCR...")

    results = []
    for i, plate in enumerate(potential_plates):
        preprocessed_plate = preprocess_for_ocr(plate)
        if preprocessed_plate is not None:
            # EasyOCR can take numpy arrays directly
            ocr_result = reader.readtext(preprocessed_plate)
            for (bbox, text, prob) in ocr_result:
                if prob > 0.3: # Confidence threshold
                    results.append((text, prob))
                    print(f"Plate {i+1} detected text: {text} (Confidence: {prob:.2f})")

    if not results:
        print("OCR could not recognize any text in the detected regions.")
    else:
        print("\n--- Final Results ---")
        for text, prob in results:
            print(f"Detected Plate: {text} | Confidence: {prob:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_image>")
    else:
        # Initialize EasyOCR reader once
        print("Initializing OCR reader...")
        reader = easyocr.Reader(['en'])
        process_image(sys.argv[1], reader)
