import cv2
import numpy as np

def detect_plate(image):
    """
    Detect potential number plate regions in an image using contour detection.
    Returns a list of cropped images containing potential plates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bilateral filter to remove noise while keeping edges sharp
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # Canny Edge Detection
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(keypoints[0], key=cv2.contourArea, reverse=True)[:10]

    potential_plates = []

    for contour in contours:
        # Approximate the contour based on its perimeter
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Number plates are typically rectangular (4 sides)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            # Filter by aspect ratio (typical plates are wider than they are tall)
            aspect_ratio = float(w) / h
            if 2.0 < aspect_ratio < 6.0:
                plate_crop = image[y:y+h, x:x+w]
                potential_plates.append(plate_crop)

    return potential_plates

def preprocess_for_ocr(plate_img):
    """
    Preprocess the cropped plate image for better OCR results.
    """
    if plate_img is None or plate_img.size == 0:
        return None

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Resizing to standard height
    height, width = gray.shape
    new_height = 100
    new_width = int((new_height / height) * width)
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Thresholding
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh
