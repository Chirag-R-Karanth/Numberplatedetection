import cv2
import numpy as np

def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel back with a and b channels
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def adjust_gamma(image, gamma=1.5):
    """
    Build a lookup table mapping the pixel values [0, 255] to
    their adjusted gamma values.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def enhance_low_light(image):
    """
    Combined enhancement for low light images.
    """
    # Step 1: Gamma correction to brighten the image
    brightened = adjust_gamma(image, gamma=1.5)

    # Step 2: CLAHE to improve contrast
    enhanced = apply_clahe(brightened)

    return enhanced
