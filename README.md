# Number Plate Detection System

This project is a Python-based Automatic Number Plate Recognition (ALPR) system specifically optimized for low-light conditions, such as nighttime traffic surveillance.

## Features

- **Low-Light Enhancement**: Utilizes Gamma Correction and Contrast Limited Adaptive Histogram Equalization (CLAHE) to brighten dark images and improve local contrast.
- **Plate Localization**: Uses computer vision techniques (Bilateral filtering, Canny edges, and contour detection) to identify potential number plate regions.
- **OCR Recognition**: Integrates [EasyOCR](https://github.com/JaidedAI/EasyOCR) for robust character recognition in natural scenes.
- **Modular Architecture**: Separate modules for image enhancement, detection, and orchestration.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Numberplatedetection
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `easyocr` will download its pre-trained models on the first run.*

## Usage

### Running the detection

To process an image, run the `main.py` script with the path to your image file:

```bash
python main.py path/to/your/image.jpg
```

### Running the test suite

To verify the system is working correctly, you can run the provided test script. This script generates a simulated low-light image and runs it through the pipeline:

```bash
python test_detection.py
```

## How it Works

1.  **Enhancement (`enhancement.py`)**: 
    - Applies **Gamma Correction** to non-linearly brighten the dark pixels.
    - Applies **CLAHE** in the LAB color space to the Luminance (L) channel, enhancing the edges of characters without over-amplifying noise.
2.  **Detection (`detector.py`)**: 
    - Reduces noise using **Bilateral Filtering**.
    - Finds edges using **Canny Edge Detection**.
    - Detects **Contours** and filters for rectangular shapes with specific aspect ratios common for number plates.
3.  **OCR (`main.py`)**: 
    - Crops the detected plate regions and passes them to **EasyOCR** for text extraction.

## Project Structure

- `main.py`: Entry point for the application.
- `enhancement.py`: Contains low-light image processing logic.
- `detector.py`: Handles plate localization and preprocessing.
- `test_detection.py`: Automated test script.
- `requirements.txt`: List of Python dependencies.
- `.gitignore`: Configured to exclude temporary artifacts and binary files.
