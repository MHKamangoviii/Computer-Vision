# Payment Slip Splitter

## Computer Vision – Quiz 1

This project implements a traditional computer vision solution to automatically split multiple payment slips (receipts) from a single image using OpenCV.

## Objective

Detect and extract individual payment slips from an image containing multiple receipts.

## Methods

Two approaches are implemented and compared:

### 1. Traditional Contour-Based Method

Grayscale conversion

Gaussian blur

Adaptive thresholding

Morphological operations

Contour detection and filtering by area and aspect ratio

### 2. Enhanced Edge & Text-Based Method

Canny edge detection

Morphological dilation

Gradient-based text region detection

Merging overlapping bounding boxes

## Technologies

Python

OpenCV

NumPy

## Project Structure
├── input.png
├── receipt_splitter.py
├── output_traditional/
│   └── annotated_original.png
├── output_enhanced/
│   └── annotated_original.png
└── README.md

## How to Run
pip install opencv-python numpy
python receipt_splitter.py


Ensure input.png is present in the same directory.

## Output

Extracted receipts saved as receipt_1.png, receipt_2.png, etc.

Annotated image showing detected regions

Automatic comparison of both methods

Course Details

Course: Computer Vision

Assessment: Quiz 1

Approach: Traditional Image Processing (No Deep Learning)

## Author

Muhammad Hussain
