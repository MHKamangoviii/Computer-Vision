# Dollar Bill Classification

## Computer Vision – Quiz 2

This project implements an automated image classification system to detect and identify U.S. dollar bill denominations ($1, $5, $10, $20) using a Deep Learning approach.

## Objective

Classify images of dollar bills into their correct denominations using a Convolutional Neural Network (CNN).

## Approach

Dataset of 130 labeled images

Image preprocessing and normalization

Custom CNN architecture designed and trained from scratch

Performance evaluated using classification accuracy

After experimenting with different architectures, a custom CNN provided the best balance between accuracy and simplicity.

## Model Performance

Final Accuracy: 87.10%

Evaluated on the validation/test set

## Technologies Used

Python

TensorFlow / Keras

NumPy

OpenCV

Project Structure
├── dataset/
│   ├── 1/
│   ├── 5/
│   ├── 10/
│   └── 20/
├── Quiz_2.py
└── README.md

## How to Run
pip install tensorflow opencv-python numpy
python train.py

## Output

Trained CNN model

Predicted denomination for input images

Printed accuracy and evaluation metrics

## Course Details

Course: Computer Vision

Assessment: Quiz 2


## Author

Muhammad Hussain 2022-SE-41
