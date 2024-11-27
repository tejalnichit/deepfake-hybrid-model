# Deepfake Detection using Hybrid Model

## Overview

This project aims to develop a deepfake detection system using a hybrid Convolutional Neural Network (CNN) model. The model combines **MesoNet4** for lightweight deepfake detection and **ResNet50** for advanced feature extraction. The goal is to classify images and videos as either real or manipulated (deepfake).

## Dataset

The dataset used for this project includes:
- **UADFV Dataset**: A collection of deepfake videos.
- **Deepfake and Real Images**: A set of labeled images where some are deepfakes, and others are real.

## Project Overview

This project involves the following key steps:
1. **Frame extraction**: Extracts frames from video files.
2. **Model architecture**: Combines **MesoNet4** and **ResNet50** for improved deepfake detection.
3. **Data preprocessing**: Prepares images and frames for training using augmentation.
4. **Training**: The model is trained using combined datasets from **UADFV**, **Deepfake**, and **Real Images**.
5. **Evaluation**: Evaluates the model's performance on validation data.

## Requirements

To run this project, you need the following Python packages:

- `tensorflow`
- `keras`
- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-learn`
- `pandas`

You can install them using:

