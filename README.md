# Computer Vision Projects

This repository contains a collection of computer vision projects that demonstrate key algorithms and techniques used in the field, including edge detection, image segmentation, and 3D reconstruction. The projects provide a practical overview of image processing, feature extraction, and machine learning applications in computer vision.

---

## Table of Contents

1. [Canny Edge Detection](#canny-edge-detection)
2. [Corners Detection & Image Stitching](#corners-detection--image-stitching)
3. [Image Segmentation](#image-segmentation)
4. [Sparse 3D Reconstruction](#sparse-3d-reconstruction)

---

## Canny Edge Detection

### Overview

This project demonstrates the implementation of the Canny Edge Detection algorithm, used to detect edges in images.

### Highlights

- **Method**: Applies Gaussian smoothing, gradient computation, non-maximum suppression, and edge tracing.
- **Result**: Successfully detects edges in a variety of test images.

### Files

- `CannyEdgeDetection.py`: Python script implementing the algorithm.

---

## Corners Detection & Image Stitching

### Overview

This project uses the Harris Corner Detection method to find key points and stitch images together seamlessly.

### Highlights

- **Method**: Harris Corner Detector, followed by image stitching techniques.
- **Result**: Demonstrates how to align images based on feature points for panorama creation.

### Files

- `ImageStitching.py`: Python script for corner detection and image stitching.

---

## Image Segmentation

### Overview

This project explores image segmentation techniques using K-means clustering and Efficient Graph-Based Image Segmentation.

### Highlights

- **Method**: K-means Clustering and Efficient Graph-Based Image Segmentation for image segmentation.
- **Result**: Segmented images into distinct regions for analysis.

### Files

- `ImageSegmentation.py`: Python script for performing image segmentation.

---

## Sparse 3D Reconstruction

### Overview

This project applies stereo vision techniques to reconstruct a 3D scene from two 2D images.

### Highlights

- **Method**: Feature matching and triangulation to recover depth information.
- **Result**: Generates a sparse 3D model from stereo image pairs.

### Files

- `Sparse3DReconstruction.py`: Python script for 3D scene reconstruction.

---

## Prerequisites

- Python 3.8 or higher
- Required Python libraries:
  - OpenCV
  - Numpy
  - Matplotlib
  
