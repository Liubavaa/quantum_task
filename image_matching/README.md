# Satellite Imagery Matching

## Project Overview

This project aims to detect environmental changes, specifically deforestation, in Ukraine by matching features between satellite images captured at different times of the year. By comparing images of the same location in different seasons, we can identify significant environmental changes.

## Solution Overview

Two feature-matching techniques were chosen for this task: ORB (Oriented FAST and Rotated BRIEF) and LoFTR (Local Feature TRansformer).

### ORB

ORB is a popular feature detection and description method, ideal for efficient and moderately accurate matching. ORB is computationally lightweight and works well for large images with moderate texture, like satellite imagery. It provides fast keypoint matching, which is useful for quickly identifying visible changes.

### LoFTR

LoFTR is a deep-learning-based method that uses transformers for dense feature matching, particularly effective in areas with few distinctive features. Satellite images often lack high texture and distinct features in natural areas, where traditional methods can struggle. LoFTR’s transformer-based approach captures subtle details, making it ideal for detecting changes in these low-texture regions. However, it requires more computational resources, so images are resized before processing.

### Solution Outline

1. **Data Loading**: Satellite images are read from `.jp2` files using `rasterio`, then converted for processing.
2. **ORB Matching**: ORB detects key points and matches features quickly, using a brute-force Hamming distance matcher to identify top matches. Matches are then filtered based on distance.
3. **LoFTR Matching**: Images are resized, normalized, and matched using LoFTR, providing dense and high-quality feature correspondences. A fundamental matrix filters outliers to focus on valid matches.
4. **Visualization**: Matched features between different image pairs are visualized to highlight matched areas.

## Requirements and Usage

1. Install dependencies from `requirements.txt`.
2. Explore ipynb (or check out [Kaggle Notebook](https://www.kaggle.com/code/libavaaa/satellite-imagery-matching)).
