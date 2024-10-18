
# README.md

# Sentinel-2 Satellite Image Matching using Classical Machine Learning

## Project Description

This project involves developing an algorithm to detect and match keypoints between Sentinel-2 satellite images captured in different seasons using classical machine learning techniques. The main challenge is to handle variations in satellite imagery caused by seasonal changes, such as differences in vegetation, snow cover, and atmospheric conditions.

## Dataset Structure

Each folder in the dataset represents a single Sentinel-2 satellite image. The RGB image, known as the True Color Image (TCI), includes "TCI" in its name and is located in the following path:

`S2A_MSIL1C_<DATE_TIME>.SAFE/GRANULE/L1C_<TILE_ID>_<DATE_TIME>/IMG_DATA/*TCI*.jp2`

The other files in the dataset are the spectral bands from the satellite.

## Prerequisites

- Python 3.6 or higher
- OpenCV with contrib modules
- NumPy
- Matplotlib
- Jupyter Notebook

## Installation

1. **Clone the Repository**

   Clone the project repository to your local machine.

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment for the project to manage dependencies.

3. **Install Dependencies**

   Install all required libraries using the command:

   ```
   pip install -r requirements.txt
   ```

## Dataset Preparation

1. **Download Sentinel-2 Images**

   - Visit the Copernicus Open Access Hub.
   - Search for Sentinel-2 images covering your area of interest.
   - Download images from different seasons ensuring they cover the same geographic area.

2. **Organize the Dataset**

   Place the satellite image folders into a structured directory. For example:

   ```
   data/
   ├── season_winter/
   │   └── S2A_MSIL1C_<WINTER_DATE>.SAFE/
   └── season_summer/
       └── S2A_MSIL1C_<SUMMER_DATE>.SAFE/
   ```

## Running the Code

### Model Training / Algorithm Creation

Run the `model_training.py` script to perform feature detection and matching. This script processes the images, detects keypoints, and computes descriptors.

### Model Inference

Use the `model_inference.py` script to apply the algorithm to new images. This script loads the saved descriptors and keypoints, performs feature matching, and visualizes the results.

### Demo Notebook

For an interactive demonstration, open the `demo.ipynb` Jupyter notebook. It provides a step-by-step guide through the process, including visualization of keypoints and matches.

## Solution Explanation

The algorithm follows these main steps:

1. **Loading Images**

   Finds and loads the TCI images from the specified satellite image folders.

2. **Preprocessing**

   - Normalizes images if they are in 16-bit format.
   - Resizes images to manage computational load.
   - Converts images to grayscale.
   - Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance local contrast.

3. **Feature Detection and Description**

   Uses the Scale-Invariant Feature Transform (SIFT) algorithm to detect keypoints and compute descriptors.

4. **Feature Matching**

   Employs a Brute-Force matcher with k-nearest neighbors and applies Lowe's ratio test to filter out false matches.

5. **Outlier Rejection**

   Utilizes RANSAC (Random Sample Consensus) to compute a homography matrix and reject outlier matches.

6. **Visualization**

   Draws the matched keypoints on the images for visual inspection.

## Project Structure

- `README.md`: Project description and setup instructions.
- `requirements.txt`: List of required Python libraries.
- `model_training.py`: Python script for model training and algorithm creation.
- `model_inference.py`: Python script for model inference.
- `demo-season-imagematching.ipynb`: Jupyter notebook demonstrating the algorithm with visualization.
- `utils.py`: Contains helper functions used across scripts.


## Dependencies

Refer to `requirements.txt` for all required Python libraries.

## Setting Up the Project

1. **Ensure Prerequisites are Installed**

   - Python 3.6 or higher
   - pip package manager

2. **Install OpenCV with Contrib Modules**

   OpenCV's contrib modules are necessary for using the SIFT algorithm.

3. **Install Other Dependencies**

   Install NumPy, Matplotlib, and Jupyter Notebook as listed in the `requirements.txt`.

4. **Verify Installation**

   Run a simple script to verify that OpenCV and other libraries are installed correctly.

## License

This project is licensed under the MIT License.

## Contact

For any questions or issues, please contact the project maintainer.
