import os
import cv2
import glob
import numpy as np
from utils import find_tci_image, preprocess_image, apply_clahe

def train_model(image_path1, image_path2, output_model_path):
    # Load and preprocess images
    image1 = preprocess_image(image_path1)
    image2 = preprocess_image(image_path2)
    
    # Convert to grayscale and apply CLAHE
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray1 = apply_clahe(gray1)
    gray2 = apply_clahe(gray2)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    # Save descriptors and keypoints if needed
    np.savez(output_model_path, descriptors1=descriptors1, descriptors2=descriptors2,
             keypoints1=keypoints1, keypoints2=keypoints2)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Model Training for Sentinel-2 Image Matching')
    parser.add_argument('--image1', type=str, required=True, help='Path to first satellite image folder')
    parser.add_argument('--image2', type=str, required=True, help='Path to second satellite image folder')
    parser.add_argument('--output', type=str, default='model.npz', help='Output model file path')
    args = parser.parse_args()
    
    # Find TCI images
    tci_image_path1 = find_tci_image(args.image1)
    tci_image_path2 = find_tci_image(args.image2)
    
    train_model(tci_image_path1, tci_image_path2, args.output)
