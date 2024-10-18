


## This file contains helper functions used in multiple scripts.



import os
import cv2
import glob
import numpy as np

def find_tci_image(satellite_folder):
    """
    Find the TCI image within the given satellite folder.
    """
    granule_folder = os.path.join(satellite_folder, 'GRANULE')
    granule_subfolders = [os.path.join(granule_folder, d) for d in os.listdir(granule_folder) if os.path.isdir(os.path.join(granule_folder, d))]
    if len(granule_subfolders) == 0:
        print(f"No granule folder found in {granule_folder}")
        return None
    # Assuming only one granule folder
    granule_subfolder = granule_subfolders[0]
    img_data_folder = os.path.join(granule_subfolder, 'IMG_DATA')
    # Search for files with 'TCI' in their name
    tci_files = glob.glob(os.path.join(img_data_folder, '*TCI*.jp2'))
    if len(tci_files) == 0:
        print(f"No TCI image found in {img_data_folder}")
        return None
    # Assuming only one TCI image
    tci_image_path = tci_files[0]
    return tci_image_path

def preprocess_image(image_path):
    """
    Load and preprocess the image.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    # Normalize 16-bit images to 8-bit
    if image.dtype == np.uint16:
        image = (image / 256).astype('uint8')
    elif image.dtype != np.uint8:
        print(f"Unexpected image dtype: {image.dtype}")
    return image

def apply_clahe(image_gray):
    """
    Apply CLAHE to improve local contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_clahe = clahe.apply(image_gray)
    return image_clahe
