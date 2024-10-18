import cv2
import numpy as np
from utils import find_tci_image, preprocess_image, apply_clahe

def inference(model_path, image_path1, image_path2):
    # Load model (descriptors and keypoints)
    data = np.load(model_path, allow_pickle=True)
    descriptors1 = data['descriptors1']
    descriptors2 = data['descriptors2']
    keypoints1 = data['keypoints1']
    keypoints2 = data['keypoints2']
    
    # Convert keypoints from NumPy array to cv2.KeyPoint objects
    keypoints1 = [cv2.KeyPoint(x=kp[0], y=kp[1], _size=kp[2]) for kp in keypoints1]
    keypoints2 = [cv2.KeyPoint(x=kp[0], y=kp[1], _size=kp[2]) for kp in keypoints2]
    
    # Load and preprocess images
    image1 = preprocess_image(image_path1)
    image2 = preprocess_image(image_path2)
    
    # Feature matching
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Draw matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=2)
    matched_image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
    
    # Save or display the result
    cv2.imwrite('matched_image.jpg', matched_image)
    print("Inference completed. Matched image saved as 'matched_image.jpg'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Model Inference for Sentinel-2 Image Matching')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model file')
    parser.add_argument('--image1', type=str, required=True, help='Path to first satellite image folder')
    parser.add_argument('--image2', type=str, required=True, help='Path to second satellite image folder')
    args = parser.parse_args()
    
    # Find TCI images
    tci_image_path1 = find_tci_image(args.image1)
    tci_image_path2 = find_tci_image(args.image2)
    
    inference(args.model, tci_image_path1, tci_image_path2)
