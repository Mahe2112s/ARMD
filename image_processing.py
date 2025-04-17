import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage import feature
from pymft import MFT

# Image Preprocessing Class
class ImageProcessor:
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def load_image(self, image_path):
        """ Load an image from the path and convert it to grayscale. """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Error loading image: {image_path}")
        return img

    def preprocess_image(self, img):
        """ Apply transformations and return a tensor. """
        img = cv2.resize(img, self.img_size)
        return self.transform(img)

    def extract_features(self, img):
        """ Extract texture-based features using skimage. """
        features = feature.hog(img, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)
        return np.array(features)

    def apply_mft(self, img):
        """ Perform Multi-Fractal Transformation using pymft. """
        mft = MFT()
        mft.fit(img)
        return mft.get_features()

# Example Usage
if __name__ == "__main__":
    processor = ImageProcessor()
    sample_path = "path/to/sample_image.jpg"

    img = processor.load_image(sample_path)
    preprocessed_img = processor.preprocess_image(img)
    feature_vector = processor.extract_features(img)
    mft_features = processor.apply_mft(img)

    print("Feature Vector Shape:", feature_vector.shape)
    print("MFT Features Shape:", mft_features.shape)
