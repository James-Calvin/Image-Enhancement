"""
Title: Assignment 2: Image Enhancement
Author: James-Calvin
Date: 2024-09-03
Description: CS4732 Machine Vision Assignment 2 submission
"""

import numpy as np
from PIL import Image


def log_transform(image, c=1):
  image_array = np.array(image, dtype=float)

  # Apply formula from assignment
  transformed = c * np.log(1 + image_array)

  # put on the range [0,255]
  transformed = np.uint8(255 * transformed / np.max(transformed))
  return Image.fromarray(transformed)

def power_transform(image, gamma=1, c=1):
  image_array = np.array(image, dtype=float)

  transformed = c * np.power(image_array, gamma)

  transformed = np.uint8(255 * transformed / np.max(transformed))

  return Image.fromarray(transformed)

if __name__ == "__main__":
  image_path = "fourierspectrum.pgm"
  image = Image.open(image_path)

  log_transform(image).save("log_transformed.jpg")
  gammas = [0.1, 0.2, 0.4, 0.67, 1, 1.5, 2.5, 5.0, 10.0]
  for gamma in gammas:
    power_transform(image, gamma=gamma).save(f"power_transform_𝛾({gamma}).jpg")
    
  