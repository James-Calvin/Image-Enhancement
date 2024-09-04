"""
Title: Assignment 2: Image Enhancement
Author: James-Calvin
Date: 2024-09-03
Description: CS4732 Machine Vision Assignment 2 submission
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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

def histogram_equalization(image):
  # References
  # [1] https://towardsdatascience.com/histogram-equalization-5d1013626e64
  # [2] https://en.wikipedia.org/wiki/Histogram_equalization
  # [3] https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
  # [4] https://stackoverflow.com/questions/29864330/histogram-of-gray-scale-values-in-numpy-image
  image_array = np.array(image)
  flat_image_array = image_array.flatten()
  histogram, bins = np.histogram(flat_image_array, bins=256, range=[0,256])
  cdf = histogram.cumsum()
  cdf_normalized = cdf * 255 / cdf[-1]
  equalized_image_array = np.interp(flat_image_array, bins[:-1], cdf_normalized)
  equalized_image_array = equalized_image_array.reshape(image_array.shape)
  return Image.fromarray(np.uint8(equalized_image_array))

def plot_histogram(image):
  image_array = np.array(image)
  plt.hist(image_array.flatten(), bins=256, range=[0,256])
  plt.xlabel("Pixel Intensity")
  plt.ylabel("Frequency")
  plt.show()

def calculate_statistics(image, willPrint = False):
  image_array = np.array(image)
  mean = np.mean(image_array)
  standard_deviation = np.std(image_array)
  if(willPrint):
    print(f"Mean: {mean:.2f}, Standard Deviation: {standard_deviation:.2f}")
  return (mean, standard_deviation)

if __name__ == "__main__":

  # Part 1: Transformation Functions
  image_path = "fourierspectrum.pgm"
  image = Image.open(image_path)

  ## log transformation
  log_transform(image).save("log_transformed.jpg")

  ## power law transformation
  gammas = [0.1, 0.2, 0.4, 0.67, 1, 1.5, 2.5, 5.0, 10.0]
  for gamma in gammas:
    power_transform(image, gamma=gamma).save(f"power_transform_𝛾({gamma}).jpg")
    
  # Part 2: Histogram Equalization
  image_path = "banker.jpeg"
  image = Image.open(image_path)

  # Equalization
  equalized = histogram_equalization(image)
  equalized.save("banker_equalized.jpeg")

  # Mean and Standard Deviation
  calculate_statistics(image, willPrint=True)
  calculate_statistics(equalized, willPrint=True)

  # Histogram plotting
  plot_histogram(image)
  plot_histogram(equalized)
