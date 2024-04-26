import cv2 as cv
from skimage.feature import hog
import numpy as np


def main():
    image = cv.imread('H:\\DAT\\HCSDL_DPT\\Data\\picture30.jpg')
    feature = extract_hog(image)
    print(feature)


def extract_hog(image):
    gray_image = convert_bgr_to_gray(image)
    (hog_vector, hog_image) = hog(gray_image,
                                  orientations=9,
                                  pixels_per_cell=(8, 8),
                                  transform_sqrt=True,
                                  cells_per_block=(2, 2),
                                  block_norm="L2",
                                  visualize=True)
    return hog_vector


def convert_bgr_to_gray(image):
    width = image.shape[1]
    height = image.shape[0]
    gray_image = np.zeros((height, width), dtype=np.float32)
    for i in range(width):
        for j in range(height):
            B, G, R = image[j, i, 0], image[j, i, 1], image[j, i, 2]
            gray_image[j, i] = 0.299 * R + 0.587 * G + 0.114 * B
    return gray_image


if __name__ == '__main__':
    main()
