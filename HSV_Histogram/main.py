import cv2
import matplotlib.pyplot as plot
import numpy as np

image = cv2.imread('H:\\DAT\\HCSDL_DPT\\Data\\Rose\\rose_1.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# h_channel = hsv_image[:,:,0]
# s_channel = hsv_image[:,:,1]
# v_channel = hsv_image[:,:,2]
#
# h_hist = cv2.calcHist([h_channel], [0], None, [256], [0, 256])
# s_hist = cv2.calcHist([s_channel], [0], None, [256], [0, 256])
# v_hist = cv2.calcHist([v_channel], [0], None, [256], [0, 256])
#
# fig, axes = plot.subplots(3, 1, figsize=(8, 6))
#
# axes[0].plot(h_hist, color='r')
# axes[0].set_xlim([0, 180])
# axes[0].set_title('Hue Histogram')
#
# axes[1].plot(s_hist, color='g')
# axes[1].set_xlim([0, 256])
# axes[1].set_title('Saturation Histogram')
#
# axes[2].plot(v_hist, color='b')
# axes[2].set_xlim([0, 256])
# axes[2].set_title('Value Histogram')
#
# plot.tight_layout()
# plot.show()

histogram = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 12, 3], [0, 180, 0, 256, 0, 256])

# Reshape the histogram to a 1D vector
histogram_vector = histogram.ravel()
print(histogram_vector)
print(len(histogram_vector))

# Normalize the histogram vector (optional)
# histogram_vector /= np.sum(histogram_vector)

# Display the histogram vector
# print(sum(histogram_vector))

# Extract the individual HSV channels
# hue_channel = hsv_image[:, :, 0]
# saturation_channel = hsv_image[:, :, 1]
# value_channel = hsv_image[:, :, 2]
#
# # Convert each channel to a numpy array
# hue_array = np.array(hue_channel)
# saturation_array = np.array(saturation_channel)
# value_array = np.array(value_channel)
#
# # Print the shape of each array
# print(len(hue_array))
# print(len(saturation_array))
# print(len(value_array))
