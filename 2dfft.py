import cv2
import numpy as np
import os 

# Define the path of the single input image
input_image_path = '/Users/ivan_gorbachenko/Desktop/thesis_practical/1.jpg'
output_path = '/Users/ivan_gorbachenko/Desktop/thesis_practical'


image = cv2.imread(input_image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#  Gaussian blur to remove noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# 2D FFT to the blurred image
fft_result = np.fft.fft2(blurred)
fft_shifted = np.fft.fftshift(fft_result)
magnitude_spectrum = 10 * np.log(np.abs(fft_shifted))
# Bandpass filter parameters
rows, cols = gray.shape
crow, ccol = rows // 2, cols // 2
d = 30  # cutoff distance from center
n = 2   # filter order, adjust this value as needed

# Creating Bandpass filter
mask = np.zeros((rows, cols), np.uint8)
mask[crow - d:crow + d, ccol - d:ccol + d] = 1
mask = cv2.GaussianBlur(mask.astype(float), (0, 0), n)


fft_shifted_filtered = fft_shifted * mask
filtered_img = np.fft.ifft2(np.fft.ifftshift(fft_shifted_filtered)).real

# Normalize the filtered image to the range 0-255 and convert to 8-bit unsigned integer
filtered_img_normalized = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)
filtered_img_normalized = filtered_img_normalized.astype(np.uint8)

# Threshold the filtered image
_, thresh = cv2.threshold(filtered_img_normalized, 120, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contour_img = image.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
count = 0

# Check for circular contours and draw bounding boxes
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

    area = cv2.contourArea(contour)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (radius ** 2)
    circularity = area / circle_area

    if 0.48 < circularity < 1.3 and area > 200:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contour_img, (x-3, y-3), (x + w + 3, y + h + 3), (0, 0, 255), 2)
        count += 1

# count text to the top left corner of the image
cv2.putText(contour_img, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

output_filename = os.path.join(output_path, 'output_image.jpg')  # Define the output file name
cv2.imwrite(output_filename, thresh)  # Save the image

cv2.imshow('Detected Stomata', contour_img)
cv2.imshow('Binary Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

