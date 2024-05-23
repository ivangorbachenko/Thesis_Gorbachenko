import cv2
import numpy as np
from ultralytics import YOLO

# Loading the YOLOv8 model 
model = YOLO("/Users/ivan_gorbachenko/Desktop/machine learning/stomata_count/best.pt")

#path to the input video
video_path = "/Users/ivan_gorbachenko/Desktop/machine learning/stomata_count/7A.avi"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Error opening video stream or file")

# Getting width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Creating an empty black image for accumulation(heatmap)
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # running model through each frame
    results = model.predict(frame)

    # createing a temporary image to hold the current frame's detections
    temp_image = np.zeros((frame_height, frame_width), dtype=np.float32)

    # extracting coordinares of stomata
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        # center of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        # circle at the center of the bounding box on the temporary image
        cv2.circle(temp_image, (center_x, center_y), radius=20, color=1, thickness=-1)

    # adding the temporary image to the heatmap
    heatmap += temp_image


cap.release()

heatmap_blurred = cv2.GaussianBlur(heatmap, (9, 9), 0)
#cv2.imshow("With Gaussian", heatmap_blurred)

# normalizing the heatmap for visualization
heatmap_normalized = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX)
heatmap_uint8 = np.uint8(heatmap_normalized)

_, heatmap_thresholded = cv2.threshold(heatmap_uint8, 15, 255, cv2.THRESH_BINARY)

# local maxima to identify stomata locations
contours, _ = cv2.findContours(heatmap_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# drawing contours on the original heatmap for visualization
heatmap_contours = cv2.cvtColor(heatmap_uint8, cv2.COLOR_GRAY2BGR)
cv2.drawContours(heatmap_contours, contours, -1, (0, 255, 0), 2)

contour_count = len(contours)

# contour count
cv2.putText(heatmap_contours, f"Count: {contour_count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Save the resulting heatmap
#output_path = "/path/to/your/heatmap.jpg"
#cv2.imwrite(output_path, heatmap_contours)

# Display the resulting heatmap (optional)
cv2.imshow("Heatmap", heatmap_uint8)
cv2.imshow("contours", heatmap_contours)
#cv2.imshow("After gaussian", heatmap_blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

