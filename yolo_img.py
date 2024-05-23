import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/Users/ivan_gorbachenko/Desktop/machine learning/stomata_count/best.pt")

# Path to the input image
image_path = ""

image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    raise ValueError(f"Image not found at {image_path}")

# Run YOLOv8 inference on the image
results = model.predict(image)


# Visualize the results on the image
annotated_image = results[0].plot()


if len(results[0].boxes.cls)> 0:
    count = len(results[0].boxes.cls)
    text = f"Count: {count}"
    position = (30, 30)  # Position to print the text
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    font_scale = 1  # Font scale factor
    font_color = (0, 255, 0)  # Font color (red in BGR)
    line_type = 3  # Line type (thickness)

    cv2.putText(annotated_image, text, position, font, font_scale, font_color, line_type)

# Save the annotated image to a file
output_path = "images/output.jpg"
cv2.imwrite(output_path, annotated_image)

# Display the annotated image (optional)
cv2.imshow("YOLOv8 Inference", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()