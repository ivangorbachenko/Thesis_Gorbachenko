import cv2


def updateValue(new_value):
    global trackbar_value
    trackbar_value = new_value

trackbar_value = 120

keypoints = []

image = cv2.imread("/Users/ivan_gorbachenko/Desktop/thesis_practical/1.jpg", 0)
#median_filtered = cv2.medianBlur(image, 11)
#gray_filtered= cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,19,11)

cv2.namedWindow("Threshold")
cv2.createTrackbar("Value", "Threshold", trackbar_value, 200, updateValue)

# Blob detector parameters
blobparams = cv2.SimpleBlobDetector_Params()

# Filter blobs by area
blobparams.filterByArea = True
blobparams.minArea = 100  
blobparams.maxArea = 5000 

# Filter by circularity
blobparams.filterByCircularity = False

# Filter by inertia (to detect oval shapes)
blobparams.filterByInertia = True
blobparams.minInertiaRatio = 0.1  

# Filter by convexity
blobparams.filterByConvexity = False

detector = cv2.SimpleBlobDetector_create(blobparams)


while True:
    #gray_filtered= cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,17,9)
    
    ret, thresh = cv2.threshold(image, trackbar_value, 255, cv2.THRESH_BINARY)
    keypoints = detector.detect(thresh)

    blob_count = len(keypoints)
    
    thresh = cv2.drawKeypoints(thresh, keypoints, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    #blur = cv2.GaussianBlur(img, (1, 1), 5)
    #gray_filtered = cv2.inRange(blur, thresh, 255,cv2.THRESH_TOZERO)
    for kp in keypoints:
        x, y = kp.pt
        s = kp.size
        top_left = (int(x - s / 2), int(y - s / 2))
        bottom_right = (int(x + s / 2), int(y + s / 2))
        cv2.rectangle(thresh, top_left, bottom_right, (0, 255, 0), 2)

    #cv2.imshow("Original", image)
    cv2.putText(thresh, f"Blobs: {blob_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
    cv2.imshow("Threshold", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
