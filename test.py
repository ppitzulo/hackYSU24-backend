import cv2
import numpy as np

# Function to filter skin color
def filter_skin_color(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 70, 140], dtype=np.uint8)

    # Filter skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological transformations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    return mask

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Filter for skin color
    mask = filter_skin_color(frame)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which will be assumed to be the hand
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 1000:  # Filter out small contours
            # Draw the contour on the original frame
            cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
