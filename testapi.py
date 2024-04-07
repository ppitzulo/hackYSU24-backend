import cv2
import mediapipe as mp
import numpy as np
import time
import requests

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Focus on detecting a single hand
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Variable to keep track of the last save time
last_save_time = time.time()
save_interval = 1.1  # Save an image every 0.5 seconds

PREDICTION_URL = "https://guitarchords826-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/352d7a3b-5364-4b69-9fba-bce2cf2c928e/classify/iterations/Iteration6/image"

headers = {
    "Prediction-Key": "4a702a82e3b34cc8bfbb96c623f8bd80",
    "Content-Type": "application/octet-stream"
}

latest_prediction = None


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame color to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            # Check if the hand detected is right hand
            if handedness.classification[0].label == "Right":
                # Draw the hand annotations on the image.
                # mp_drawing.draw_landmarks(
                #     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                #     mp_drawing.DrawingSpec(color=(0, 255, 0Detailed error message: {"error":{"code":"401","message":"Access denied due to invalid subscription key or wrong API endpoint. Make sure to provide a valid key for an active subscription and use a correct regional API endpoint for your resource."}}
    # ), thickness=2, circle_radius=2),
                #     mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                # )

                # If the current time is more than 0.5 seconds since the last save
                if time.time() - last_save_time > save_interval:
                    last_save_time = time.time()

                    # Calculate the bounding box of the hand
                    h, w, _ = frame.shape
                    landmark_array = np.array(
                        [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
                    )
                    x, y, w, h = cv2.boundingRect(landmark_array.astype(np.int32))

                    # Calculate the center of the bounding box
                    cx, cy = x + w // 2, y + h // 2

                    # Make the bounding box twice as big while keeping it centered
                    x = max(cx - w, 0)
                    y = max(cy - h, 0)
                    w = min(w * 2, frame.shape[1] - x)
                    h = min(h * 2, frame.shape[0] - y)

                    # Ensure the expanded bounding box is within the frame
                    x, y = max(x, 0), max(y, 0)
                    w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

                    # Check if the bounding box is valid (non-zero area) and within the frame boundaries
                    if (
                        w > 0
                        and h > 0
                        and x + w <= frame.shape[1]
                        and y + h <= frame.shape[0]
                    ):
                        hand_img = frame[y : y + h, x : x + w]

                        # Check if the cropped image is not empty
                        if hand_img.size > 0:
                            # Convert the image to the format expected by the API (bytes)
                            _, img_encoded = cv2.imencode('.jpg', hand_img)
                            img_bytes = img_encoded.tobytes()

                            # Make prediction by sending a POST request
                            try:
                                response = requests.post(PREDICTION_URL, headers=headers, data=img_bytes)
                                response.raise_for_status()  # This will raise an exception for HTTP error codes

                                predictions = response.json().get('predictions', [])
                                # Right after receiving and processing predictions:
                                if predictions:
                                    # Assuming the most confident prediction is first
                                    most_confident_prediction = predictions[0]
                                    latest_prediction = f"{most_confident_prediction['tagName']}: {most_confident_prediction['probability'] * 100:.2f}%"
                                else:
                                    latest_prediction = "No predictions"

                                # if predictions:  # Check if the predictions list is not empty
                                #     for prediction in predictions:
                                #         print("\t" + prediction['tagName'] + ": {0:.2f}%".format(prediction['probability'] * 100))
                                # else:
                                #     print("No predictions were returned.")
                            except requests.exceptions.HTTPError as http_err:
                                print(f"HTTP error occurred: {http_err} - Status Code: {response.status_code}")
                                print("Detailed error message:", response.text)
                            except Exception as err:
                                print(f"An unexpected error occurred: {err}")
    if latest_prediction:
        # Setting the font scale and thickness
        font_scale = 0.6
        thickness = 2
        
        # Getting the width and height of the text box
        text_size = cv2.getTextSize(latest_prediction, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = frame.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
        text_y = 20  # 20 pixels from the top
        
        # Setting the rectangle background
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        
        # Putting the text on the frame
        cv2.putText(frame, latest_prediction, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)



    cv2.imshow("MediaPipe Hands (Right Hand Only)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
