import cv2
import mediapipe as mp
import numpy as np
import time
import uuid

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
save_interval = 0.25  # Save an image every 0.5 seconds

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
                #     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
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
                            folder_path = "testC/"
                            unique_filename = str(uuid.uuid4())  # Generate a random UUID string
                            img_name = f"hand_{unique_filename}.jpg"
                            cv2.imwrite(folder_path + img_name, hand_img)
                            print(f"Saved {img_name} to {folder_path}")
                        else:
                            print("Cropped image is empty. Skipping save.")

    cv2.imshow("MediaPipe Hands (Right Hand Only)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
