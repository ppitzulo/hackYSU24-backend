from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import requests

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allows only requests from your React app origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

PREDICTION_URL = "https://guitarchords826-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/352d7a3b-5364-4b69-9fba-bce2cf2c928e/classify/iterations/Iteration6/image"
#it5
# PREDICTION_URL = "https://guitarchords826-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/352d7a3b-5364-4b69-9fba-bce2cf2c928e/classify/iterations/Iteration5/image"
headers = {
    "Prediction-Key": "4a702a82e3b34cc8bfbb96c623f8bd80",
    "Content-Type": "application/octet-stream"
}

@app.post("/predict/")
async def predict_chord(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    predictions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Assuming processing and prediction logic here
            # For demonstration, we just print the hand landmarks
            # print(hand_landmarks)
            
            # Normally, you'd send the image or relevant data for prediction here
            # For simplicity, let's assume we send the entire image
            response = requests.post(PREDICTION_URL, headers=headers, data=contents)
            prediction_result = response.json()
            predictions.append(prediction_result)
            print(prediction_result)

    return JSONResponse(status_code=200, content={"predictions": predictions})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
