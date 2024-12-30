import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model
model = joblib.load("random_forest_model.pkl")

# Initialize Mediapipe Hands Detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDrawing = mp.solutions.drawing_utils


# Functions for Normalization
def norm_lm(landmarks):
    # Reference point is the wrist
    wrist = landmarks[0]  # (x, y, z) of the wrist landmark

    # Initialize an empty list for normalized landmarks
    norm = []

    # Iterate through each landmark
    for lm in landmarks:
        # Subtract wrist coordinates from each landmark
        normalized = (lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2])
        norm.append(normalized)

    return norm


# Process a Video
cam = cv2.VideoCapture(0)

new_data = []  # NEW data storage for comparison

while True:
    ret, frame = cam.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2 and mediapipe use different color format, so we need to change
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append((lm.x, lm.y, lm.z))  # Extract (x, y, z)

            # Normalize and flatten landmarks
            normalized_landmarks = norm_lm(landmarks)
            normalized_landmarks = np.array(normalized_landmarks).reshape(1, -1)  # Reshape for the model

            # Make prediction
            labels = ['rock', 'paper', 'scissors']
            prediction = model.predict(normalized_landmarks)[0]
            print(prediction)

            # Draw landmarks on the hand
            mpDrawing.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # Display the video feed with predictions
    cv2.imshow("Hand Gesture Recognition", frame)

    # break the loop if press q
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


# Release the video capture
cam.release()