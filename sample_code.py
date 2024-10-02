import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hands model outside the loop to improve efficiency
hands = mp_hands.Hands(max_num_hands=4)
webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    success, img = webcam.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue  # Skip the rest of the loop if no frame is captured

    # Convert image to RGB for MediaPipe processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the processed image with hand annotations
    cv2.imshow('Artemis', img)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
