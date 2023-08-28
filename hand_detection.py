import cv2
import numpy as np
import mediapipe as mp
import pygame

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize Pygame
pygame.mixer.init()

# Open the camera feed
cap = cv2.VideoCapture(0)  # Use camera index 0 (usually built-in camera)

# Define landmark indices for each finger (tip and middle)
finger_landmarks = {
    "thumb": (4, 3),
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18)
}

# Initialize previous finger statuses
prev_finger_status = {finger: "Unknown" for finger in finger_landmarks}

# Define music notes and corresponding sound files
music_notes = {
    "thumb": "note_c.wav",
    "index": "note_d.wav",
    "middle": "note_e.wav",
    "ring": "note_f.wav",
    "pinky": "note_g.wav"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe uses RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(frame_rgb)

    # Finger status dictionary
    finger_status = {}

    # Analyze finger positions
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for finger, (tip_idx, middle_idx) in finger_landmarks.items():
                h, w, c = frame.shape
                tip_y = int(landmarks.landmark[tip_idx].y * h)
                middle_y = int(landmarks.landmark[middle_idx].y * h)

                if tip_y < middle_y:  # Finger is up
                    finger_status[finger] = 'Up'
                else:  # Finger is down
                    finger_status[finger] = 'Down'

    # Compare with previous finger statuses
    for finger, status in finger_status.items():
        if status != prev_finger_status[finger]:
            print(f"{finger.capitalize()} finger changed to: {status}")
            prev_finger_status[finger] = status

            if status == 'Down':
                # Play the corresponding music note
                sound_filename = music_notes.get(finger)
                if sound_filename:
                    sound = pygame.mixer.Sound(sound_filename)
                    sound.play()

    # Display the frame with drawn lines
    cv2.imshow('Hand Detection and Drawing', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
