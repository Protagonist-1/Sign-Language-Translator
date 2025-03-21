import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for hand_type in ['left_hand', 'right_hand']:  # Loop through both hands
    HAND_DIR = os.path.join(DATA_DIR, hand_type)
    
    for label in os.listdir(HAND_DIR):  # 0-26 folders
        label_path = os.path.join(HAND_DIR, label)
        
        for img_path in os.listdir(label_path):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(label_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(int(label))  # Convert folder name to int

# Save data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
