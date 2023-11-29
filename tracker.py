import cv2
import mediapipe as mp
import pyautogui
import numpy as np

from rich.console import Console
from config import colors, usage

c = Console()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(usage.CAMERA)
cap.set(3, 320)
cap.set(4, 240)

cv2.namedWindow("tracker", cv2.WINDOW_NORMAL)
c.log("setting cv2.WINDOW_PROP")

pyautogui.PAUSE = 0.01

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            index_tip = tuple(np.multiply((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y), [frame.shape[1], frame.shape[0]]).astype(int))

            pyautogui.moveTo(index_tip[0], index_tip[1], pyautogui.FAILSAFE)

            for j, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                if j == mp_hands.HandLandmark.THUMB_CMC.value or j == mp_hands.HandLandmark.THUMB_TIP.value:
                    finger_index = 0
                else:
                    finger_index = (j - 1) // 4

                color = colors.finger_colors[min(finger_index, len(colors.finger_colors) - 1)]
                overlay = frame.copy()
                cv2.circle(overlay, (cx, cy), 5, color, -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.imshow("tracker", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
