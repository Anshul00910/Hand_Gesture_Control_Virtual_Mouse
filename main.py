import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize camera and hand tracking
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Cursor smoothing
smoothening = 5
prev_x, prev_y = 0, 0

# Click debounce times
click_time = 0
double_click_time = 0
last_click = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            index_x, index_y, thumb_x, thumb_y = 0, 0, 0, 0
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index finger tip
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # Green circle on index tip

                if id == 4:  # Thumb tip
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)  # Red circle on thumb tip

            # Cursor movement with smooth transition
            curr_x = prev_x + (index_x - prev_x) / smoothening
            curr_y = prev_y + (index_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Distance calculations
            index_thumb_dist = np.hypot(thumb_x - index_x, thumb_y - index_y)

            current_time = time.time()

            # Left Click (Thumb and Index finger close)
            if index_thumb_dist < 30:
                if (current_time - click_time) > 0.5:  # Prevents multiple clicks
                    pyautogui.click()
                    click_time = current_time
                    last_click = current_time  # Update last click time for double-click detection

            # Double Click (Thumb and Index close twice quickly)
            if index_thumb_dist < 30 and (current_time - last_click) < 0.3:
                if (current_time - double_click_time) > 0.5:
                    pyautogui.doubleClick()
                    double_click_time = current_time

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
