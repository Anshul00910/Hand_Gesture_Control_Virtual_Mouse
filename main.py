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

# Scroll gesture tracking
scroll_threshold = 50  # The required distance to trigger a scroll gesture
scroll_time = 0
last_scroll_direction = None

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

            index_x, index_y, thumb_x, thumb_y, middle_x, middle_y = 0, 0, 0, 0, 0, 0
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

                if id == 12:  # Middle finger tip
                    middle_x = screen_width / frame_width * x
                    middle_y = screen_height / frame_height * y
                    cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)  # Blue circle on middle tip

            # Cursor movement with smooth transition
            curr_x = prev_x + (index_x - prev_x) / smoothening
            curr_y = prev_y + (index_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Distance calculations for clicking
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

            # Detect if both fingers are close and moving vertically for scroll
            if abs(index_y - middle_y) > scroll_threshold:  # Check vertical distance for scroll
                if index_y < middle_y:  # Scroll down
                    pyautogui.scroll(-10)  # Scroll down by 10 units
                    last_scroll_direction = "down"
                elif index_y > middle_y:  # Scroll up
                    pyautogui.scroll(10)  # Scroll up by 10 units
                    last_scroll_direction = "up"

                # Prevent scrolling too fast
                if (current_time - scroll_time) > 0.3:
                    scroll_time = current_time  # Reset scroll time to avoid multiple scrolling

            else:
                last_scroll_direction = None  # Reset if not in scroll gesture position

            # Show scrolling direction on screen for debugging
            if last_scroll_direction:
                cv2.putText(frame, f"Scrolling {last_scroll_direction}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Virtual Mouse with Scroll Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
