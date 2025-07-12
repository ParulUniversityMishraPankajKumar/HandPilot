import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Gesture tracking
prev_gesture = None
last_action_time = 0
gesture_delay = 0.8  # seconds

# FPS tracking
start_time = time.time()
frame_count = 0

# Function to check which fingers are up
def fingers_up(landmarks):
    finger_states = []
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    finger_dips = [6, 10, 14, 18]  # Corresponding lower joints

    for tip, dip in zip(finger_tips, finger_dips):
        finger_states.append(landmarks[tip].y < landmarks[dip].y)

    return finger_states  # List of booleans

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    current_gesture = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            finger_state = fingers_up(landmarks)
            open_count = sum(finger_state)

            # Gesture logic
            if finger_state == [True, False, False, False]:
                current_gesture = "right"
            elif finger_state == [True, True, False, False]:
                current_gesture = "left"
            elif open_count == 4:
                current_gesture = "up"
            elif open_count == 0:
                current_gesture = "down"

            # Trigger key press safely
            if current_gesture and current_gesture != prev_gesture and time.time() - last_action_time > gesture_delay:
                pyautogui.press(current_gesture)
                print(f"Gesture: {current_gesture}")
                prev_gesture = current_gesture
                last_action_time = time.time()

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show FPS (every 10 frames)
    frame_count += 1
    if frame_count >= 10:
        fps = frame_count / (time.time() - start_time)
        print("FPS:", round(fps, 2))
        frame_count = 0
        start_time = time.time()

    # Show video
    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
