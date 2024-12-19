import cv2
import numpy as np
import dlib
from math import hypot
import time

# Camera initialization
cap = cv2.VideoCapture(0)

# Dlib face and landmark detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_SIMPLEX

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_length / ver_line_length
    return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
    eye_region = np.array([(facial_landmarks.part(eye_points[i]).x, facial_landmarks.part(eye_points[i]).y) for i in range(6)], np.int32)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(frame, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    gray_eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0:height, 0:int(width / 2)]
    right_side_threshold = threshold_eye[0:height, int(width / 2):width]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white == 0:
        gaze_ratio = 0.5
    elif right_side_white == 0:
        gaze_ratio = 2
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

# Initialize variables for feedback
blink_count = 0
start_time = time.time()
session_duration = float(input("Introduceti durata sesiunii in minute: ")) * 60
attention_frames = 0
total_frames = 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        
        # Count blinks
        if blinking_ratio > 4.7:
            blink_count += 1

        gaze_ratio_left = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right + gaze_ratio_left) / 2

        # Attention detection
        total_frames += 1
        if 0.85 < gaze_ratio < 1.3:  # Looking at the center
            attention_frames += 1

        # Display gaze direction
        if gaze_ratio <= 0.85:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
        elif 0.85 < gaze_ratio < 1.3:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)

    elapsed_time = time.time() - start_time
    if elapsed_time > session_duration:
        break

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

# Calculate session metrics0.1
attention_percentage = (attention_frames / total_frames) * 100
sleepiness = "Da" if blink_count > (session_duration / 60) * 20 else "Nu"  # Adjust threshold as needed
#print(f"frame uri: {total_frames}")
# Feedback
print(f"Durata sesiune: {session_duration / 60} minute")
print(f"Numar clipiri: {blink_count}")
print(f"somnolenta: {sleepiness}")
print(f"Procentaj de atentie: {attention_percentage:.2f}%")

cap.release()
cv2.destroyAllWindows()
