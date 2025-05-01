from scipy.spatial import distance      
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# Initialize mixer for alert sound
mixer.init()

# Thresholds and duration for head movement detection
no_movement_duration = 10  
movement_threshold = 10  
buffer_size = 5  

# Yawning and Eye Blink Detection Constants
EAR_THRESHOLD = 0.25  
MAR_THRESHOLD = 0.60  
EYE_AR_CONSEC_FRAMES = 3  
MOUTH_AR_CONSEC_FRAMES = 3  

# Initialize HOG-based face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()  # HOG + SVM detector
predictor = dlib.shape_predictor(r"C:\Users\aravi\AppData\Local\Programs\Python\Python313\drowsiness_mini_project\models\shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize variables
start_time = None
face_center_prev = None
blink_counter = 0
yawn_counter = 0
head_movement_detected = False
movement_buffer = []

plt.ion()  
fig, ax = plt.subplots()
ax.axis('off')

# EAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR calculation
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

frame_count = 0  # To control face detection frequency

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces every 2 frames to improve efficiency
    if frame_count % 2 == 0:
        faces = detector(gray, 1)  # The '1' makes detection more accurate by upscaling image
    
    for face in faces:
        shape = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)

        # Face center detection
        face_center = ((face.left() + face.right()) // 2, (face.top() + face.bottom()) // 2)
        radius = (face.right() - face.left()) // 2
        cv2.circle(frame, face_center, radius, (0, 255, 0), 2)  

        # Head movement detection
        if face_center_prev is not None:
            movement = distance.euclidean(face_center, face_center_prev)
            movement_buffer.append(movement)
            if len(movement_buffer) > buffer_size:
                movement_buffer.pop(0)

            smoothed_movement = np.mean(movement_buffer)
            if smoothed_movement > movement_threshold:
                start_time = time.time()
                head_movement_detected = True
            else:
                head_movement_detected = False
        else:
            start_time = time.time()  

        face_center_prev = face_center  

        # Detect Eyes and Yawning
        left_eye, right_eye, mouth = shape_np[42:48], shape_np[36:42], shape_np[48:68]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        if ear < EAR_THRESHOLD:
            blink_counter += 1
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "ALERT: Blink Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            blink_counter = 0

        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= MOUTH_AR_CONSEC_FRAMES:
                cv2.putText(frame, "ALERT: Yawn Detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            yawn_counter = 0

        for (x, y) in shape_np:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.001)

    frame_count += 1  

    if not plt.fignum_exists(fig.number):
        break

cap.release()
plt.ioff()
plt.show()
