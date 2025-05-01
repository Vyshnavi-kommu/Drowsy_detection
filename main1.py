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
no_movement_duration = 10  # Time in seconds
movement_threshold = 10  # Adjust based on real-world tests
buffer_size = 5  # Number of frames for smoothing movement

# Yawning and Eye Blink Detection Constants
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for blinking
MAR_THRESHOLD = 0.60  # Mouth Aspect Ratio threshold for yawning
EYE_AR_CONSEC_FRAMES = 3  # Consecutive frames for eye blink detection
MOUTH_AR_CONSEC_FRAMES = 3  # Consecutive frames for yawning detection

# Initialize face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\aravi\AppData\Local\Programs\Python\Python313\drowsiness_mini_project\models\shape_predictor_68_face_landmarks.dat")

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize variables for head movement, yawning, and blinking detection
start_time = None
face_center_prev = None
blink_counter = 0
yawn_counter = 0
head_movement_detected = False
movement_buffer = []  # Store recent movements for smoothing

# Set up Matplotlib for displaying video frames
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
ax.axis('off')

# Calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Calculate the Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    for face in faces:
        # Get facial landmarks
        shape = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)

        # Calculate the center of the face
        face_center = ((face.left() + face.right()) // 2, (face.top() + face.bottom()) // 2)

        # Draw circular grid on the face
        radius = (face.right() - face.left()) // 2
        cv2.circle(frame, face_center, radius, (0, 255, 0), 2)  # Outer circle
        for r in range(10, radius, 20):
            cv2.circle(frame, face_center, r, (0, 255, 0), 1)  # Inner concentric circles

        # Improved Head Movement Detection
        if face_center_prev is not None:
            movement = distance.euclidean(face_center, face_center_prev)

            # Add movement to buffer and keep only recent `buffer_size` movements
            movement_buffer.append(movement)
            if len(movement_buffer) > buffer_size:
                movement_buffer.pop(0)

            # Compute smoothed movement
            smoothed_movement = np.mean(movement_buffer)

            if smoothed_movement > movement_threshold:
                start_time = time.time()  # Reset the timer if movement is detected
                head_movement_detected = True
            else:
                head_movement_detected = False
        else:
            start_time = time.time()  # Initialize the timer for the first detected face

        face_center_prev = face_center  # Update previous face center 

        # Check if the timer exceeds the no-movement duration
        elapsed_time = time.time() - start_time
        if elapsed_time >= no_movement_duration and not head_movement_detected:
            cv2.putText(frame, "ALERT: Tilt Your Head!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Detect Eyes and Yawning
        left_eye = shape_np[42:48]
        right_eye = shape_np[36:42]
        mouth = shape_np[48:68]

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Calculate MAR for the mouth
        mar = mouth_aspect_ratio(mouth)

        # Check for eye blink
        if ear < EAR_THRESHOLD:
            blink_counter += 1
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "ALERT: Blink Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            blink_counter = 0

        # Check for yawning
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= MOUTH_AR_CONSEC_FRAMES:
                cv2.putText(frame, "ALERT: Yawn Detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            yawn_counter = 0

        # Draw facial landmarks
        for (x, y) in shape_np:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Display the frame using Matplotlib
    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.001)

    # If the window is closed, break the loop
    if not plt.fignum_exists(fig.number):
        break

# Release resources
cap.release()
plt.ioff()
plt.show()