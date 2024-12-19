# Code for implementation without breathing detection functionality
# This is a simplified version of the Digital Twin Yoga Master without breathing detection.

import math
import cv2
import mediapipe as mp
import time
from datetime import datetime
import pyttsx3
import os
import numpy as np
import threading

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Initialize drawing utils


# Function to calculate angles between three points
def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
    )
    if angle < 0:
        angle += 360
    return angle


# Initialize the TTS engine
tts_engine = pyttsx3.init()

# Set TTS properties
tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
tts_engine.setProperty('rate', 100)   # Slower speech rate

def text_to_speech(text):
    def speak():
        try:
            tts_engine.say(text) 
            tts_engine.runAndWait()
        except Exception as e:
            print(f"An error occurred: {e}")
     # Run the speech function in a separate thread
    speech_thread = threading.Thread(target=speak)
    speech_thread.start()
for step in range(7):  # Simulating camera frames
    print(f"Processing camera frame ")
    if step == 0:  # Example: Speak when processing frame 3
        text_to_speech("Step 1: Stand straight with legs together.")
    elif step == 1:  # Example: Speak when processing frame 3
        text_to_speech("Step 2: widen your legs apart.")
    elif step == 2:  # Example: Speak when processing frame 3
        text_to_speech("Step 3: Spread your arms out horizontally.")
    elif step == 3:  # Example: Speak when processing frame 3
        text_to_speech("Step 4: Bend towards your right foot.")
    elif step == 4:  # Example: Speak when processing frame 3
        text_to_speech("Step 5: Raise your back up; hands and legs remain the same.")
    elif step == 5:  # Example: Speak when processing frame 3
        text_to_speech("Step 6: Put lower your hands.")
    elif step == 6:  # Example: Speak when processing frame 3
        text_to_speech("Step 7: Join your legs, stand straight.")
    #time.sleep(1)  # Simulate frame processing time
# Function to guide user through each step
def guideStep(landmarks, step, feedback_timer):
    try:
        if step == 1:  # Step 1: Stand straight
            left_knee_angle = calculateAngle(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            )
            right_knee_angle = calculateAngle(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            )

            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][1]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][1]

            if (
                170 < left_knee_angle < 190 and 170 < right_knee_angle < 190  # Knees straight
                and abs(left_hip - right_hip) < 20  # Hips level
                and abs(left_shoulder - right_shoulder) < 20  # Shoulders level
                and left_hip < left_ankle and right_hip < right_ankle  # Hips above ankles
            ):
                feedback_timer += 1
            else:
                feedback_timer = 0  # Reset timer if the pose isn't correct
            return feedback_timer, "Step 1: Stand straight with legs together."

        elif step == 2:  # Step 2: Spread your legs
            left_leg_distance = abs(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0] -
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][0]
            )
            right_leg_distance = abs(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0] -
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0]
            )
            if left_leg_distance > 50 and right_leg_distance > 50:  # Adjust thresholds as needed
                feedback_timer += 1
            else:
                feedback_timer = 0
            return feedback_timer, "Step 2: widen your legs apart."

        elif step == 3:  # Step 3: Spread your arms
            left_arm_distance = abs(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] -
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0]
            )
            right_arm_distance = abs(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] -
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0]
            )
            legs_still_spread = (
                abs(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0] -
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][0]) > 50 and
                abs(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0] -
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0]) > 50
            )
            if left_arm_distance > 50 and right_arm_distance > 50 and legs_still_spread:
                feedback_timer += 1
            else:
                feedback_timer = 0
            return feedback_timer, "Step 3: Spread your arms out horizontally."

        elif step == 4:  # Step 4: Bend towards the right foot
            bending_angle = calculateAngle(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            )
            if 80 < bending_angle < 110:  # Adjust thresholds as needed
                feedback_timer += 1
            else:
                feedback_timer = 0
            return feedback_timer, "Step 4: Bend towards your right foot."

        elif step == 5:  # Step 5: Spread your arms
            left_arm_distance = abs(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] -
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0]
            )
            right_arm_distance = abs(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] -
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0]
            )
            legs_still_spread = (
                abs(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0] -
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][0]) > 50 and
                abs(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0] -
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0]) > 50
            )
            if left_arm_distance > 80 and right_arm_distance > 80 and legs_still_spread:
                feedback_timer += 1
            else:
                feedback_timer = 0
            return feedback_timer, "Step 5: Raise your back up; hands and legs remain the same."

        elif step == 6:  # Step 6: Spread your legs
            left_arm_distance = abs(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] -
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][0]
            )
            right_arm_distance = abs(
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0] -
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][0]
            )
            left_leg_distance = abs(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0] -
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][0]
            )
            right_leg_distance = abs(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0] -
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0]
            )
            if left_leg_distance > 50 and right_leg_distance > 50 and 90 > left_arm_distance and 90 > right_arm_distance:  # Adjust thresholds as needed
                feedback_timer += 1
            else:
                feedback_timer = 0
            return feedback_timer, "Step 6: Lower your arms."

        if step == 7:  # Step 7: Stand straight
            left_knee_angle = calculateAngle(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            )
            right_knee_angle = calculateAngle(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            )

            left_leg_distance = abs(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0] -
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][0]
            )
            right_leg_distance = abs(
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0] -
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][0]
            )

            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][1]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][1]

            if (
                170 < left_knee_angle < 190 and 170 < right_knee_angle < 190  # Knees straight
                and abs(left_hip - right_hip) < 20  # Hips level
                and abs(left_shoulder - right_shoulder) < 20  # Shoulders level
                and left_hip < left_ankle and right_hip < right_ankle  # Hips above ankles
                and left_leg_distance < 20 and right_leg_distance < 20  # straight leg
            ):
                feedback_timer += 1
            else:
                feedback_timer = 0  # Reset timer if the pose isn't correct
            return feedback_timer, "Step 7: Join your legs, stand straight."

    except IndexError:
        feedback_timer = 0  # Reset if landmarks are not detected
        return feedback_timer, "Waiting for proper detection..."

VISIBILITY_THRESHOLD = 0.8

# Reference angles for Triangle Pose
REFERENCE_ANGLES = {
    "Right Elbow": 152,
    "Left Elbow": 152,
    "Right Knee": 152,
    "Left Knee": 152,
}

# Helper function for angle calculation
def calculate_angle_3d(point1, point2, point3):
    v1 = np.array([point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]])
    v2 = np.array([point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2]])

    v1_magnitude = np.linalg.norm(v1)
    v2_magnitude = np.linalg.norm(v2)

    if v1_magnitude == 0 or v2_magnitude == 0:
        return 0

    cosine_angle = np.clip(np.dot(v1, v2) / (v1_magnitude * v2_magnitude), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def draw_arc_with_feedback(frame, point1, point2, point3, angle, ref_angle, label, color=(0, 255, 0)):
    p1 = tuple(np.array(point1[:2], dtype=int))
    p2 = tuple(np.array(point2[:2], dtype=int))
    p3 = tuple(np.array(point3[:2], dtype=int))

    v1 = np.array([point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]])
    v2 = np.array([point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2]])
    cross_product_z = np.cross(v1[:2], v2[:2])
    clockwise = cross_product_z < 0

    start_angle = np.degrees(np.arctan2(point1[1] - point2[1], point1[0] - point2[0]))
    end_angle = np.degrees(np.arctan2(point3[1] - point2[1], point3[0] - point2[0]))

    if clockwise:
        start_angle, end_angle = end_angle, start_angle

    start_angle = (start_angle + 360) % 360
    end_angle = (end_angle + 360) % 360

    if end_angle < start_angle:
        end_angle += 360

    radius = 18

    angle_text = f"{int(angle)}"
    cv2.putText(frame, angle_text, (p2[0] + 20, p2[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0, 255, 0], 1)

    feedback = ""
    if angle < ref_angle - 20:
        feedback = "Straighten"
    elif angle > ref_angle + 20:
        feedback = "Bend more"
    else:
        feedback = "Good"

    if feedback == "Good":
        cv2.ellipse(frame, p2, (radius, radius), 0, start_angle, end_angle, [0, 255, 0], 2)
    else:
        cv2.ellipse(frame, p2, (radius, radius), 0, start_angle, end_angle, [0, 0, 255], 2)

    cv2.putText(frame, feedback, (p2[0] + 20, p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 255, 0], 1)

def process_joint_group(frame, group, landmarks, width, height, label):
    p1, p2, p3 = group
    if (
        landmarks[p1].visibility > VISIBILITY_THRESHOLD
        and landmarks[p2].visibility > VISIBILITY_THRESHOLD
        and landmarks[p3].visibility > VISIBILITY_THRESHOLD
    ):
        point1 = np.multiply(
            [landmarks[p1].x, landmarks[p1].y, landmarks[p1].z], [width, height, width]
        ).astype(int)
        point2 = np.multiply(
            [landmarks[p2].x, landmarks[p2].y, landmarks[p2].z], [width, height, width]
        ).astype(int)
        point3 = np.multiply(
            [landmarks[p3].x, landmarks[p3].y, landmarks[p3].z], [width, height, width]
        ).astype(int)

        angle = calculate_angle_3d(point1, point2, point3)

        ref_angle = REFERENCE_ANGLES.get(label, 90)

        draw_arc_with_feedback(frame, point1, point2, point3, angle, ref_angle, label)

# Joint groups with labels
JOINT_GROUPS = [
    (
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_WRIST.value,
        "Right Elbow",
    ),
    (
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value,
        "Left Elbow",
    ),
    (
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        "Right Knee",
    ),
    (
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
        "Left Knee",
    ),
]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    cv2.namedWindow("Triangle Pose Guide", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Triangle Pose Guide", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1) as pose:
        step = 1
        feedback_timer = 0
        flag = 0
        speak_flag = 0
        t = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(imageRGB)
            output_image = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark
                width, height = frame.shape[1], frame.shape[0]
                for group in JOINT_GROUPS:
                    process_joint_group(output_image, group[:3], landmarks, width, height, group[3])

                landmarks = [
                    (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]), landmark.z)
                    for landmark in results.pose_landmarks.landmark
                ]

                if flag == 0:
                    start_time = datetime.now()
                    flag = 1
                feedback_timer, instruction = guideStep(landmarks, step, feedback_timer)
                if speak_flag == 0:
                    text_to_speech(instruction)
                    speak_flag = 1

                new_instruction = instruction + " Timer: " + str(int(t))
                cv2.putText(output_image, new_instruction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # if step != 3:
                #     cv2.putText(output_image, new_instruction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # else:
                #     cv2.putText(output_image, new_instruction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if feedback_timer >= 1:
                    end_time = datetime.now()
                    time_difference = end_time-start_time
                    t = time_difference.total_seconds()
                    if t > 5:
                        step += 1
                        speak_flag = 0
                        feedback_timer = 0
                        flag = 0
                else:
                    start_time = datetime.now()
                    t = 0

 
                # Reset to first step if all steps are completed
                if step > 7:
                    #step = 1
                    break
 
            # Display the frame
            cv2.imshow("Triangle Pose Guide", output_image)
 
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
cap.release()
cv2.destroyAllWindows()