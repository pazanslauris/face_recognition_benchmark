# Imports
import cv2
import time
import numpy as np
import dlib


# Init Network
pose_predictor_5_point = dlib.shape_predictor("dlib/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib/dlib_face_recognition_resnet_model_v1.dat")


# Load Image
loaded_img = cv2.imread("images/obama-aligned.jpg")
resized_img = cv2.resize(loaded_img, (300, 300), interpolation=cv2.INTER_LINEAR)  # Resize to 300x300
blank_img = np.zeros((300, 300, 3), np.uint8)


# Run on blank
# Done because some networks do some setup on the first call...
face_landmarks = pose_predictor_5_point(resized_img, dlib.rectangle(0, 0, 300, 300))
face_encoder.compute_face_descriptor(blank_img, face_landmarks, 1)


# Run Benchmark
times_to_run = 1000
time1 = time.clock()
for x in range(times_to_run):
    face_encoder.compute_face_descriptor(resized_img, face_landmarks, 1)

time2 = time.clock()
elapsed_time = (time2 - time1)


# Output results
average_time = elapsed_time / times_to_run
print(" - Face embeddings(With Dlib): {:.4f}s ({:.2f} fps)".format(average_time, 1 / average_time))