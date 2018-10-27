# Imports
import cv2
import time
import numpy as np
import dlib

# Init Network
pose_predictor_68_point = dlib.shape_predictor("dlib/shape_predictor_68_face_landmarks.dat")


# Load Image
loaded_img = cv2.imread("images/obama-unaligned.jpg")
resized_img = cv2.resize(loaded_img, (300, 300), interpolation=cv2.INTER_LINEAR)  # Resize to 300x300
blank_img = np.zeros((300, 300, 3), np.uint8)


# Run on blank
# Done because some networks do some setup on the first call...
pose_predictor_68_point(blank_img, dlib.rectangle(0, 0, 300, 300))


# Run Benchmark
times_to_run = 1000
time1 = time.clock()
for x in range(times_to_run):
    pose_predictor_68_point(loaded_img, dlib.rectangle(0, 0, loaded_img.shape[1], loaded_img.shape[0]))

time2 = time.clock()
elapsed_time = (time2 - time1)


# Output results
average_time = elapsed_time / times_to_run
print(" - Face alignment(With Dlib-68pt): {:.4f}s ({:.2f} fps)".format(average_time, 1 / average_time))