# Imports
import cv2
import time
import numpy as np
import dlib


# Init Network
cnn_face_detector = dlib.cnn_face_detection_model_v1("dlib/mmod_human_face_detector.dat")


# Load Image
loaded_img = cv2.imread("images/obama-1080p.jpg")
cropped_img = loaded_img[0:loaded_img.shape[1], 0:loaded_img.shape[1]]  # Crop to 1:1 aspect ratio
resized_img = cv2.resize(cropped_img, (416, 416), interpolation=cv2.INTER_LINEAR)  # Resize to 416x416
blank_img = np.zeros((416, 416, 3), np.uint8)


# Run on blank
# Done because some networks do some setup on the first call...
cnn_face_detector(blank_img, 1)


# Run Benchmark
times_to_run = 1000
time1 = time.clock()
for x in range(times_to_run):
    cnn_face_detector(resized_img, 1)

time2 = time.clock()
elapsed_time = (time2 - time1)


# Output results
average_time = elapsed_time / times_to_run
print(" - Face detection(With Dlib CNN): {:.4f}s ({:.2f} fps)".format(average_time, 1 / average_time))