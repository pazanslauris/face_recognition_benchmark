# Imports
import cv2
import time
import numpy as np
from faceboxes.face_detector import FaceDetector


# Init Network
face_detector = FaceDetector("faceboxes/model.pb", gpu_memory_fraction=0.6, visible_device_list='0')


# Load Image
loaded_img = cv2.imread("images/obama-1080p.jpg")
cropped_img = loaded_img[0:loaded_img.shape[1], 0:loaded_img.shape[1]]  # Crop to 1:1 aspect ratio
resized_img = cv2.resize(cropped_img, (416, 416), interpolation=cv2.INTER_LINEAR)  # Resize to 416x416
blank_img = np.zeros((416, 416, 3), np.uint8)


# Run on blank
# Done because some networks do some setup on the first call...
face_detector(blank_img, score_threshold=0.6)


# Run Benchmark
times_to_run = 1000
time1 = time.clock()
for x in range(times_to_run):
    face_detector(resized_img, score_threshold=0.6)
time2 = time.clock()
elapsed_time = (time2 - time1)


# Output results
average_time = elapsed_time / times_to_run
print(" - Face detection(With FaceBoxes): {:.4f}s ({:.2f} fps)".format(average_time, 1 / average_time))