# Imports
import cv2
import time
import numpy as np
import tensorflow as tf
import facenet.facenet as facenet

# Init Network
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
facenet.load_model("facenet/20180402-114759.pb")

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")



# Load Image
loaded_img = cv2.imread("images/obama-aligned.jpg")
resized_img = cv2.resize(loaded_img, (160, 160), interpolation=cv2.INTER_LINEAR)  # Resize to 160x160
images = []
images.append(facenet.prewhiten(resized_img))
images = np.stack(images)


# Run on blank
# Done because some networks do some setup on the first call...
feed_dict = {images_placeholder: images, phase_train_placeholder: False}
sess.run(embeddings, feed_dict=feed_dict)


# Run Benchmark
times_to_run = 1000
time1 = time.clock()
for x in range(times_to_run):
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    sess.run(embeddings, feed_dict=feed_dict)

time2 = time.clock()
elapsed_time = (time2 - time1)


# Output results
average_time = elapsed_time / times_to_run
print(" - Face embeddings(With FaceNet): {:.4f}s ({:.2f} fps)".format(average_time, 1 / average_time))