#!python3
"""
Python 3 wrapper for single class object detection using darknet

original: https://github.com/AlexeyAB/darknet/blob/master/build/darknet/x64/darknet.py

@author: Lauris Pazans
@date: 20181021
"""

import cv2
from darknet.DarknetInterface import *
import numpy as np

def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr


class DarknetImplementation:
    classes = 1
    def __init__(self, configPath, weightPath, batchSize = 1):
        self.net = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, batchSize)  # batch size = 1
        
    #Wrapper around detect
    def performDetect(self, image, thresh=0.25):
        detections = self.detect(self.net, image, thresh)
        return detections

    def performDetectBatch(self, images, thresh=0.25, batch_size=4):
        self.detect_batch(self.net, images, batch_size, thresh)
        return

    def detect_batch(self, net, images, batch_size=4, thresh=.5, hier_thresh=.5, nms=.45):
        image_arrays = []
        for image in images:
            # DarkNet resizes the image internally(slow, should speed it up!)
            im, arr = array_to_image(image)
            image_arrays.append(arr)

        images = np.concatenate(image_arrays).ravel().ctypes.data_as(POINTER(c_float))
        network_predict(net, images)


        #assign the neccessary variables
        num = c_int(0) #num will hold the number of detections
        pnum = pointer(num)
        #batch_detections = get_network_boxes_batch(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
        return



    def detect(self, net, image, thresh=.5, hier_thresh=.5, nms=.45):
        #prep the image
        custom_image = cv2.resize(image, (lib.network_width(net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
        #custom_image = image

        #load the image
        im, arr = array_to_image(custom_image)
        
        #assign the neccessary variables
        num = c_int(0) #num will hold the number of detections
        pnum = pointer(num)
        
        #do the prediction
        #predict_image(net, im)  # Slow resizing function, we pass a pre-resized image
        network_predict(net, im.data)


        #this is a wrapper around the C function
        detections = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0) 
        
        num = pnum[0]
        if nms:
            do_nms_sort(detections, num, self.classes, nms)
        
        res = []
        for j in range(num):
            for i in range(self.classes):
                if detections[j].prob[i] > 0:
                    b = detections[j].bbox
                    
                    img_height = custom_image.shape[0]
                    img_width = custom_image.shape[1]
                    rel_center_x = b.x/img_height
                    rel_center_y = b.y/img_width
                    rel_width = b.w/img_width
                    rel_height = b.h/img_height
                    
                    res.append((detections[j].prob[i], (rel_center_x, rel_center_y, rel_width, rel_height)))

        free_detections(detections, num)

        #return the results
        return res