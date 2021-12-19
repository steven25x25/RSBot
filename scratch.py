'''
from yolov3 import yolov3
from PIL import Image
import pyautogui as pag
import time
import matplotlib.pyplot as plt

cfg = "config/yolov3-tiny-rs.cfg"
weights = "weights/yolov3-tiny-rs.weights"
classes = "config/rs.names"

#image_path = "images/3.png"
#image = Image.open(image_path)
image_path = "screen"




print("Detecting in " + image_path)

#yolov3_detect(image_path, cfg, classes, weights)
yolo = yolov3(cfg, classes, weights, nms_thres=0.8)

while True:
    screen = pag.screenshot()
    print("Performing detection...", end="\r")
    output = yolo(screen, save_image_plot=True)


    if output:
        print("Detection complete. Number of detections: " + str(len(output)), end = "\r")
    else:
        print("No detections")
    

    time.sleep(0.1)
'''
'''
from PIL import Image
from math import sqrt

dp_thres = 120

def image_thres_matrix(img, thres):
    pixels = list(img.getdata())
    pixels = [int(sqrt(v[0]**2 + v[1]**2 + v[2]**2)) for v in pixels]
    pixels = [(lambda v: 1 if v > dp_thres else 0)(val) for val in pixels]
    rolling_sum = 0
    for val in pixels: rolling_sum += val

    above_density = rolling_sum / len(pixels)

    pixels = [pixels[i * img.width : (i+1) * img.width] for i in range(img.height)]

    #for row in pixels: print(row)
    print(str((above_density)))
    
print("Evaluating images...")

file_paths = ["colored_tree.jpg"] 

for file_path in file_paths:

    img = Image.open(file_path)

    image_thres_matrix(img, dp_thres)
    img.close()
'''

'''
import pyautogui
import cv2
import numpy as np
from yolov3 import yolov3
import math
import random


from pynput import keyboard

import time



COLOR_THRESHOLD = []
color_thres_list = []

with open("config/color_mask_config.cfg") as color_mask_config:
    color_thres_list = color_mask_config.read()
    color_thres_list = color_thres_list.split("\n")

for object_id in color_thres_list:
    COLOR_THRESHOLD.append(object_id.split(","))

COLOR_THRESHOLD.pop()


#def color_mask_detect(frame):
    #detections = []
sieved_pixels = []
color_filter = []
##
unrolled_frame = []


frame = pyautogui.screenshot()

for object_type in COLOR_THRESHOLD:


    unrolled_frame = np.array(frame)
    unrolled_frame = unrolled_frame.ravel()

    mask_color = []
    for color in object_type[1:4]:
        mask_color.append(int(color))

    color_filter = np.full((frame.height, frame.width,3), mask_color)
    color_filter = color_filter.ravel()
    

    sieved_pixels = np.square(np.subtract(color_filter, unrolled_frame))
        
    sieved_pixels = sieved_pixels.reshape(frame.height,frame.width,3)

    cv2.imwrite("fitlered.png",sieved_pixels)
    #cv2.
 #   return sieved_pixels
    #For pixel in frame:
        

#detections = color_mask_detect(pyautogui.screenshot())

'''


import cv2
import numpy as np 

image = cv2.imread("demo.jpg")

det = cv2.SimpleBlobDetector_create()
kp = det.detect(image)

kp_image = cv2.drawKeypoints(image, kp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

cv2.imshow("Keypoints", kp_image)
cv2.waitKey(0)