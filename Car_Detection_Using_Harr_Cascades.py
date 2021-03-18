# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 03:14:04 2021

@author: prakh
"""
# Import necessary libraries
import numpy as np
import cv2
cv2.__version__
    
"""
NOTE:
    This is an outdated method to detect cars in video. 
    We will use YOLO as it gives better results by better detecting cars.
    This implementation is just for reference and understanding.
"""
# Read the assignment clip
carVideo = cv2.VideoCapture('test/assignment-clip.mp4')

# https://gist.github.com/199995/37e1e0af2bf8965e8058a9dfa3285bc6
cars_cascade = cv2.CascadeClassifier('test/cars.xml')

# https://kalebujordan.com/real-time-vehicle-detection-with-opencv-in-10-minutes/
def detect_cars(frame):
    cars = cars_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
    return frame

while carVideo.isOpened():
    ret, frame = carVideo.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        break
    else :        
        cars_frame = detect_cars(frame)
        cv2.imshow('frame', cars_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


carVideo.release()
cv2.destroyAllWindows()