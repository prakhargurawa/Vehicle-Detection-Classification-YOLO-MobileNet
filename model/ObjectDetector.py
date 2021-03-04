# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 20:10:22 2021

@author: prakh
"""
# Import necessary libraries
from yolo import YOLO
import numpy as np
import cv2
from model.Utility import Car
from model.CarClassifier import CarClassifier

class ObjectDetector:
    def __init__(self):
        print("<<::::: Initializing YOLO Model :::::>>")
        self.yolo = YOLO()
        print("<<::::: Initializing YOLO Model Completed :::::>>")
        self.carClassifier = CarClassifier()
        self.detectedCars = dict()
        
    def detection(self,queue):
        for frame in queue[:10]:
            frame_no,image = frame.get_frame_no(),frame.get_image()
            image,position_list = self.yolo.detect_image(image)
            np_image = np.asarray(image)
            queue = list()
            for position in position_list:
                if position["class"] == "car":
                    prediction = self.carClassifier.classify_car(image,position)
                    position = (position["left"] + 10,position["top"] + 10)
                    cv2.putText(np_image,str(prediction),position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2) #font stroke
                    car = Car(position,prediction)
                    queue.append(car)
            self.detectedCars[frame_no] = queue
            
            result = np.asarray(np_image)
            cv2.imshow("Output Video", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return self.detectedCars