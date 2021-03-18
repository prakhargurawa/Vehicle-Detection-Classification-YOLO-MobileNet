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
from concurrent.futures import ThreadPoolExecutor
import time
from multiprocessing import Process
from multiprocessing import Pool

class ObjectDetector:
    def __init__(self):
        print("<<::::: Initializing YOLO Model :::::>>")
        self.yolo = YOLO()
        print("<<::::: Initializing YOLO Model Completed :::::>>")
        self.carClassifier = CarClassifier()
        self.detectedCars = dict()
        
                
    def detection_and_classification_task(self,frame_no,image):
        # This function does same job to call car classifier but is used by executor framework
        image,position_list = self.yolo.detect_image(image)
        q = list()
        i = 0
        for position in position_list:
            if position["class"] == "car": # Only useful classes for our case is Car
                prediction = self.carClassifier.classify_car(image,position,frame_no,i)
                i = i+1
                position = (position["left"] + 10,position["top"] + 10)
                car = Car(position,prediction)
                q.append(car)
            self.detectedCars[frame_no] = (q,image)
                  
    def detection(self,queue):
        # Object Detection is done using YOLO or TinyYOLO with is trained on COCO Dataset which has 80 classes 
        # can find list of classes from the file model_data/coco_classes
        # APPROCH 1: Standard implementation of car detection and classification 
        start_time = time.time()
        for frame in queue:
            frame_no,image = frame.get_frame_no(),frame.get_image()
            image,position_list = self.yolo.detect_image(image) # detect the objects using already trained YOLO 
            image_size = image.size
            q,i = list(),0 
            for position in position_list:
                if position["class"] == "car": # Only useful classes for our case is Car
                    prediction = self.carClassifier.classify_car(image,position,frame_no,i) # use car classifier which contained our ML Model 
                    i=i+1
                    position = (position["left"] + 10,position["top"] + 10)
                    car = Car(position,prediction)
                    q.append(car)
            self.detectedCars[frame_no] = (q,image)
        print("Standard implementation time for car detection and classification task : %s seconds" % (time.time() - start_time))
        ##########################
        """
        # APPROCH 2: Optimized implementation of car detection and classification with multithreading/thread pool
        start_time = time.time()
        executor = ThreadPoolExecutor(100)
        for frame in queue:
            frame_no,image = frame.get_frame_no(),frame.get_image()
            image_size = image.size
            executor.submit(self.detection_and_classification_task(frame_no,image))
        print("--- Optimized implementation time for car detection and classification task : %s seconds ---" % (time.time() - start_time))
        """
        ##########################  
        # Save the output video with proper marking and Car types 
        # Reference : https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
        video = cv2.VideoWriter('Output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, image_size)
        for i in range(1,len(queue)+1):
            q,image = self.detectedCars[i]
            np_image = np.asarray(image)
            for car in q:
                pos,pred = car.get_position(),car.get_carType()
                color = (0,0,255)
                if pred=="SUV":
                    color = (0,170,230) 
                cv2.putText(np_image,str(pred),pos,cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
                position = (10,20)
                cv2.putText(np_image,"Result Video",position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) #font stroke
                position = (10,40)
                cv2.putText(np_image,"Frame : "+str(i),position,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2) #font stroke
                position = (10,60)
                cv2.putText(np_image,"Count Car : "+str(len(q)),position,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2) #font stroke
            result = np.asarray(np_image)
            video.write(result)
        video.release()
            
        # Displaying Saved Video to user once
        cap = cv2.VideoCapture('Output.avi')
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Display the resulting frame
                cv2.imshow('Frame',frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break

        # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        return self.detectedCars



