# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 20:01:54 2021

@author: prakh
"""
# Import necessary libraries
import numpy as np
import cv2
from PIL import Image
from model.ObjectDetector import ObjectDetector
from model.Utility import Frame

class VideoReader:
    def __init__(self,videopath):
        self.videopath = videopath
        self.objectDetector = ObjectDetector()
        self.frame_no = 1
        self.queue = list()
        
    def process(self):
        # Read the assignment clip
        carVideo = cv2.VideoCapture(self.videopath)
        carVideo.set(cv2.CAP_PROP_FPS, 30)
        fps = carVideo.get(cv2.CAP_PROP_FPS)
        print("FPS : ",fps)
        print("<<::::: Displaying Original Video Entry :::::>>")
        while True:
            ret, frame = carVideo.read()
            if not ret:
                break
            # Code reference: yolo.py => detect_video
            image = Image.fromarray(frame)
            self.queue.append(Frame(self.frame_no,image))            
            image = np.asarray(image)
            # https://pythonexamples.org/python-opencv-write-text-on-image-puttext/
            position = (10,10)
            cv2.putText(image,"Original Video",position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3) #font stroke
            position = (10,50)
            cv2.putText(image,"Frame : "+str(self.frame_no),position,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),3) #font stroke
            result = np.asarray(image)
            cv2.imshow("Original Video", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.frame_no = self.frame_no + 1
    
        carVideo.release()
        cv2.destroyAllWindows()
        print("<<::::: Displaying Original Video Exit :::::>>")
        # Initiate pipeline which first detect objects and then classify
        self.pipeline_initiate()
        
    def pipeline_initiate(self):
        print("<<::::: VEHICLE DETECTION AND CLASSIFICATION STARTED :::::>>")
        detectedCars = self.objectDetector.detection(self.queue)
        
        for k in detectedCars:
            print("\nFrame no : ",k)
            for car in detectedCars[k]:
                print(car.get_carType())
                
                
            
        
if __name__ == "__main__":
    print("<<::::: PROCESS OF VIDEO READING STARTED :::::>>")
    VideoReader('test/assignment-clip.mp4').process()