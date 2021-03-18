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
import xlwt 

class VideoReader:
    def __init__(self,videopath):
        self.videopath = videopath
        self.objectDetector = ObjectDetector() # Object of ObjectDetector (Which detects cars and then clasify them in SUV or Sedan)
        self.frame_no = 1
        self.queue = list()
        
    def process(self):
        # Read the clip
        carVideo = cv2.VideoCapture(self.videopath)
        carVideo.set(cv2.CAP_PROP_FPS,30)
        fps = carVideo.get(cv2.CAP_PROP_FPS)
        print("FPS : ",fps)
        print("<<::::: DISPLAYING ORIGINAL VIDEO :::::>>")
        while True:
            ret, frame = carVideo.read()
            if not ret:
                break
            # Code reference: yolo.py => detect_video
            image = Image.fromarray(frame)
            self.queue.append(Frame(self.frame_no,image))            
            image = np.asarray(image)
            # https://pythonexamples.org/python-opencv-write-text-on-image-puttext/
            position = (10,20)
            cv2.putText(image,"Original Video",position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) #font stroke
            position = (10,50)
            cv2.putText(image,"Frame : "+str(self.frame_no),position,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2) #font stroke
            result = np.asarray(image)
            cv2.imshow("Original Video", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.frame_no = self.frame_no + 1
    
        carVideo.release()
        cv2.destroyAllWindows()
        print("<<::::: DISPLAYING ORIGINAL VIDEO EXIT :::::>>")
        # Initiate pipeline which first detect objects and then classify
        self.pipeline_initiate()
        
    def pipeline_initiate(self):
        print("<<::::: VEHICLE DETECTION AND CLASSIFICATION STARTED :::::>>")
        detectedCars = self.objectDetector.detection(self.queue)
        print("<<::::: VEHICLE DETECTION AND CLASSIFICATION ENDED :::::>>")
        print("<<::::: STATISTICS AND RESULTS GENERATION STARTED :::::>>")

        # Reference : https://www.geeksforgeeks.org/writing-excel-sheet-using-python/
        workbook = xlwt.Workbook()  # Create a excel workbook
        sheet = workbook.add_sheet("Count of cars per frame") 
        # Specifying style 
        style = xlwt.easyxf('font: bold 1') 
        # Specifying column 
        sheet.write(0, 0, 'Frame', style) 
        sheet.write(0, 1, 'Sedan', style) 
        sheet.write(0, 2, 'SUV', style) 
        sheet.write(0, 3, 'Total', style) 
        # Insert Frame number, number of sedan cars, number of SUV cars and number of total cars
        for k in detectedCars:
            #print("\nFrame no : ",k)
            countSedan,countSUV = 0,0
            for car in detectedCars[k][0]:
                carType = car.get_carType()
                if carType == "Sedan":
                    countSedan = countSedan + 1
                else:
                    countSUV = countSUV + 1 
            sheet.write(k, 0, str(k)) 
            sheet.write(k, 1, str(countSedan))
            sheet.write(k, 2, str(countSUV)) 
            sheet.write(k, 3, len(detectedCars[k][0])) 
        # save the workbook             
        workbook.save("Car_Results.xls") 
        print("<<::::: Statistics and result generation ended :::::>>")
                
                
if __name__ == "__main__":
    print("<<::::: PROCESS OF VIDEO READING STARTED :::::>>")
    # Create object of VideoReader that initaiaite the pipeline (Video Reading -> Object Detection using YOLO 
    # -> Car classification using Mobilenet Model trained using transfer learning techniques)
    VideoReader('test/assignment-clip.mp4').process()
    print("<<::::: TERMINATING PROCESS :::::>>")