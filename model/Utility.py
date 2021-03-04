# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 20:12:41 2021

@author: prakh
"""

class Frame:
    def __init__(self,frame_no,image):
        self.frame_no = frame_no
        self.image = image
        
    def get_frame_no(self):
        return self.frame_no
    
    def get_image(self):
        return self.image
    
# Region of interest
class Car:
    def __init__(self,position=None,carType=None):
        self.position = position
        self.carType = carType
        
    def get_position(self):
        return self.position
    
    def get_carType(self):
        return self.carType
        
    def __str__(self):
        print("Position : ",self.position," CarType : ",self.carType,"\n")