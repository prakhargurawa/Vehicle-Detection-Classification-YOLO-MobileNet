# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 20:14:28 2021

@author: prakh
"""
# Import necessary libraries
import keras
from keras.preprocessing import image
import numpy as np


class CarClassifier:
    def __init__(self):
        # https://www.tensorflow.org/guide/keras/save_and_serialize
        print("<<::::: Loading PreTrained Model Weigths :::::>>")
        self.model = keras.models.load_model('saved_models/mobilenet_adam_100epoch_car_classification')
        print("<<::::: Loading PreTrained Model Weigths Completed :::::>>")
            
    def load_image(img, show=False):
        img_tensor = image.img_to_array(img)                    # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
        return img_tensor
                  
    def classify_car(self,image,position):
        cropped_image = image.crop((position["left"] + 1, position["top"] + 1, position["right"] + 1, position["bottom"] + 1))
        cropped_image = cropped_image.resize((180,180))
        cropped_image = np.asarray(cropped_image)
        img_tensor = np.expand_dims(cropped_image, axis=0)     # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        #img_tensor /= 255. 
        pred = self.model.predict(img_tensor)
        rounded = float(np.round(pred))
        if rounded == 0.0:
            return "SUV"
        elif rounded == 1.0:
            return "Sedan"