# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 21:01:46 2021

@author: prakh
"""
# Code reference : https://github.com/ferhat00/Deep-Learning/blob/master/Transfer%20Learning%20CNN/Transfer%20Learning%20in%20Keras%20using%20MobileNet.ipynb

# Import necessary libraries
import keras
import numpy as np
from keras.applications import MobileNetV2
from keras.layers import Dense, Flatten,Dropout
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image
from PIL import Image 
import PIL
print(PIL.PILLOW_VERSION)

"""
Notes:
    Not used regularization as was not giving good results
    Removed Data augmentation, as was making epoch reuirement higher and model more complex
    Dropout helped a lot to reduce overfitting
    Adam provides better results than RMSProp
"""

# Reference : https://analyticsindiamag.com/a-practical-guide-to-implement-transfer-learning-in-tensorflow/
base_model = MobileNetV2(weights='imagenet',include_top=False,input_shape=(180, 180, 3))  # imports the mobilenet model and discards the last 1000 neuron layer.
# Freeze the base model
base_model.trainable = False

"""
# Regularization:
    These layers expose 3 keyword arguments:

        kernel_regularizer: Regularizer to apply a penalty on the layer's kernel
        bias_regularizer: Regularizer to apply a penalty on the layer's bias
        activity_regularizer: Regularizer to apply a penalty on the layer's output
    
    Avaialble regularizers:
        The L1 regularization penalty is computed as: loss = l1 * reduce_sum(abs(x))
        The L2 regularization penalty is computed as: loss = l2 * reduce_sum(square(x))
"""
# Reference : https://keras.io/api/layers/regularizers/ (the default value used is l1=0.01)
model = Sequential([
        base_model,
        #Dense(512, activation='relu',kernel_regularizer='l1'), 
        Dense(512, activation='relu'), 
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
])
"""
# Unfreeze the new layers 
for layer in model.layers[0:]:
    layer.trainable =  True"""
    
print("************************ LAYERS ************************")        
for layer in model.layers:
    print(layer.name,layer.trainable)
print("********************************************************")    

# Reference : https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
"""
train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,validation_split=0.2,
                                   rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True)"""

train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,validation_split=0.2)

# Create a dataset with 2 classes SUV and Sedan, the data has been scraped using google images
# and Stanford Car Dataset (https://www.kaggle.com/jutrera/stanford-car-dataset-by-classes-folder?select=anno_train.csv or https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
train_data_dir = 'D:\\NUIG-2\\Research-Topics-AI\\Assignment1\keras-yolo3\\dataset'

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(180,180),
        batch_size=64,
        class_mode='binary',
        subset='training',
        shuffle=True)

validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(180, 180),
        batch_size=64,
        class_mode='binary',
        subset='validation',
        shuffle=True) 
print(model.summary())

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are take into account

# Selection of optimizer : https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e
# Adam is the best among the adaptive optimizers in most of the cases.

# Adam optimizer,loss function will be binary cross entropy,evaluation metric will be accuracy
algo = "Adam"
model.compile(optimizer = keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Uncomment below to use RMSProp
# algo = "RMSProp"
# model.compile(optimizer= keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
"""
NOTE :
    There might be some isuue like:
            "cannot identify image file %r" % (filename if filename else fp)
            PIL.UnidentifiedImageError: cannot identify image file <_io.BytesIO object at 0x00000214A7DAE678>
    To resolve this use below command inside your environment:
    # pip install --upgrade Pillow  <-- https://github.com/ContinuumIO/anaconda-issues/issues/10737
    # uninstalling pillow installed using conda and re-installing using pip works    
"""
step_size_train=train_generator.n//train_generator.batch_size
step_size_val=validation_generator.n//validation_generator.batch_size
ep = 20

history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data=validation_generator,
                   validation_steps=step_size_val,
                   epochs=ep) 

# Saving the model 
model.save("saved_models\Mobilenet_"+str(algo)+"_"+str(ep)+"epoch_car_classification")

# https://www.kaggle.com/vasantvohra1/transfer-learning-using-mobilenet
# Plotting the progress
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()

