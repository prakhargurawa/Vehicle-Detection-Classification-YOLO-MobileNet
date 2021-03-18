# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 11:14:18 2021

@author: prakh

This is a python script to calculate F1 score of car count,F1 score of sedan car count and F1 score of SUV car Count
"""
# Import libraries
from sklearn.metrics import f1_score,accuracy_score
import numpy as np
import pandas as pd

Ground_Truth_Results = pd.read_excel("GroundTruth.xlsx")
Ground_Truth_Results = Ground_Truth_Results.rename(columns={"Frame#":"Frame"})
Predicted_Results = pd.read_excel("Car_Results.xls")

# Toatl Cars
TotalCarTrue = Ground_Truth_Results.Total
TotalCarPredicted = Predicted_Results.Total

print("\n***********************************************************")
print("      Accuracy/F1 Score with respect to Ground Truth")
print("***********************************************************")
print("Accuracy : ",accuracy_score(TotalCarTrue,TotalCarPredicted))

# Query-1
TotalCarsScore = f1_score(TotalCarTrue, TotalCarPredicted, average='weighted')
print("F1 Score for Total Cars in each frame: ", TotalCarsScore)

#SUV Counts
TrueSUVPerFrame = Ground_Truth_Results.SUV
PredictedSUVPerFrame = Predicted_Results.SUV
TotalSUVScore = f1_score(TrueSUVPerFrame, PredictedSUVPerFrame, average='weighted')
print("F1 Score for SUV in each frame: ", TotalSUVScore)

#Sedan Counts
TrueSedanPerFrame = Ground_Truth_Results.iloc[:,[1]]
PredictedSedanPerFrame = Predicted_Results.iloc[:,[1]]
TotalSedanScore = f1_score(TrueSedanPerFrame, PredictedSedanPerFrame, average='weighted')
print("F1 Score for Sedan in each frame: ", TotalSedanScore)

