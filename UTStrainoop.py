#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import joblib
import xgboost as xgb


# In[2]:


# Load the machine learning model and encode
model = joblib.load('Ranfor_train.pkl')
target_encoded= joblib.load('target_encoded.pkl')
meal_plane=joblib.load('meal_plan.pkl')
room_type=joblib.load('room_type.pkl')
market_segment=joblib.load('market_segment.pkl')


# In[3]:


# preparing data
class DataHandler:   
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None
    def load_data(self):
        self.data = pd.read_csv(self.file_path) 
    def create_input_output(self, booking_status):
        self.output_df = self.data[booking_status]
        self.input_df = self.data.drop(booking_status, axis=1)


# In[4]:


# model preprocessing 
class ModelPreprocessing: 
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.le = LabelEncoder()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
        self.model = RandomForestClassifier(random_state=42)
        self.y_predict = None  
    def SplitData(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split( 
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)
    def DropColumns(self, columns_to_drop):
        self.x_train = self.x_train.drop(columns=columns_to_drop, inplace=False)  # or inplace=True
        self.x_test = self.x_test.drop(columns=columns_to_drop, inplace=False)
    def ModeMeal(self, column_name):
        return self.x_train[column_name].mode()[0]
    def ImputeMode(self,column_name):
        mode = self.ModeMeal(column_name)
        self.x_train[column_name].fillna(mode, inplace=True)
        self.x_test[column_name].fillna(mode, inplace=True)
    def ImputeMedian(self, columns):
        for column in columns:
            median_value = self.x_train[column].median()
            self.x_train[column].fillna(median_value, inplace=True)
            self.x_test[column].fillna(median_value, inplace=True)
    def DropDuplicate(self):  
        duplicates = self.x_train.duplicated(keep='first')
        self.x_train = self.x_train[~duplicates]
        self.y_train = self.y_train[~duplicates]  
    def LabelEncoding(self,columns):
        for column in columns:
            self.x_train[column] = self.le.fit_transform(self.x_train[column])
            self.x_test[column] = self.le.transform(self.x_test[column])
    def OneHot(self,column_name): 
        self.x_train = pd.get_dummies(self.x_train, columns=[column_name], dtype=int)
        self.x_test = pd.get_dummies(self.x_test, columns=[column_name], dtype=int)
    def BinaryEncode(self):
        self.y_train_encoded = self.le.fit_transform(self.y_train)
        self.y_test_encoded = self.le.transform(self.y_test)
    def TrainModel(self):
        self.model.fit(self.x_train, self.y_train)
    def PredictModel(self):
        self.y_predict = self.model.predict(self.x_test)
        return self.y_predict
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['0','1']))
    def SaveModel(self, filename):
        with open(filename, 'wb') as file:  # Open the file in write-binary mode
            pickle.dump(self.model, file)
        


# In[5]:


file_path = 'Dataset_B_Hotel.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('booking_status')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_process = ModelPreprocessing(input_df, output_df)
model_process.SplitData()
model_process.DropColumns(['Booking_ID'])
mode_replace = model_process.ModeMeal('type_of_meal_plan')
model_process.ImputeMode('type_of_meal_plan')
model_process.ImputeMedian(['required_car_parking_space'])
model_process.ImputeMedian(['avg_price_per_room'])
model_process.DropDuplicate()
model_process.LabelEncoding(['type_of_meal_plan', 'room_type_reserved'])
model_process.OneHot('market_segment_type')
model_process.BinaryEncode()
model_process.TrainModel()
model_process.PredictModel()
model_process.createReport()
model_process.SaveModel('Ranfor_train.pkl') 


# In[ ]:




