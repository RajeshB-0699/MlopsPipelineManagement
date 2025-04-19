import pandas as pd
import numpy as np
import os
import pickle
from fastapi import FastAPI
import joblib
from sklearn.ensemble import RandomForestClassifier
from data_model import Water

app = FastAPI(title = "Water Potability Prediction", description = "Predicting Water Potability")

# model = pickle.load(open("model.pkl","rb"))

# with open("D:\MlopsPipelineManagement\model.pkl","rb") as f:
#     model = pickle.load(f)
train_processed = pd.read_csv("D:/MlopsPipelineManagement/df/processed/train_processed.csv")
test_processed = pd.read_csv('D:/MlopsPipelineManagement/df/processed/test_processed.csv')

X_train = train_processed.drop(columns=['Potability'],axis=1)
y_train = train_processed['Potability']

clf = RandomForestClassifier()
clf.fit(X_train,y_train)
model = clf
print(model)
# model = joblib.load("D:\MlopsPipelineManagement\model2.pkl")

@app.get("/")
def index():
    sample1 = input("Enter your name:")
    return {"message":f"Welcome to Water Potability Prediction using FastAPI. Pls try it out {sample1}"}

@app.post("/predict")
async def model_predict(water: Water):
    # input_ph = [float(input("Enter your pH value:"))]
    # input_Hardness = [float(input("Enter your Hardness value:"))]
    # input_Solids = [float(input("Enter your Solids value:"))]
    # input_Chloramines = [float(input("Enter your Hardness value:"))]
    # input_Sulfate = [float(input("Enter your Hardness value:"))]
    # input_Conductivity =[ float(input("Enter your Conductivity value:"))]
    # input_Organic_carbon = [float(input("Enter your Organic_carbon value:"))]
    # input_Trihalomethanes = [float(input("Enter your Trihalomethanes value:"))]
    # input_Turbidity = [float(input("Enter your Hardness value:"))]

    # sample = pd.DataFrame(
    #     {
    #         'ph': [water.ph],
    #         'Hardness': [water.Hardness],
    #         'Solids': [water.Solids],
    #         'Chloramines': [water.Chloramines],
    #         'Sulfate': [water.Sulfate],
    #         'Conductivity': [water.Conductivity],
    #         'Organic_carbon': [water.Organic_carbon],
    #         'Trihalomethanes': [water.Trihalomethanes],
    #         'Turbidity': [water.Turbidity]
    #     }
    # )
    # sample = pd.DataFrame(
    #     {
    #         'ph': input_ph,
    #         'Hardness': input_Hardness,
    #         'Solids': input_Solids,
    #         'Chloramines': input_Chloramines,
    #         'Sulfate': input_Sulfate,
    #         'Conductivity': input_Conductivity,
    #         'Organic_carbon': input_Organic_carbon,
    #         'Trihalomethanes': input_Trihalomethanes,
    #         'Turbidity': input_Turbidity
    #     }
    # )
    input_dict = water.dict()
    df_new = pd.DataFrame([input_dict])
    predicted_value = model.predict(df_new)

    if predicted_value == 1:
        return "Water is consumable"
    else:
        return "Water is not consumable"

    # return {"received":water.dict()}