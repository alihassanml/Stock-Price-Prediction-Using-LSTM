import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import pickle


df = pd.DataFrame(pd.read_csv('BTC-Data.csv'))
model = tf.keras.models.load_model('model.h5')
scalar_X = pickle.load(open('scalar_X.pkl','rb'))
scalar_y = pickle.load(open('scalar_y.pkl','rb'))



data = np.array([[65180.6640625,66480.6953125,64852.9921875,32058813449]])

train = scalar_X.transform(data)
y_pred = model.predict(train)
y_pred = y_pred.reshape(-1,1)
y_pred_original = scalar_y.inverse_transform(y_pred)
print('Model Predict ',y_pred_original[:-1])