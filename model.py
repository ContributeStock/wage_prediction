import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('salary.csv')

x = data.iloc[:,:3]
y = data.iloc[:,-1]


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x,y)

pickle.dump(regressor, open('model.pkl','wb'))