import numpy as np
import pandas as pd
import math
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt

class Ml_Model():
    def __init__(self,loc='Kolkata', tea_type='LEAF_AND_ALL_DUST') -> None:
        self.location = loc
        self.tea_type = tea_type
        self.read_data()
        self.train_tst_split()
        self.normalize()
        pass
    def read_data(self):
        self.tea_data = pd.read_csv('input/WEEKLY_AVERAGE_PRICES_OF_CTC_' + self.tea_type + '_SOLD_AT_INDIAN_AUCTION_DURING_2021_2023_lag_data.csv')
        self.tea_data.rename(columns = {"Week Ending/Date":"Date"}, inplace = True)
        self.tea_data['Date'] = pd.to_datetime(self.tea_data['Date'] )  
        column_list = [s for s in self.tea_data.columns if self.location in s]
        column_list = ["Date"] + column_list
        self.data_location  = self.tea_data[column_list]

        self.weather_data = pd.read_csv('input/Kolkata_data_filtered.csv')
        self.weather_data.rename(columns = {"date":"Date"},inplace = True)
        self.weather_data['Date'] = pd.to_datetime(self.weather_data['Date'])

        self.data = self.data_location.merge(self.weather_data,on = "Date",how = "inner")
        self.data.dropna(inplace = True)

        self.df = self.data.copy()
        # Split df into X and y
        self.y = self.df[self.location]
        self.X = self.df.drop([self.location, "Date"], axis=1)
    
    def train_tst_split(self):
        # Train-test split
        self.X_train= self.X.iloc[:110]
        self.X_test= self.X.iloc[110:].reset_index(drop=True)

        self.Y_train= self.y.iloc[:110]
        self.Y_test= self.y.iloc[110:].reset_index(drop=True)

        correlation = pd.DataFrame(self.data.corr())
        self.corr_feature = self.correlation(self.X_train, 0.6)
        self.X_train.drop(labels = self.corr_feature, axis = 1, inplace = True)
        self.X_test.drop(labels = self.corr_feature, axis = 1, inplace = True)

    def correlation(self,dataset, threshold):
        col_corr = set() # Set of all the names of deleted columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] # getting the name of column
                    col_corr.add(colname)
        return col_corr
    def normalize(self):
        # Scale X with a standard scaler
        scaler = StandardScaler()
        scaler.fit(self.X_train)

        self.X_train = pd.DataFrame(scaler.transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_train.columns)


        pass
    def models(self):
        models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
        }
        out  = {}
        
        for name, model in models.items():
            model.fit(self.X_train, self.Y_train)
            # st.write(f"{name} trained.")
        st.write('Input data')
        st.dataframe(self.X_test)
        for name, model in models.items():
            pred = pd.DataFrame(model.predict(self.X_test))
            pred.columns = ['Prediction']

            act = pd.DataFrame(self.Y_test)

            print(act.columns)
            act.columns = ['Actual']
            a = pd.concat([act,pred],axis = 1)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"{name} Model Prediction")
                st.dataframe(a[['Actual','Prediction']],width=500)
                
            with col2:
                st.subheader(f"{name} Model Prediction Vs Actual")
                st.line_chart(a[['Actual','Prediction']])
            # out[f'{name}R^2 Score: '] = model.score(self.X_test, self.Y_test)
            st.write(f'{name} R^2 Score: {model.score(self.X_test, self.Y_test):.5f}')

        for name, model in models.items():
            plt.plot(model.predict(self.X_test),color='red')
            plt.plot(self.Y_test,color='green')
            plt.show()
        return out


if "__name__"=='__main__':
    Ml_Model()
