import numpy as np
import pandas as pd
import math
import re

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

location = 'Kolkata'
Tea_type = 'LEAF_AND_ALL_DUST'

tea_data = pd.read_csv('input/WEEKLY_AVERAGE_PRICES_OF_CTC_' + Tea_type + '_SOLD_AT_INDIAN_AUCTION_DURING_2021_2023_lag_data.csv')
tea_data.rename(columns = {"Week Ending/Date":"Date"}, inplace = True)
column_list = [s for s in tea_data.columns if location in s]
column_list = ["Date"] + column_list
data_location  = tea_data[column_list]

weather_data = pd.read_csv("input/"+location+'_data_filtered.csv')
weather_data.rename(columns = {"date":"Date"},inplace = True)

data = data_location.merge(weather_data,on = "Date",how = "inner")
data.dropna(inplace = True)

print(data.shape)
print(weather_data.shape)

df = data.copy()
# Split df into X and y
y = df[location]
X = df.drop([location, "Date"], axis=1)
X.shape


# Train-test split
X_train= X.iloc[:110]
X_test= X.iloc[110:].reset_index(drop=True)

Y_train= y.iloc[:110]
Y_test= y.iloc[110:].reset_index(drop=True)



correlation = pd.DataFrame(data.corr())
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
    return col_corr
corr_feature = correlation(X_train, 0.6)

data.corr()

X_train

X_train.drop(labels = corr_feature, axis = 1, inplace = True)
X_test.drop(labels = corr_feature, axis = 1, inplace = True)

# Scale X with a standard scaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)



# def calculate_vif(data_frame):
#     vif_data = pd.DataFrame()
#     vif_data["Variable"] = data_frame.columns
#     vif_data["VIF"] = [variance_inflation_factor(data_frame.values, i) for i in range(data_frame.shape[1])]
#     return vif_data



# vif_data = pd.DataFrame(calculate_vif(X_train))
# vif_data = vif_data.sort_values('VIF')


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Neural Network": MLPRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()#,
#     "XGBoost": XGBRegressor(),
#     "LightGBM": LGBMRegressor(),
#     "CatBoost": CatBoostRegressor(verbose=0)
}

for name, model in models.items():
    model.fit(X_train, Y_train)
    print(name + " trained.")

for name, model in models.items():
    print(name + " R^2 Score: {:.5f}".format(model.score(X_test, Y_test)))

for name, model in models.items():
    plt.plot(model.predict(X_test),color='red')
    plt.plot(Y_test,color='green')
    plt.show()


