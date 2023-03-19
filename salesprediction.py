
# Importing **Libraries**


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Data Collecting and Processing

# Load data
train_df = pd.read_csv('/content/Train.csv')
test_df = pd.read_csv('/content/Test.csv')

# first 5 rows of the dataframe
train_df.head()

# number of data points & number of features
train_df.shape

# getting some information about thye dataset
train_df.info()

"""### Categorical Features:
- Item_Identifier
- Item_Fat_Content
- Item_Type
- Outlet_Identifier
- Outlet_Size
- Outlet_Location_Type
- Outlet_Type
"""

# checking for missing values
train_df.isnull().sum()

train_df.describe()

sns.set()

# Item_Weight distribution
plt.figure(figsize=(6,6))
sns.distplot(train_df['Item_Weight'])
plt.show()

# Item Visibility distribution
plt.figure(figsize=(6,6))
sns.distplot(train_df['Item_Visibility'])
plt.show()

# Item MRP distribution
plt.figure(figsize=(6,6))
sns.distplot(train_df['Item_MRP'])
plt.show()

# Item_Outlet_Sales distribution
plt.figure(figsize=(6,6))
sns.distplot(train_df['Item_Outlet_Sales'])
plt.show()

# Outlet_Establishment_Year column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=train_df)
plt.show()

# Item_Fat_Content column
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=train_df)
plt.show()

# Item_Type column
plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=train_df)
plt.show()

# Outlet_Size column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=train_df)
plt.show()

"""# Preprocessing"""

# Preprocessing
train_df['Item_Weight'].fillna(train_df['Item_Weight'].mean(), inplace=True)
test_df['Item_Weight'].fillna(test_df['Item_Weight'].mean(), inplace=True)

train_df['Outlet_Size'].fillna('Unknown', inplace=True)
test_df['Outlet_Size'].fillna('Unknown', inplace=True)

train_df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)
test_df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)

print(train_df.columns)

"""# **One-hot encoding**

"""

train_df = pd.get_dummies(train_df, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
test_df = pd.get_dummies(test_df, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])

"""# Split into train and validation sets

"""

X = train_df.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis=1)
y = train_df['Item_Outlet_Sales']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""# Define XGBoost model

"""

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, seed=42)

"""# Train the model

"""

xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric='rmse', eval_set=[(X_train, y_train), (X_valid, y_valid)])

"""# Make predictions on test data

"""

X_test = test_df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)
test_df['Item_Outlet_Sales'] = xgb_model.predict(X_test)

"""# Save predictions to file

"""

test_df[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']].to_csv('submission.csv', index=False)

"""# Calculate and print metrics

"""

from sklearn.metrics import mean_absolute_error, r2_score

y_valid_pred = xgb_model.predict(X_valid)
mse = mean_squared_error(y_valid, y_valid_pred)
mae = mean_absolute_error(y_valid, y_valid_pred)
r2 = r2_score(y_valid, y_valid_pred)

# Print metrics
print('MSE: {}, MAE: {}, R-squared: {}'.format(mse, mae, r2))