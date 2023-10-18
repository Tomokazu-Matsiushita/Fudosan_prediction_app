import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("物件情報_2.csv", index_col=0)
import joblib

df['Price'] = np.log1p(df['Price'])
df['Space'] = np.log1p(df['Space'])
df['Belanda'] = np.log1p(df['Belanda'])
df['Walk'] = np.log1p(df['Walk'])
df['DaysAgo'] = np.log1p(df['DaysAgo'])

df = pd.get_dummies(df, columns=['Type'], prefix='Type')

df.dropna(subset=["DaysAgo"], inplace=True)
df['Belanda'].fillna(0, inplace=True)
df.dropna(inplace=True)

# Define features (X) and target (y)
#X = df[['Space', 'Latitude', 'Longitude', 'Walk', 'DaysAgo']]
#X = df[['Space', 'Latitude', 'Longitude', 'Walk']]
X = df[['Space', 'Walk', 'DaysAgo']]
y = df['Price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate the IQR for y_train
Q1 = np.percentile(y_train, 25)
Q3 = np.percentile(y_train, 75)
IQR = Q3 - Q1

# Define the upper and lower bounds to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify and remove outliers in the training set
outliers = (y_train < lower_bound) | (y_train > upper_bound)
X_train_cleaned = X_train[~outliers]
y_train_cleaned = y_train[~outliers]

# Initialize and train your machine learning model (Linear Regression)
model = LinearRegression()
model.fit(X_train_cleaned, y_train_cleaned)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model on the test set
score = model.score(X_test, y_test)
print(f'R-squared score: {score}')

# df['Price'] = np.expm1(df['Price'])
# df['Space'] = np.expm1(df['Space'])
# df['Belanda'] = np.expm1(df['Belanda'])
# df['Walk'] = np.expm1(df['Walk'])
# df['DaysAgo'] = np.expm1(df['DaysAgo'])
def load_model():
    # Load your trained machine learning model here
    model = joblib.load("fudosan_model_file.pkl")
    return model