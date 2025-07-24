# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

# Load dataset from the specified path
dataset_path = r"D:\Bhawnesh\ass 1\Automobile.csv"
column_names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
                'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 
                'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 
                'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

df = pd.read_csv(dataset_path, names=column_names)

# Data Cleaning
df.replace("?", np.nan, inplace=True)  # Replace missing values represented by '?'
df.dropna(inplace=True)  # Dropping rows with missing values

# Converting relevant columns to appropriate data types
df['price'] = df['price'].astype(float)
df['horsepower'] = df['horsepower'].astype(float)

# Label Encoding categorical columns to convert them into numeric format
le = LabelEncoder()
categorical_columns = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 
                       'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Splitting the dataset into features and target
X = df[['horsepower', 'engine-size', 'curb-weight', 'city-mpg', 'highway-mpg']]
y = df['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

### Linear Regression Model ###
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

# Calculating MSE and R2 score for Linear Regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f"\nLinear Regression - MSE: {mse_linear}, R2 Score: {r2_linear}")

### Polynomial Regression Model ###
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)

X_test_poly = poly.transform(X_test)
y_pred_poly = poly_model.predict(X_test_poly)

# Calculating MSE and R2 score for Polynomial Regression
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"\nPolynomial Regression - MSE: {mse_poly}, R2 Score: {r2_poly}")

### Random Forest Regressor Model ###
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Calculating MSE and R2 score for Random Forest Regressor
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"\nRandom Forest Regressor - MSE: {mse_rf}, R2 Score: {r2_rf}")

### Logistic Regression Model ###
# Logistic Regression is for classification, so let's convert the target variable into a binary classification problem
# We'll predict whether the car price is above or below the median value.

median_value = df['price'].median()
y_class = np.where(df['price'] >= median_value, 1, 0)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.3, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_class, y_train_class)

y_pred_log = log_model.predict(X_test_class)

# Calculating accuracy, precision, recall, and F1-score for Logistic Regression
accuracy_log = accuracy_score(y_test_class, y_pred_log)
precision_log = precision_score(y_test_class, y_pred_log)
recall_log = recall_score(y_test_class, y_pred_log)
f1_log = f1_score(y_test_class, y_pred_log)

print(f"\nLogistic Regression - Accuracy: {accuracy_log}, Precision: {precision_log}, Recall: {recall_log}, F1 Score: {f1_log}")
