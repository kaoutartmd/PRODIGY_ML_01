import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data_train=pd.read_csv('train.csv')
data_test=pd.read_csv('test.csv')

print(data_train.head())

with open('data_description.txt','r') as s:
    description=s.read()
print(description)


# Fill missing values with the median
data_train['LotFrontage'] = data_train['LotFrontage'].fillna(data_train['LotFrontage'].median())
# Data Preprocessing: Check for missing values
print(data_train.isnull().sum())

print(data_train['SalePrice'].describe())

# Select relevant features and target variable
features = ['LotArea','OverallQual', 'GrLivArea', 'TotRmsAbvGrd', 'GarageArea', 'YearBuilt', 'FullBath','HalfBath','BsmtFullBath', 'BsmtHalfBath']
target = 'SalePrice'

X_train=data_train[features]
y_train=data_train[target]
X_test=data_test[features]

# Split the training data into training and validation sets (80% training, 20% validation)
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()
# Train the model
model.fit(X_train_split, y_train_split)
#prediction
y_pred = model.predict(X_test_split)


# Evaluate the model using Mean Squared Error (MSE)
mse=mean_squared_error(y_test_split,y_pred)
print(f'Mean Squared Error on Validation Set is : {mse}')
rmse = np.sqrt(mean_squared_error(y_test_split, y_pred))  # y_val = actual values, y_pred = model predictions
print("RMSE:", rmse)

# Visualize the predictions vs actual values
plt.scatter(y_test_split, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
data_test = data_test.dropna(subset=features)
# Now, predict house prices for the test set
y_test_pred = model.predict(data_test[features])

# Prepare the submission file (assuming test_data has an 'ID' column for the submission format)
submission = pd.DataFrame({'Id': data_test['Id'], 'PredictedPrice': y_test_pred})

# Save the predictions to a CSV file
submission.to_csv('house_price_predictions.csv', index=False)

print("Predictions saved to 'house_price_predictions.csv'.")