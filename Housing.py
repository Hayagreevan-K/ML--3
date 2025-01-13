# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Generate a sample dataset (or replace this with your own dataset)
data = {
    'Size (sqft)': [1500, 2000, 2500, 3000, 3500],
    'Bedrooms': [3, 4, 3, 5, 4],
    'Location_Score': [7, 8, 9, 6, 7],  # Simplified location metric
    'Price': [300000, 400000, 500000, 600000, 700000]
}
df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df[['Size (sqft)', 'Bedrooms', 'Location_Score']]
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Predict price for a new house
new_house = [[2500, 3, 8]]  # Example: 2500 sqft, 3 bedrooms, location score of 8
new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)
print("Predicted Price for the new house:", predicted_price[0])
