 import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  

# Load dataset (replace with real dataset link)
data = pd.read_csv("carbon_emissions.csv")  

# Select features & target
X = data[["GDP", "Energy Consumption", "Industrial Output"]]  
y = data["Carbon Emissions"]  

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Train model
model = LinearRegression()  
model.fit(X_train, y_train)  

# Make predictions
y_pred = model.predict(X_test)  

# Evaluate model
mse = mean_squared_error(y_test, y_pred)  
print(f"Mean Squared Error: {mse:.2f}")  

# Plot results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Emissions")
plt.ylabel("Predicted Emissions")
plt.title("Carbon Emissions Prediction")
plt.show()