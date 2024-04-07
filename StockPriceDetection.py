# Import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Replace with your chosen model

# Define the file path (replace with the actual location of your CSV file)
data_path = r"C:\Users\ACER-PC\Documents\Git\Minor Project Data set (Stock Price Prediction).csv"

# Read the stock data
try:
  data = pd.read_csv(data_path, index_col="Date", parse_dates=True)
except FileNotFoundError:
  print("Error: File not found. Please check the data path.")
  exit()  # Exit the script if file not found

# Print the column names for reference (optional)
print(data.columns)  # This helps you verify the column names

# Select relevant features (assuming the first three are features)
features = list(data.columns)[:-1]  # Select all columns except the last one (assuming target)
target = data.columns[-1]  # Assuming the last column is the target variable

# Preprocess data (handle missing values, scaling)
# ... (add your data preprocessing steps here)

# Scale data (optional, but recommended for some models)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[features])

# Split data into training and testing sets (exclude last 10 days from training)
X_train, X_test, y_train, y_test = train_test_split(data_scaled[:-10], data[target][:-10], test_size=0.2)

# Define and train your model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future prices for the next 10 days
future_days = 10
future_data = data[features].iloc[-future_days:]
future_data_scaled = scaler.transform(future_data)  # If you used scaling
future_prices = model.predict(future_data_scaled)

# Print predicted prices
print(f"Predicted prices for the next {future_days} days:")
for i in range(future_days):
  print(f"Day {i+1}: {future_prices[i]}")

# Evaluate model performance (optional)
# ... (add your model evaluation metrics here, but keep in mind limitations of predicting future)
