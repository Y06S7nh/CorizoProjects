# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import LabelEncoder  # For target encoding (if needed)

# Load the wine dataset (replace path with your actual CSV file)
data = pd.read_csv("C:\\Users\\ACER-PC\\Documents\\Git\\Major Project Dataset (Wine Classification).csv")

# Separate features and target variable
features = data.iloc[:, :-1]  # All columns except the last
target = data.iloc[:, -1]  # Last column

# Check for missing values in the target variable
print(target.isnull().sum())

# Handle missing values (example: mean imputation)
if target.isnull().sum() > 0:
  imputer = SimpleImputer(strategy='mean')
  target_imputed = imputer.fit_transform(target.values.reshape(-1, 1))
  target = target_imputed

# Check if target variable needs encoding (replace with your checks)
if pd.api.types.is_numeric_dtype(target):
  from sklearn.preprocessing import LabelEncoder

  # Encode the target variable
  encoder = LabelEncoder()
  target_encoded = encoder.fit_transform(target)
else:
  target_encoded = target  # No encoding if already categorical

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

# Create a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
predictions = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")

# Print the predicted classes in a clear format
for i in range(len(predictions)):
  print(f"Predicted Class for Sample {i+1}: {predictions[i]}")

# Now you can use the trained model to predict the class of a new wine sample
# (commented out for clarity)
# new_wine = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, ...]],  # Replace ... with your new wine features
#                        columns=features.columns)
# prediction = model.predict(new_wine)
# print(f"Predicted Class: {prediction[0]}")
