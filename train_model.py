import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Load your dataset (make sure 'data.csv' is in the same directory as this script)
data = pd.read_csv('data.csv')

# Inspect the column names to ensure they are correct
print("Columns in the dataset:", data.columns)

# Clean column names (if necessary)
data.columns = data.columns.str.strip()  # This removes any extra spaces in column names

# Prepare the features (X) and target variable (y)
# Add 'Mileage', 'Brand', 'EngineType', 'Age' as features for training
X = data[['Mileage', 'Brand', 'EngineType', 'Age']]  # Features
y = data['MSRP']  # Target (Price)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate the model (optional)
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Save the trained model to a pickle file (model.pkl)
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model training complete. The model is saved as 'model.pkl'.")