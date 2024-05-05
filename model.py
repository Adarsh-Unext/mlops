import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
titanic_data = pd.read_csv('titanic.csv')

# Preprocess the data
titanic_data.dropna(inplace=True)  # Drop rows with missing values for simplicity
X = titanic_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


import joblib

# Save the trained model to a file
joblib.dump(model, 'titanic_model.pkl')

# Later, you can load the model back using:
# loaded_model = joblib.load('titanic_model.pkl')
print(model.predict(X_test_scaled[0].reshape(1, -1)))