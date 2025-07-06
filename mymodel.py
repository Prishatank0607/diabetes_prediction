import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
df = pd.read_csv("diabetes.csv")  # Make sure this file is in your folder

# Separate features and target
X = df.drop("Outcome", axis=1)  # Features
y = df["Outcome"]               # Target

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initializing and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model has been trained and saved as 'diabetes_model.pkl'")