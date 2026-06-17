import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load processed data
X_train = np.load("models/X_train.npy")
X_test = np.load("models/X_test.npy")

y_train = np.load("models/y_train.npy")
y_test = np.load("models/y_test.npy")


print("Data Loaded")


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(
    X_train,
    y_train
)


print("Model Training Completed")


# Prediction
predictions = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    "Model Accuracy:",
    accuracy*100,
    "%"
)


print(
    classification_report(
        y_test,
        predictions
    )
)


# Save model
joblib.dump(
    model,
    "models/model.pkl"
)


print("Model Saved Successfully")