import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

df = pd.read_csv(r"dataset\machine_data.csv")

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

np.save("models/X_train.npy", X_train)
np.save("models/X_test.npy", X_test)
np.save("models/y_train.npy", y_train)
np.save("models/y_test.npy", y_test)

print("Preprocessing Completed Successfully")