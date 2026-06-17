import numpy as np
import joblib


# Load saved model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


print("SMART FACTORY AI PREDICTION SYSTEM")
print("--------------------------------")


# User input

temperature = float(input("Enter Air Temperature (K): "))
process_temp = float(input("Enter Process Temperature (K): "))
speed = float(input("Enter Rotational Speed (rpm): "))
torque = float(input("Enter Torque (Nm): "))
wear = float(input("Enter Tool Wear (min): "))


# Create input array

machine_data = np.array([[
    temperature,
    process_temp,
    speed,
    torque,
    wear
]])


# Scale input

machine_data = scaler.transform(machine_data)


# Prediction

prediction = model.predict(machine_data)


probability = model.predict_proba(machine_data)


failure_probability = probability[0][1] * 100


print("\nRESULT")
print("----------------")


print(
    f"Failure Probability: {failure_probability:.2f}%"
)


if prediction[0] == 1:

    print("Status: ⚠ MACHINE FAILURE RISK")

    print(
        "Recommendation: Schedule Maintenance"
    )

else:

    print("Status: ✅ MACHINE HEALTHY")

    print(
        "Recommendation: Continue Operation"
    )