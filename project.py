import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("machine_data.csv")

X = data[["Temperature", "Vibration", "Pressure"]]
y = data["Failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy * 100, "%")

new_machines = pd.DataFrame(
    [
        [50, 2, 30],
        [72, 5, 52],
        [88, 8, 68],
        [100, 10, 85]
    ],
    columns=["Temperature", "Vibration", "Pressure"]
)

new_predictions = model.predict(new_machines)

new_machines["Prediction"] = new_predictions
new_machines["Result"] = new_machines["Prediction"].map({
    0: "Machine Operating Normally",
    1: "Machine Failure Likely"
})

print("\nNew Machine Predictions:")
print(new_machines)

new_machines.to_csv("prediction_results.csv", index=False)

print("\nResults saved to prediction_results.csv")