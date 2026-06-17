import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("defect_data.csv")

X = data[["Temperature", "Speed", "Pressure"]]
y = data["Defect"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)

print("Model Accuracy:", accuracy * 100, "%")

new_products = pd.DataFrame(
    [
        [50, 120, 30],
        [65, 150, 45],
        [85, 190, 65],
        [100, 220, 80]
    ],
    columns=["Temperature", "Speed", "Pressure"]
)

predictions = model.predict(new_products)

new_products["Prediction"] = predictions
new_products["Result"] = new_products["Prediction"].map({
    0: "Good Product",
    1: "Defective Product"
})

print("\nProduct Predictions:")
print(new_products)

new_products.to_csv("defect_prediction_results.csv", index=False)

print("\nResults saved to defect_prediction_results.csv")