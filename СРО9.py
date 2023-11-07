import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("numeric_data.csv")

X = data.drop("Severity", axis=1)
y = data["Severity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ada_model = AdaBoostRegressor(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

mse_ada = mean_squared_error(y_test, y_pred_ada)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

print(f"разницу между фактическими и предсказанными значениями (AdaBoost): {mse_ada}")
print(f"разницу между фактическими и предсказанными значениями (XGBoost): {mse_xgb}")
print(f"предсказание (AdaBoost): {y_pred_ada}")
print(f"предсказание (XGBOOST): {y_pred_xgb}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_ada, label="AdaBoost", marker="o")
plt.scatter(y_test, y_pred_xgb, label="XGBoost", marker="x")
plt.xlabel("True Severity")
plt.ylabel("Predicted Severity")
plt.legend()
plt.title("Comparing AdaBoost and XGBoost Models")
plt.show()



