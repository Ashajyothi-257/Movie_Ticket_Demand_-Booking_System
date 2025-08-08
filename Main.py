import pandas as pd
import numpy as np
import pickle
import time
import json
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.read_csv("data.csv")
logger.info("✅ Data loaded successfully.")

df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["day_of_week"] = df["date"].dt.weekday + 1
df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(int)

df["price_per_seat"] = df["ticket_price"] / df["capacity"]

def map_show_time(hour):
    if 0 <= hour < 12:
        return 1
    elif 12 <= hour < 17:
        return 2
    elif 17 <= hour < 21:
        return 3
    else:
        return 4

df["show_time"] = df["show_time"].apply(map_show_time)

df["film_code"] = df["film_code"].astype(str)
df["cinema_code"] = df["cinema_code"].astype(str)
df = pd.get_dummies(df, columns=["film_code", "cinema_code"], drop_first=True)

features = [
    "day", "month", "quarter", "day_of_week", "show_time",
    "ticket_price", "capacity", "is_weekend", "price_per_seat"
] + [col for col in df.columns if col.startswith("film_code_") or col.startswith("cinema_code_")]

target = "tickets_sold"
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.8, 1],
    "colsample_bytree": [0.8, 1],
}

logger.info("🔍 Hyperparameter tuning...")
search = RandomizedSearchCV(
    XGBRegressor(random_state=42),
    param_distributions=params,
    n_iter=10,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=0,
    n_jobs=-1
)
search.fit(X_train, y_train)
model = search.best_estimator_

cv_scores = cross_val_score(model, X, y, scoring='r2', cv=5)
mean_cv_score = np.mean(cv_scores)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

start_time = time.time()
_ = model.predict([X_test.iloc[0]])
response_time_ms = (time.time() - start_time) * 1000

print("\n✅ Model trained!")
print(f"📊 RMSE: {rmse:.2f}")
print(f"📊 MAE: {mae:.2f}")
print(f"📊 R²: {r2:.4f}")
print(f"📈 CV R²: {mean_cv_score:.4f}")
print(f"⚡ Prediction Time: {response_time_ms:.3f} ms")

model_output_path = "ticket_model.pkl"
with open(model_output_path, "wb") as f:
    pickle.dump({"model": model, "features": features}, f)
logger.info(f"📦 Model saved to {model_output_path}")

metadata = {
    "version": "v1.0",
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "cross_val_r2": mean_cv_score,
    "response_time_ms": response_time_ms,
    "features": features,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open("ticket_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
logger.info("📝 Metadata saved.")

with open("model_features.json", "w") as f:
    json.dump(features, f, indent=4)
logger.info("📄 model_features.json saved.")

try:
    from xgboost import plot_importance
    plot_importance(model, importance_type="gain", max_num_features=10)
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
except Exception as e:
    logger.warning(f"⚠️ Could not plot feature importances: {e}")
