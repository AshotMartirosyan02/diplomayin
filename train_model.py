"""
Real Estate Rent Price Prediction System - Model Training Script
Generates a realistic mock dataset for Armenian real estate and trains
a RandomForestRegressor to predict rental prices in AMD.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# ── 1. Seed for reproducibility ─────────────────────────────────────────────
np.random.seed(42)
N = 500

# ── 2. Generate realistic mock dataset ──────────────────────────────────────
# District encoding:
#   1 = Kentron (Center)   – most expensive
#   2 = Arabkir             – upper-mid
#   3 = Davtashen           – mid
#   4 = Erebuni             – lower-mid
#   5 = Nor Nork            – budget

district_encoded = np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.25, 0.20, 0.20, 0.20, 0.15])

# Condition encoding:
#   0 = Old / needs renovation
#   1 = Normal
#   2 = Newly renovated / excellent
condition_encoded = np.random.choice([0, 1, 2], size=N, p=[0.20, 0.50, 0.30])

# Area in square meters (30–200 m²)
area_sqm = np.round(np.random.uniform(30, 200, size=N), 1)

# Number of rooms (1–6), loosely correlated with area
rooms = np.clip(np.round(area_sqm / 35 + np.random.normal(0, 0.5, size=N)).astype(int), 1, 6)

# Floor (1–16)
floor = np.random.randint(1, 17, size=N)

# ── 3. Construct a realistic target price (AMD) ────────────────────────────
# Base price per m²  ≈  1500 AMD/m² on average
base_price_per_sqm = 1500

# District multiplier (center is ~1.6×, budget area ~0.8×)
district_multiplier = {1: 1.60, 2: 1.25, 3: 1.00, 4: 0.85, 5: 0.75}
dist_mult = np.array([district_multiplier[d] for d in district_encoded])

# Condition multiplier
condition_multiplier = {0: 0.80, 1: 1.00, 2: 1.25}
cond_mult = np.array([condition_multiplier[c] for c in condition_encoded])

# Floor bonus: mid-floors (4-10) get a small premium
floor_bonus = np.where((floor >= 4) & (floor <= 10), 1.05, 1.00)

# Room premium: extra rooms add a flat bonus
room_premium = rooms * 8000

price_amd = (
    area_sqm * base_price_per_sqm * dist_mult * cond_mult * floor_bonus
    + room_premium
    + np.random.normal(0, 15000, size=N)   # noise
)
price_amd = np.clip(price_amd, 50_000, None).astype(int)

# ── 4. Assemble DataFrame ──────────────────────────────────────────────────
df = pd.DataFrame({
    "area_sqm": area_sqm,
    "rooms": rooms,
    "floor": floor,
    "district_encoded": district_encoded,
    "condition_encoded": condition_encoded,
    "price_amd": price_amd,
})

print("Dataset shape:", df.shape)
print(df.head(10))
print("\nDescriptive statistics:")
print(df.describe().round(2))

# ── 5. Train / Test split ──────────────────────────────────────────────────
X = df.drop("price_amd", axis=1)
y = df["price_amd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}   Test size: {X_test.shape[0]}")

# ── 6. Train RandomForestRegressor ─────────────────────────────────────────
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
)
model.fit(X_train, y_train)

# ── 7. Evaluate ────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n══════════════════════════════════════")
print(f"  R-squared (R²) : {r2:.4f}")
print(f"  MAE (AMD)      : {mae:,.0f}")
print("══════════════════════════════════════\n")

# ── 8. Save model ──────────────────────────────────────────────────────────
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "rf_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")
