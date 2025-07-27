
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Load and preprocess data
df = pd.read_csv("./data/musicdata_cleaned.csv")

df['Length'] = '0:' + df['Length'].astype(str)
df['Length'] = pd.to_timedelta(df['Length'], errors='coerce').dt.total_seconds()
df.dropna(inplace=True)

features = ['BPM', 'Energy', 'Dance', 'Loud', 'Valence', 'Length', 'Acoustic']
X = df[features]
y = df['Pop.']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR()
}

best_model = None
best_score = -np.inf

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name} - MSE: {mse:.2f} | R2: {r2:.3f}")
    
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

# Save best model and scaler
joblib.dump(best_model, f"./models/{best_model_name}_popularity_model.pkl")
joblib.dump(scaler, "./models/scaler.pkl")
print(f"\nBest model saved: {best_model_name}")
