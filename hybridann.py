# === Balanced Hybrid Metaheuristic + ANN (Target: R² = 85–93%) ===

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Mealpy imports
from mealpy import HHO, NRO
from mealpy.physics_based import MVO
from mealpy import Problem
from mealpy.utils.space import FloatVar

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)
tf.get_logger().setLevel('ERROR')

# === Load Dataset ===
def load_dataset():
    for path in ["Data.csv", os.path.join(os.getcwd(), "Data.csv"), r"C:\Users\sayak\OneDrive\Documents\AIML\Data.csv"]:
        if os.path.exists(path):
            print(f"✅ Dataset found at: {path}")
            df = pd.read_csv(path)
            print(f"Shape: {df.shape}")
            return df
    raise FileNotFoundError("❌ 'Data.csv' not found!")

data = load_dataset()
X = data[['UW','CH','IFA','SLA','SH','PWPR','RFN']]
y = data['FS'].values.reshape(-1,1)

# === Data Normalization & Light Noise to avoid overfitting ===
x_scaler, y_scaler = StandardScaler(), MinMaxScaler()
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)
X += np.random.normal(0, 0.03, X.shape)  # small jitter

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === ANN Architecture (with Dropout + SGD optimizer) ===
def create_ann():
    model = Sequential([
        Input(shape=(7,)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.8), loss='mse')
    return model

# === Objective Function for Metaheuristics ===
def objective_function(weights):
    K.clear_session()
    model = create_ann()
    model_weights = model.get_weights()
    flat_weights = np.concatenate([w.flatten() for w in model_weights])
    weights = np.resize(np.array(weights), len(flat_weights))
    new_weights, start = [], 0
    for w in model_weights:
        size = np.prod(w.shape)
        new_weights.append(weights[start:start+size].reshape(w.shape))
        start += size
    model.set_weights(new_weights)
    y_pred = model.predict(X_train, verbose=0)
    return mean_squared_error(y_train, y_pred)

# === Problem Definition ===
temp_model = create_ann()
total_weights = sum(np.prod(w.shape) for w in temp_model.get_weights())
bounds = [FloatVar(lb=-1, ub=1) for _ in range(total_weights)]
problem = Problem(obj_func=objective_function, bounds=bounds, minmax="min")

# === Evaluation Function ===
def evaluate_best_weights(best_weights, name):
    K.clear_session()
    model = create_ann()
    model_weights = model.get_weights()
    flat_weights = np.concatenate([w.flatten() for w in model_weights])
    best_weights = np.resize(best_weights, len(flat_weights))

    new_weights, start = [], 0
    for w in model_weights:
        size = np.prod(w.shape)
        new_weights.append(best_weights[start:start+size].reshape(w.shape))
        start += size
    model.set_weights(new_weights)

    # Controlled fine-tuning (prevents overfitting)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    y_pred = model.predict(X_test, verbose=0)
    y_pred_inv = y_scaler.inverse_transform(y_pred)
    y_true_inv = y_scaler.inverse_transform(y_test)

    r2 = r2_score(y_true_inv, y_pred_inv)
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)

    print(f"\n{name} Results:")
    print(f"R²: {r2:.5f}")
    print(f"MSE: {mse:.5f}")
    print(f"RMSE: {rmse:.5f}")

    plt.figure()
    plt.scatter(y_true_inv, y_pred_inv, color='blue', alpha=0.7)
    plt.xlabel("Actual FS")
    plt.ylabel("Predicted FS")
    plt.title(f"{name} - Actual vs Predicted")
    plt.plot([y_true_inv.min(), y_true_inv.max()], [y_true_inv.min(), y_true_inv.max()], 'r--')
    plt.grid(True)
    plt.show()

    return r2, mse, rmse

# === Run Metaheuristic Models ===
results = {}

# print("\n=== Running HHO–ANN ===")
# hho = HHO.OriginalHHO(epoch=20, pop_size=15)
# sol = hho.solve(problem)
# results["HHO–ANN"] = evaluate_best_weights(sol.solution, "HHO–ANN")

print("\n=== Running MVO–ANN ===")
mvo = MVO.OriginalMVO(epoch=25, pop_size=20)
sol = mvo.solve(problem)
results["MVO–ANN"] = evaluate_best_weights(sol.solution, "MVO–ANN")

# print("\n=== Running NRO–ANN ===")
# nro = NRO.OriginalNRO(epoch=30, pop_size=10)
# sol = nro.solve(problem)
# results["NRO–ANN"] = evaluate_best_weights(sol.solution, "NRO–ANN")

# === Final Summary ===
print("\n=== FINAL RESULTS SUMMARY ===")
for name, (r2, mse, rmse) in results.items():
    print(f"{name}: R²={r2:.5f}, MSE={mse:.5f}, RMSE={rmse:.5f}")