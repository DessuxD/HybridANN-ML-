# === Hybrid Metaheuristic + ANN Models with 5-Fold Cross Validation ===
# Compatible with Python 3.11 + Mealpy 3.0.3

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# === Mealpy Imports ===
from mealpy import HHO
from mealpy.physics_based import MVO, NRO
from mealpy import Problem
from mealpy.utils.space import FloatVar

# === Reproducibility ===
np.random.seed(42)
tf.random.set_seed(42)
tf.get_logger().setLevel('ERROR')

# === 1. LOAD DATASET ===
# Remove <<r"C:\Users\sayak\OneDrive\Documents\AIML\Data.csv">> to avoid errors and keep Data.csv file in the same folder as hybridann.py
def load_dataset():
    possible_paths = [
        "Data.csv",
        os.path.join(os.getcwd(), "Data.csv"),
        r"C:\Users\sayak\OneDrive\Documents\AIML\Data.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Dataset found: {path}")
            df = pd.read_csv(path)
            print(f"Loaded successfully! Shape: {df.shape}")
            print("Columns:", list(df.columns))
            return df
    raise FileNotFoundError("❌ 'Data.csv' not found!")

data = load_dataset()

# === 2. PREPARE INPUTS & OUTPUT ===
X = data[['UW', 'CH', 'IFA', 'SLA', 'SH', 'PWPR', 'RFN']].values
y = data['FS'].values.reshape(-1, 1)

x_scaler = StandardScaler()
y_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

# === 3. BUILD BASE ANN MODEL ===
def create_ann():
    model = Sequential([
        Input(shape=(7,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === 4. DEFINE SEARCH SPACE ===
temp_model = create_ann()
total_weights = sum(np.prod(w.shape) for w in temp_model.get_weights())
bounds = [FloatVar(lb=-1, ub=1) for _ in range(total_weights)]

# === 5. EVALUATION FUNCTION ===
def evaluate_best_weights(best_weights, X_train, X_test, y_train, y_test, name):
    K.clear_session()
    model = create_ann()

    model_weights = model.get_weights()
    flat_weights = np.concatenate([w.flatten() for w in model_weights])
    best_weights = np.resize(best_weights, len(flat_weights))

    new_weights, start = [], 0
    for w in model_weights:
        shape, size = w.shape, np.prod(w.shape)
        new_weights.append(best_weights[start:start + size].reshape(shape))
        start += size
    model.set_weights(new_weights)

    # Fine-tune ANN
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    y_pred = model.predict(X_test, verbose=0)
    y_pred_inv = y_scaler.inverse_transform(y_pred)
    y_true_inv = y_scaler.inverse_transform(y_test)

    r2 = r2_score(y_true_inv, y_pred_inv)
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    accuracy = r2 * 100

    print(f"\n{name} Results:")
    print(f"R² (Accuracy): {accuracy:.2f}%")
    print(f"MSE: {mse:.5f}")
    print(f"RMSE: {rmse:.5f}")

    plt.figure()
    plt.scatter(y_true_inv, y_pred_inv, color='blue', alpha=0.7)
    plt.xlabel("Actual FS")
    plt.ylabel("Predicted FS")
    plt.title(f"{name} - Actual vs Predicted")
    plt.plot([y_true_inv.min(), y_true_inv.max()],
             [y_true_inv.min(), y_true_inv.max()], 'r--')
    plt.grid(True)
    plt.show()

    return accuracy, mse, rmse

# === 6. 5-FOLD CROSS VALIDATION FOR EACH OPTIMIZER ===
K_FOLDS = 5
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
optimizers = {
    "HHO–ANN": HHO.OriginalHHO(epoch=40, pop_size=15),
    "MVO–ANN": MVO.OriginalMVO(epoch=40, pop_size=15),
    "NRO–ANN": NRO.OriginalNRO(epoch=40, pop_size=15)
}

final_results = {}

for name, optimizer in optimizers.items():
    print(f"\n==============================")
    print(f"Running {name} with {K_FOLDS}-Fold Cross Validation")
    print(f"==============================")

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\n--- Fold {fold}/{K_FOLDS} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        def objective_function(weights):
            K.clear_session()
            model = create_ann()
            model_weights = model.get_weights()
            flat_weights = np.concatenate([w.flatten() for w in model_weights])
            weights = np.resize(np.array(weights), len(flat_weights))

            new_weights, start = [], 0
            for w in model_weights:
                shape, size = w.shape, np.prod(w.shape)
                new_weights.append(weights[start:start + size].reshape(shape))
                start += size
            model.set_weights(new_weights)
            y_pred = model.predict(X_train, verbose=0)
            return mean_squared_error(y_train, y_pred)

        problem = Problem(obj_func=objective_function, bounds=bounds, minmax="min")

        solution = optimizer.solve(problem)
        acc, mse, rmse = evaluate_best_weights(solution.solution, X_train, X_test, y_train, y_test, f"{name} Fold {fold}")
        fold_metrics.append((acc, mse, rmse))

    # Average across folds
    avg_r2 = np.mean([r[0] for r in fold_metrics])
    avg_mse = np.mean([r[1] for r in fold_metrics])
    avg_rmse = np.mean([r[2] for r in fold_metrics])
    final_results[name] = (avg_r2, avg_mse, avg_rmse)

    print(f"\n>>> {name} Average Results across {K_FOLDS} folds <<<")
    print(f"Average R²: {avg_r2:.2f}%")
    print(f"Average MSE: {avg_mse:.5f}")
    print(f"Average RMSE: {avg_rmse:.5f}")

# === 7. FINAL SUMMARY ===
print("\n==============================")
print("FINAL CROSS-VALIDATION SUMMARY")
print("==============================")
for name, (r2, mse, rmse) in final_results.items():
    print(f"{name}: Avg R²={r2:.2f}%, Avg MSE={mse:.5f}, Avg RMSE={rmse:.5f}")