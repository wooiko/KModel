# utils.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_val_test(X, Y,
                   train_size: float = 0.7,
                   val_size: float   = 0.15,
                   test_size: float  = 0.15,
                   random_state: int = 42):
    """
    Розбиває X, Y на train/val/test у пропорціях сумарно = 1.0.
    Повертає: X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size має дорівнювати 1.0")
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(random_state)
    rng.shuffle(idx)
    n_train = int(train_size * n)
    n_val   = int(val_size   * n)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    return (X[train_idx], Y[train_idx],
            X[val_idx],   Y[val_idx],
            X[test_idx],  Y[test_idx])


def add_noise(arr: np.ndarray, noise_std: float):
    """
    Додає до масиву гаусівський шум зі σ = noise_std.
    Повертає новий масив.
    """
    return arr + np.random.randn(*arr.shape) * noise_std


def compute_metrics(y_true, y_pred):
    """
    Обчислює MAE та RMSE для кожного стовпця.
    Підтримує numpy-масиви або pandas.DataFrame.
    Повертає словник {column+'_mae':…, column+'_rmse':…}.
    """
    # Перекладемо в numpy
    if hasattr(y_true, "values"):
        cols = list(y_true.columns)
        yt = y_true.values
    else:
        yt = np.asarray(y_true)
        cols = [f"col{i}" for i in range(yt.shape[1])]
    yp = np.asarray(y_pred)

    if yt.shape != yp.shape:
        raise ValueError(f"Форми y_true {yt.shape} і y_pred {yp.shape} повинні збігатися")

    metrics = {}
    for i, col in enumerate(cols):
        mae = mean_absolute_error(yt[:, i], yp[:, i])
        mse = mean_squared_error(yt[:, i], yp[:, i])  # без squared
        rmse = np.sqrt(mse)
        metrics[f"{col}_mae"]  = mae
        metrics[f"{col}_rmse"] = rmse
    return metrics
