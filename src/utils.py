# utils.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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

def train_val_test_time_series(X, Y,
                               train_size: float = 0.7,
                               val_size: float   = 0.15,
                               test_size: float  = 0.15):
    """
    Послідовне розбиття X, Y на train/val/test у пропорціях сумарно = 1.0,
    без перемішування.
    Повертає: X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size має дорівнювати 1.0")

    n = X.shape[0]
    n_train = int(train_size * n)
    n_val   = int(val_size   * n)
    # останні n_test = n - n_train - n_val
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    X_val   = X[n_train:n_train + n_val]
    Y_val   = Y[n_train:n_train + n_val]
    X_test  = X[n_train + n_val:]
    Y_test  = Y[n_train + n_val:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

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

def plot_mpc_diagnostics(
    results_df,
    w_fe: float,
    w_mass: float,
    λ: float
):
    """
    Малює u_k та значення cost_term = −(w_fe·conc_fe + w_mass·conc_mass) + λ·(u_k−u_{k−1})²
    за індексом кроку.
    """
    # Кроки
    t = np.arange(len(results_df))
    # Керуючий сигнал
    u = results_df['solid_feed_percent'].to_numpy()
    # попередній u (для першого кроку вважатимемо u_prev=u0)
    u_prev = np.roll(u, 1)
    u_prev[0] = u[0]
    # техн. складова
    conc_fe   = results_df['conc_fe'].to_numpy()
    conc_mass = results_df['conc_mass'].to_numpy()
    linear_term    = - (w_fe   * conc_fe + w_mass * conc_mass)
    smoothing_term = λ * (u - u_prev)**2
    cost = linear_term + smoothing_term

    # Побудова двох графіків
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(t, u, '-o', label='u (solid_feed_percent)')
    ax1.set_ylabel('u')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.plot(t, cost, '-o', color='C1', label='cost term')
    ax2.set_xlabel('Крок симуляції')
    ax2.set_ylabel('Цільова функція')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.suptitle('MPC: керуючий сигнал та цільова функція по часу')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    
def analyze_correlation(results_df):
    # 1. Обчислюємо коефіцієнт кореляції Пірсона
    corr = results_df['conc_fe'].corr(results_df['conc_mass'])
    print(f"Коефіцієнт кореляції conc_fe vs conc_mass = {corr:.4f}")

    t = np.arange(len(results_df))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 2. Часові ряди conc_fe та conc_mass
    axes[0,0].plot(t, results_df['conc_fe'], label='conc_fe')
    axes[0,0].plot(t, results_df['conc_mass'], label='conc_mass')
    axes[0,0].set_title('Часові ряди conc_fe та conc_mass')
    axes[0,0].set_xlabel('Крок симуляції')
    axes[0,0].set_ylabel('Значення')
    axes[0,0].legend()
    axes[0,0].grid(True)

    # 3. Розсіювання conc_fe vs conc_mass
    axes[0,1].scatter(results_df['conc_fe'], results_df['conc_mass'], s=20, alpha=0.7)
    axes[0,1].set_title('Scatter conc_fe vs conc_mass')
    axes[0,1].set_xlabel('conc_fe')
    axes[0,1].set_ylabel('conc_mass')
    axes[0,1].grid(True)

    # 4. Гістограми обох
    axes[1,0].hist(results_df['conc_fe'], bins=20, alpha=0.7, label='conc_fe')
    axes[1,0].hist(results_df['conc_mass'], bins=20, alpha=0.7, label='conc_mass')
    axes[1,0].set_title('Гістограми conc_fe та conc_mass')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # 5. Пустий субплот для примітки
    axes[1,1].axis('off')
    note = f"Кореляція = {corr:.3f}\n" + \
           ("Практично лінійно не різняться" if abs(corr) > 0.99 else "")
    axes[1,1].text(0.1, 0.5, note, fontsize=12)

    plt.tight_layout()
    plt.show()
    
def analyze_sensitivity(results_df, preds_df):
    """
    Виводить:
      1) Scatter-plot conc_fe(pred) та conc_mass(pred) vs u
      2) Лінійні апроксимації зв’язку і їхні коефіцієнти (slope)
    """
    u = results_df['solid_feed_percent'].to_numpy()
    conc_fe_pred   = preds_df['conc_fe'].to_numpy()
    conc_mass_pred = preds_df['conc_mass'].to_numpy()

    # Лінійна апроксимація: slope та intercept
    slope_fe,   intercept_fe   = np.polyfit(u, conc_fe_pred,   1)
    slope_mass, intercept_mass = np.polyfit(u, conc_mass_pred, 1)

    print(f"Slope conc_fe(u):   {slope_fe:.4f}")
    print(f"Slope conc_mass(u): {slope_mass:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # conc_fe vs u
    axes[0].scatter(u, conc_fe_pred, color='C0', alpha=0.7, label='pred conc_fe')
    axes[0].plot(u, slope_fe*u + intercept_fe, color='C1',
                 label=f'lin fit: y={slope_fe:.3f}·u+{intercept_fe:.1f}')
    axes[0].set_xlabel('u (solid_feed_percent)')
    axes[0].set_ylabel('conc_fe_pred')
    axes[0].legend()
    axes[0].grid(True)

    # conc_mass vs u
    axes[1].scatter(u, conc_mass_pred, color='C2', alpha=0.7, label='pred conc_mass')
    axes[1].plot(u, slope_mass*u + intercept_mass, color='C3',
                 label=f'lin fit: y={slope_mass:.3f}·u+{intercept_mass:.1f}')
    axes[1].set_xlabel('u (solid_feed_percent)')
    axes[1].set_ylabel('conc_mass_pred')
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle('Чутливість прогнозів до зміни керування u')
    plt.tight_layout()
    plt.show()