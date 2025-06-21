# utils.py

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2

from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from typing import List, Optional  

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
    
def analize_errors(results_df, ref_fe, ref_mass):
    err_fe   = results_df['conc_fe']   - ref_fe
    err_mass = results_df['conc_mass'] - ref_mass
    
    plt.figure(figsize=(8,3))
    plt.plot(err_fe,   label='error conc_fe')
    plt.plot(err_mass, label='error conc_mass')
    plt.axhline(0, color='k', lw=0.8)
    plt.legend(); plt.xlabel('крок'); plt.ylabel('помилка')
    plt.title('Tracking error')
    plt.show()
    
def control_aggressiveness_metrics(u: np.ndarray,
                                   delta_u_max: float,
                                   threshold_ratio: float = 0.9
                                   ) -> dict:
    """
    Обчислює метрики агресивності керування.
    Параметри:
      u              – масив керування (u[0],…,u[T])
      delta_u_max    – гранична |Δu|
      threshold_ratio– доля порогу для лічильника переключень (за замовчуванням 0.9·Δu_max)
    Повертає словник з:
      mean_delta_u     – середнє |Δu|
      std_delta_u      – std |Δu|
      energy_u         – E_u = mean(u^2)
      switch_count     – кількість кроків, де |Δu| ≥ threshold_ratio·Δu_max
      switch_frequency – switch_count / (T−1)
    """
    du = np.diff(u)
    abs_du = np.abs(du)
    mean_du = abs_du.mean()
    std_du = abs_du.std()
    energy = np.mean(u**2)
    threshold = delta_u_max * threshold_ratio
    switches = int((abs_du >= threshold).sum())
    freq = switches / len(abs_du) if len(abs_du)>0 else 0.0
    return {
        'mean_delta_u': mean_du,
        'std_delta_u': std_du,
        'energy_u': energy,
        'switch_count': switches,
        'switch_frequency': freq
    }

def plot_delta_u_histogram(u: np.ndarray, bins: int = 20) -> None:
    """
    Побудова гістограми Δu = u[k]−u[k−1].
    """
    du = np.diff(u)
    plt.figure()
    plt.hist(du, bins=bins, edgecolor='black')
    plt.xlabel('Δu')
    plt.ylabel('Частота')
    plt.title('Гістограма змін керування Δu')
    plt.grid(True)
    plt.show()

# def plot_control_vs_disturbance(u: np.ndarray,
#                                 d: np.ndarray,
#                                 time: np.ndarray = None
#                                 ) -> None:
#     """
#     Візуалізує u(t) разом з збуреннями d(t).
#     Параметри:
#       u     – масив керування довжини T
#       d     – масив збурень форми (T, 2): [feed_fe_percent, ore_mass_flow]
#       time  – ось часу (довжина T), якщо None – використовує np.arange(T)
#     """
#     T = len(u)
#     if time is None:
#         time = np.arange(T)

#     fig, ax1 = plt.subplots()
#     ax1.step(time, u, where='post', color='tab:blue', label='u (solid_feed_percent)')
#     ax1.set_xlabel('Крок симуляції')
#     ax1.set_ylabel('u', color='tab:blue')
#     ax1.tick_params(axis='y', labelcolor='tab:blue')

#     ax2 = ax1.twinx()
#     ax2.plot(time, d[:T,0], '--', color='tab:green', label='feed_fe_percent')
#     ax2.plot(time, d[:T,1], '--', color='tab:orange', label='ore_mass_flow')
#     ax2.set_ylabel('Збурення', color='tab:green')
#     ax2.tick_params(axis='y', labelcolor='tab:green')

#     # легенда з обох осей
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

#     plt.title('Керування u та збурення d')
#     fig.tight_layout()
#     plt.show()

def plot_control_and_disturbances(u_seq: np.ndarray,
                                  d: np.ndarray,
                                  time: np.ndarray = None,
                                  title: str = None):
    """
    Малює:
     1) у верхньому субплоті – керуючий сигнал u,
     2) у нижньому – два канали d[:,0] та d[:,1], кожен на своїй Y-осі.
    """
    T = len(u_seq)
    if time is None:
        time = np.arange(T)

    fig, (ax_u, ax_d1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # 1) Керуючий сигнал u
    ax_u.step(time, u_seq, where='post', color='tab:blue')
    ax_u.set_ylabel('u (feed solid %)')
    ax_u.grid(True)

    # 2) Збурення d[:,0] на лівій осі
    ax_d1.plot(time, d[:T, 0], '--', color='tab:green', label='feed_fe_percent')
    ax_d1.set_ylabel('feed_fe_percent', color='tab:green')
    ax_d1.tick_params(axis='y', labelcolor='tab:green')
    ax_d1.grid(True)

    # 3) Збурення d[:,1] на правій осі
    if d.shape[1] > 1:
        ax_d2 = ax_d1.twinx()
        ax_d2.plot(time, d[:T, 1], '--', color='tab:orange', label='ore_mass_flow')
        ax_d2.set_ylabel('ore_mass_flow', color='tab:orange')
        ax_d2.tick_params(axis='y', labelcolor='tab:orange')

    ax_d1.set_xlabel('Крок симуляції')

    if title:
        fig.suptitle(title)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def plot_historical_data(hist_df: pd.DataFrame,
                         columns: Optional[List[str]] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         figsize: tuple = (14, 10),
                         timestamp_col: Optional[str] = 'timestamp'):
    """
    Візуалізація історичних даних з DataFrame.

    Параметри:
    - hist_df: DataFrame з datetime індексом або стовпцем 'timestamp'.
    - columns: Список стовпців для візуалізації. Якщо None — використовуються всі числові.
    - start_date, end_date: Фільтрація по датах (формат рядка, наприклад '2023-01-01').
    - figsize: Розмір фігури.

    Виводить субплоти для кожного обраного параметра.
    """
    df = hist_df.copy()

    # Встановлюємо індекс часу, якщо задано
    if timestamp_col is not None and timestamp_col in df.columns:
        df = df.set_index(timestamp_col)

    # Перетворюємо індекс у DatetimeIndex, якщо потрібно
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("Індекс не є DatetimeIndex і не вдалося конвертувати.")
            
    # Індекс як DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        else:
            raise ValueError("DataFrame повинен мати індекс DatetimeIndex або стовпець 'timestamp'.")

    # Фільтруємо за датами
    if start_date:
        df = df.loc[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df.loc[df.index <= pd.to_datetime(end_date)]

    # Визначаємо стовпці
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if not columns:
            raise ValueError("Числових стовпців не знайдено.")
    else:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Стовпці відсутні в DataFrame: {missing_cols}")

    n = len(columns)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.plot(df.index, df[col], label=col)
        ax.set_ylabel(col)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Час')
    plt.tight_layout()
    plt.show()
    
def plot_fact_vs_mpc_plans(results_df, all_u_sequences, control_steps, var_name="solid_feed_percent"):
    """
    Порівняння фактичних значень var_name з оптимізованими планами MPC.
    results_df      — DataFrame з фактичними результатами (індекс 0,1,...)
    all_u_sequences — список масивів [u_k, u_{k+1}, ...] для кожного кроку MPC
    control_steps   — список індексів (0,1,2,...) кроків, на яких оптимізували
    var_name        — назва стовпця в results_df, який малюємо
    """
    plt.figure(figsize=(12, 6))

    # 1. Фактичні значення
    plt.plot(results_df.index,
             results_df[var_name],
             'b-',
             linewidth=2,
             label=f'Факт {var_name}')

    # 2. Всі горизонти планів MPC
    for i, u_seq in enumerate(all_u_sequences):
        start = control_steps[i]
        t_plan = [start + j for j in range(len(u_seq))]
        plt.plot(t_plan, u_seq, 'r-', alpha=0.13, lw=2)

    # 3. Виділити декілька планів пунктиром
    for k in [0, len(all_u_sequences)//2, len(all_u_sequences)-1]:
        u_seq = all_u_sequences[k]
        start = control_steps[k]
        t_plan = [start + j for j in range(len(u_seq))]
        plt.plot(t_plan,
                 u_seq,
                 '--',
                 lw=2,
                 label=f'План MPC (крок {start})')

    plt.xlabel('Крок симуляції')
    plt.ylabel(var_name)
    plt.legend()
    plt.grid(True)
    plt.title(f"Факт та оптимальні плани MPC для {var_name}")
    plt.tight_layout()
    plt.show()
    
def plot_disturbance_estimation(dist_history_df: pd.DataFrame):
    """
    Візуалізує якість роботи оцінювача збурень.
    
    Створює графіки для кожної з оцінених компонент збурення (`d_hat`),
    показуючи їх динаміку протягом симуляції. Це демонструє, як
    контролер виявляє стале відхилення моделі від реальності.

    Args:
        dist_history_df (pd.DataFrame): DataFrame з історією оцінок збурень.
    """
    if dist_history_df.empty:
        print("Історія збурень порожня, візуалізація роботи оцінювача неможлива.")
        return

    # Назви для графіків
    output_names = {
        'd_conc_fe': 'Fe в концентраті (%)',
        'd_tail_fe': 'Fe в хвостах (%)',
        'd_conc_mass': 'Потік концентрату (т/год)',
        'd_tail_mass': 'Потік хвостів (т/год)'
    }
    
    num_plots = len(dist_history_df.columns)
    if num_plots == 0:
        return
        
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, num_plots * 3.5), sharex=True)
    if num_plots == 1:
        axes = [axes]
        
    fig.suptitle('Динаміка оціненого збурення (d_hat = y_real - y_model)', fontsize=18, y=0.99)
    sns.set_theme(style="whitegrid")

    for i, col in enumerate(dist_history_df.columns):
        ax = axes[i]
        sns.lineplot(data=dist_history_df, x=dist_history_df.index, y=col, ax=ax, 
                     label='Оцінка збурення d_hat', color='darkorange', linewidth=2)
        
        # Розрахунок середнього значення для візуалізації сталого зміщення
        mean_dist = dist_history_df[col].mean()
        ax.axhline(0, color='r', linestyle='--', lw=1.5, label='Нульовий зсув')
        ax.axhline(mean_dist, color='b', linestyle=':', lw=1.5, 
                   label=f'Середнє = {mean_dist:.2f}')
        
        title_text = output_names.get(col, col)
        ax.set_title(f'Збурення для виходу: «{title_text}»', fontsize=14)
        ax.set_ylabel('Величина зсуву')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    axes[-1].set_xlabel('Крок симуляції на тестових даних (t)', fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()
    
def evaluate_ekf_performance(
        x_true_hist: np.ndarray,
        x_hat_hist: np.ndarray,
        P_hist: np.ndarray,
        innov_hist: np.ndarray,
        R_hist: np.ndarray,
        conf_level: float = 0.95
):
    """
    Оцінка ефективності EKF з обчисленням показників та відповідних графіків.
    
    x_true_hist : (T, n_x)  – реальний стан із генератора
    x_hat_hist  : (T, n_x_full)  – оцінка EKF після корекції, може містити більше стовпців
    P_hist      : (T, n_x_full, n_x_full) – коваріації після корекції
    innov_hist  : (T, n_y)  – інновація v_k = y - y_pred
    R_hist      : (T, n_y, n_y) – вимірювальна коваріація (перед оновленням)

    Повертає dict з метриками та будує базові графіки.
    """
    T, n_x = x_true_hist.shape
    n_y = innov_hist.shape[1]

    # Визначення кількості стовпців, які ви хочете використовувати для оцінки
    selected_columns = min(n_x, x_hat_hist.shape[1])

    # ---- RMSE ----
    rmse_vec = np.sqrt(((x_true_hist[:, :selected_columns] - x_hat_hist[:, :selected_columns])**2).mean(axis=0))
    rmse_tot = float(np.linalg.norm(rmse_vec) / np.sqrt(selected_columns))

    # ---- NEES ----
    nees = np.empty(T)
    for k in range(T):
        e = (x_true_hist[k] - x_hat_hist[k, :selected_columns]).reshape(-1, 1)
        pinv = np.linalg.pinv(P_hist[k][:selected_columns, :selected_columns])
        nees[k] = float(e.T @ pinv @ e)
    nees_mean = nees.mean()

    # ---- NIS ----
    nis = np.empty(T)
    for k in range(T):
        v = innov_hist[k].reshape(-1, 1)
        rinv = np.linalg.pinv(R_hist[k])
        nis[k] = float(v.T @ rinv @ v)
    nis_mean = nis.mean()

    # ---- χ²-границі ----
    alpha = 1 - conf_level
    nees_lo = chi2.ppf(alpha / 2, df=selected_columns)
    nees_hi = chi2.ppf(1 - alpha / 2, df=selected_columns)
    nis_lo = chi2.ppf(alpha / 2, df=n_y)
    nis_hi = chi2.ppf(1 - alpha / 2, df=n_y)

    nees_cov = np.mean((nees >= nees_lo) & (nees <= nees_hi))
    nis_cov = np.mean((nis >= nis_lo) & (nis <= nis_hi))

    # ---- Вивід ----
    print("\n===== EKF PERFORMANCE =====")
    print(f"RMSE (each state): {np.round(rmse_vec, 4)}")
    print(f"RMSE (total)     : {rmse_tot:.4f}")
    print(f"NEES mean        : {nees_mean:.2f}  (ideal ≈ {selected_columns})")
    print(f"NIS mean         : {nis_mean:.2f}  (ideal ≈ {n_y})")
    print(f"NEES 95% coverage: {nees_cov * 100:.1f}% (target 95%)")
    print(f"NIS 95% coverage : {nis_cov * 100:.1f}% (target 95%)")

    # ---- Графіки ----
    t = np.arange(T)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax[0].plot(t, nees, label='NEES', color='blue')
    ax[0].axhline(nees_lo, color='red', ls='--', lw=0.8, label='χ² 95%')
    ax[0].axhline(nees_hi, color='red', ls='--', lw=0.8)
    ax[0].set_title('Normalized Estimation Error Squared')
    ax[0].legend()

    ax[1].plot(t, nis, label='NIS', color='green')
    ax[1].axhline(nis_lo, color='red', ls='--', lw=0.8, label='χ² 95%')
    ax[1].axhline(nis_hi, color='red', ls='--', lw=0.8)
    ax[1].set_title('Normalized Innovation Squared')
    ax[1].set_xlabel('Time step')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    return dict(
        rmse_vec=rmse_vec, 
        rmse_total=rmse_tot,
        nees=nees, 
        nees_mean=nees_mean,
        nis=nis,  
        nis_mean=nis_mean,
        nees_cov=nees_cov, 
        nis_cov=nis_cov
    )