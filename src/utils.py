# utils.py

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2

import matplotlib.pyplot as plt

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

        # --- МОДИФІКАЦІЯ: Встановлення динамічних меж осі Y ---
        min_val = dist_history_df[col].min()
        max_val = dist_history_df[col].max()
        
        # Додаємо невеликий запас (padding)
        padding = (max_val - min_val) * 0.1 # 10% від діапазону
        if padding == 0: # Якщо всі значення однакові
            padding = 0.1 # Встановлюємо мінімальний запас
            
        ax.set_ylim(min_val - padding, max_val + padding)
        # --- КІНЕЦЬ МОДИФІКАЦІЇ ---

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

    # Нормалізація RMSE відносно мінімальних та максимальних значень
    x_hat_min = np.min(x_hat_hist[:, :selected_columns], axis=0)
    x_hat_max = np.max(x_hat_hist[:, :selected_columns], axis=0)
    rmse_normalized = rmse_vec / (x_hat_max - x_hat_min + 1e-8)  # Додаємо мале значення для уникнення ділення на нуль

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
    print(f"Normalized RMSE (each state): {np.round(rmse_normalized, 4)}")
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
        rmse_normalized=rmse_normalized,
        rmse_total=rmse_tot,
        # nees=nees, 
        nees_mean=nees_mean,
        # nis=nis,  
        nis_mean=nis_mean,
        nees_cov=nees_cov, 
        nis_cov=nis_cov
    )