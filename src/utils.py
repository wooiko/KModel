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
      delta_u_max    – гранична |Δu| для розрахунку порогу значних змін
      threshold_ratio– доля порогу для лічильника значних змін (за замовчуванням 0.9·Δu_max)

    Повертає словник з:
      mean_delta_u                       – середнє |Δu|
      std_delta_u                        – std |Δu|
      energy_u                           – E_u = mean(u^2)
      significant_magnitude_changes_count– кількість кроків, де |Δu| ≥ threshold_ratio·Δu_max
      significant_magnitude_changes_frequency – significant_magnitude_changes_count / (T−1)
      directional_switch_count           – кількість змін напрямку Δu (з + на - або з - на +)
      directional_switch_frequency       – directional_switch_count / (T−1)
      max_abs_delta_u                    – максимальне абсолютне значення Δu
      mean_abs_nonzero_delta_u           – середнє абсолютне значення ненульових Δu
      num_steps_at_delta_u_max           – кількість кроків, де |Δu| дорівнює delta_u_max
      percentage_of_max_delta_u_used     – середній відсоток використаного Δu_max
    """
    du = np.diff(u)
    abs_du = np.abs(du)

    # 1. Основні метрики за величиною зміни
    mean_du = abs_du.mean()
    std_du = abs_du.std()
    energy = np.mean(u**2) # Слід звернути увагу, що це енергія самого u, а не його змін.

    # 2. Метрики значних змін за величиною
    threshold = delta_u_max * threshold_ratio
    significant_switches = int((abs_du >= threshold).sum())
    significant_freq = significant_switches / len(du) if len(du) > 0 else 0.0

    # 3. Метрики зміни напрямку
    non_zero_du_indices = np.where(du != 0)[0]
    
    directional_switches = 0
    if len(non_zero_du_indices) > 1:
        signs = np.sign(du[non_zero_du_indices])
        directional_switches = np.sum(np.diff(signs) != 0)
    
    directional_freq = directional_switches / len(du) if len(du) > 0 else 0.0

    # 4. Нові метрики для оцінки "різкості" керування
    max_abs_du = abs_du.max() if len(abs_du) > 0 else 0.0

    non_zero_abs_du = abs_du[abs_du != 0]
    mean_abs_nonzero_du = non_zero_abs_du.mean() if len(non_zero_abs_du) > 0 else 0.0

    # Кількість кроків, де |Δu| досягає delta_u_max (з певним допуском через float)
    tolerance = 1e-6 # Допуск для порівняння чисел з плаваючою комою
    num_at_delta_u_max = np.sum(np.isclose(abs_du, delta_u_max, atol=tolerance))

    # Середній відсоток використання максимального Δu
    if delta_u_max > tolerance and len(abs_du) > 0:
        percentage_of_max_du_used = (abs_du / delta_u_max).mean() * 100
    else:
        percentage_of_max_du_used = 0.0

    return {
        'mean_delta_u': mean_du,
        'std_delta_u': std_du,
        'energy_u': energy,
        'significant_magnitude_changes_count': significant_switches,
        'significant_magnitude_changes_frequency': significant_freq,
        'directional_switch_count': directional_switches,
        'directional_switch_frequency': directional_freq,
        'max_abs_delta_u': max_abs_du,
        'mean_abs_nonzero_delta_u': mean_abs_nonzero_du,
        'num_steps_at_delta_u_max': num_at_delta_u_max,
        'percentage_of_max_delta_u_used': percentage_of_max_du_used
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

def plot_trust_region_evolution(trust_stats_hist):
    """Візуалізація еволюції trust region."""
    import matplotlib.pyplot as plt
    
    if not trust_stats_hist:
        print("Немає даних для візуалізації trust region")
        return
    
    steps = range(len(trust_stats_hist))
    radii = [stats['current_radius'] for stats in trust_stats_hist]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(steps, radii, 'b-', linewidth=2, label='Trust Region Radius')
    plt.axhline(y=trust_stats_hist[0].get('min_radius', 0.1), 
                color='r', linestyle='--', alpha=0.7, label='Min Radius')
    plt.axhline(y=trust_stats_hist[0].get('max_radius', 5.0), 
                color='r', linestyle='--', alpha=0.7, label='Max Radius')
    plt.ylabel('Trust Region Radius')
    plt.title('Еволюція Trust Region Radius')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Аналіз стабільності
    plt.subplot(2, 1, 2)
    if len(radii) > 10:
        moving_avg = pd.Series(radii).rolling(window=10, center=True).mean()
        plt.plot(steps, moving_avg, 'g-', linewidth=2, label='Ковзне середнє (10 кроків)')
    
    plt.plot(steps, radii, 'b-', alpha=0.5, label='Фактичний radius')
    plt.xlabel('Кроки симуляції')
    plt.ylabel('Trust Region Radius')
    plt.title('Згладжена еволюція Trust Region')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
def plot_linearization_quality(lin_quality_hist, params):
    """Візуалізація якості лінеаризації."""
    import matplotlib.pyplot as plt
    
    if not lin_quality_hist:
        print("Немає даних для візуалізації якості лінеаризації")
        return
    
    # ВИПРАВЛЕННЯ: правильно обробляємо різні типи даних
    if isinstance(lin_quality_hist[0], dict):
        # Якщо це словники з детальною інформацією
        distances = [h['euclidean_distance'] for h in lin_quality_hist]
    else:
        # Якщо це прості числа
        distances = lin_quality_hist
    
    steps = range(len(distances))
    
    plt.figure(figsize=(12, 4))
    plt.plot(steps, distances, 'purple', alpha=0.7, label='Відстань лінеаризації')
    
    # Додаємо поріг
    threshold = params.get('max_linearization_distance', 2.0)
    plt.axhline(y=threshold, color='red', linestyle='--', 
                label=f'Поріг ({threshold})', alpha=0.8)
    
    # Ковзне середнє
    if len(distances) > 20:
        moving_avg = pd.Series(distances).rolling(window=20, center=True).mean()
        plt.plot(steps, moving_avg, 'darkred', linewidth=2, 
                 label='Ковзне середнє (20 кроків)')
    
    plt.xlabel('Кроки симуляції')
    plt.ylabel('Відстань від точки лінеаризації')
    plt.title('Якість лінеаризації протягом симуляції')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Статистика
    avg_quality = np.mean(distances)
    max_quality = np.max(distances)
    violations = sum(1 for q in distances if q > threshold)
    
    print(f"\n--- Статистика якості лінеаризації ---")
    print(f"Середня відстань: {avg_quality:.4f}")
    print(f"Максимальна відстань: {max_quality:.4f}")
    print(f"Порушень порогу: {violations}/{len(distances)} ({100*violations/len(distances):.1f}%)")
    
def run_post_simulation_analysis_enhanced(results_df, analysis_data, params):
    """Розширений аналіз результатів симуляції з trust region статистикою."""
    print("\n" + "="*20 + " РОЗШИРЕНИЙ АНАЛІЗ РЕЗУЛЬТАТІВ " + "="*20)
    
    u_applied = results_df['solid_feed_percent'].values
    d_all = analysis_data['d_all_test']
    
    # 1. Загальна поведінка керування та збурень
    plot_control_and_disturbances(u_applied, d_all, title="Фінальне керування та збурення")
    
    # 2. Помилки відстеження уставки
    analize_errors(results_df, params['ref_fe'], params['ref_mass'])
    
    # 3. Агресивність керування
    agg_metrics = control_aggressiveness_metrics(u_applied, params['delta_u_max'])
    print("\n--- Метрики агресивності керування ---")
    for key, val in agg_metrics.items():
        print(f"{key:<20}: {val:.4f}")
    plot_delta_u_histogram(u_applied)

    # 4. Аналіз роботи EKF
    if analysis_data['y_true'] is not None:
        evaluate_ekf_performance(
            analysis_data['y_true'], analysis_data['x_hat'], analysis_data['P'],
            analysis_data['innov'], analysis_data['R']
        )
    
    # 5. Аналіз оцінки збурень
    if analysis_data['d_hat'].size > 0:
        d_hat_df = pd.DataFrame(analysis_data['d_hat'], columns=['d_conc_fe', 'd_conc_mass'])
        plot_disturbance_estimation(d_hat_df)
    
    # 6. Візуалізація планів MPC
    if analysis_data['u_seq']:
        plot_fact_vs_mpc_plans(results_df, analysis_data['u_seq'], control_steps=results_df.index)
    
    # === НОВІ АНАЛІЗИ ===
    # 7. Аналіз Trust Region
    if 'trust_region_stats' in analysis_data and analysis_data['trust_region_stats']:
        plot_trust_region_evolution(analysis_data['trust_region_stats'])
        analyze_trust_region_performance(analysis_data['trust_region_stats'])
    
    # 8. Аналіз якості лінеаризації
    if 'linearization_quality' in analysis_data and analysis_data['linearization_quality']:
        plot_linearization_quality(analysis_data['linearization_quality'], params)
        
    print("="*60 + "\n")

def analyze_trust_region_performance(trust_stats_hist):
    """Аналіз ефективності адаптивного trust region."""
    if not trust_stats_hist:
        return
    
    radii = [stats['current_radius'] for stats in trust_stats_hist]
    
    # Статистика
    avg_radius = np.mean(radii)
    std_radius = np.std(radii)
    min_radius = np.min(radii)
    max_radius = np.max(radii)
    
    # Аналіз адаптивності
    radius_changes = np.diff(radii)
    num_increases = sum(1 for change in radius_changes if change > 0.01)
    num_decreases = sum(1 for change in radius_changes if change < -0.01)
    
    print(f"\n--- Аналіз Trust Region ---")
    print(f"Середній radius: {avg_radius:.4f} ± {std_radius:.4f}")
    print(f"Діапазон: [{min_radius:.4f}, {max_radius:.4f}]")
    print(f"Збільшень radius: {num_increases}")
    print(f"Зменшень radius: {num_decreases}")
    print(f"Коефіцієнт адаптивності: {(num_increases + num_decreases)/len(radius_changes):.3f}")
