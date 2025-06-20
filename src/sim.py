# sim.py

import numpy as np
import pandas as pd
from typing import Callable
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_gen import DataGenerator
from model import KernelModel
from objectives import  MaxIronMassTrackingObjective
from mpc import MPCController
from utils import (compute_metrics, train_val_test_time_series, analyze_sensitivity, plot_mpc_diagnostics,
                   analize_errors, plot_control_and_disturbances, plot_delta_u_histogram, analyze_correlation,
                   plot_historical_data, plot_fact_vs_mpc_plans, plot_disturbance_estimation)
from ekf import ExtendedKalmanFilter

def simulate_mpc(
    reference_df: pd.DataFrame,
    N_data: int = 5000,
    control_pts: int = 1000,
    lag: int = 2,
    Np: int = 6,
    Nc: int = 4,
    n_neighbors: int = 5,
    seed: int = 0,
    noise_level: str = 'none',
    model_type: str = 'krr',
    kernel: str = 'rbf',
    alpha: float = 1.0,
    gamma: float = None,
    find_optimal_params: bool = True,
    λ_obj: float = 0.1,
    K_I: float = 0.01,
    w_fe: float = 7.0,
    w_mass: float = 1.0,
    ref_fe: float = 53.5,
    ref_mass: float = 57.0,
    train_size: float = 0.7,
    val_size: float   = 0.15,
    test_size: float  = 0.15,
    u_min: float  = 20.0, 
    u_max: float  = 40.0, 
    delta_u_max: float  = 1.0,
    use_disturbance_estimator: bool = True,
    y_max_fe: float = 54.5,
    y_max_mass: float = 58.0,
    rho_trust: float = 0.1,
    use_soft_constraints: bool = True,
    plant_model_type: str = 'rf',
    progress_callback: Callable[[int, int, str], None] = None
):

    # Базова вага - середня вага трекінгу
    avg_tracking_weight = (w_fe + w_mass) / 2.   
    # Штраф за порушення виходів має бути на 2-3 порядки більшим
    rho_y_val = avg_tracking_weight * 1000 
    # Штраф за порушення дельти керування має бути співмірним з λ
    rho_du_val = λ_obj * 100
    
    # 1. «Справжній» генератор процесу
    true_gen = DataGenerator(
        reference_df,
        ore_flow_var_pct=3.0,
        time_step_s=5.0,
        time_constant_s=8.0,
        dead_time_s=20.0,
        true_model_type=plant_model_type
    )    
    anomaly_config = None 
    df_true  = true_gen.generate(N_data, control_pts, n_neighbors, noise_level=noise_level, anomaly_config=anomaly_config)
    
    # 2. Лаговані X, Y і послідовне розбиття на train/val/test
    # DataGenerator.create_lagged_dataset повертає numpy.ndarray
    X, Y_full_np = DataGenerator.create_lagged_dataset(df_true, lags=lag) #
    
    # Вибираємо тільки потрібні виходи для моделювання та MPC (Fe та Mass концентрату) за індексами
    # concentrate_fe_percent - індекс 0, concentrate_mass_flow - індекс 2
    Y = Y_full_np[:, [0, 2]] #
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = \
        train_val_test_time_series(X, Y, train_size, val_size, test_size)
        
    # ================== ПОЧАТОК БЛОКУ НОРМАЛІЗАЦІЇ ==================
    
    # 2a. Створюємо та навчаємо скалери ТІЛЬКИ на тренувальних даних
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_scaled = x_scaler.fit_transform(X_train)
    Y_train_scaled = y_scaler.fit_transform(Y_train)
    
    # 2b. Трансформуємо валідаційні та тестові дані за допомогою навчених скалерів
    X_val_scaled = x_scaler.transform(X_val)
    Y_val_scaled = y_scaler.transform(Y_val)
    X_test_scaled = x_scaler.transform(X_test)
    Y_test_scaled = y_scaler.transform(Y_test) # Для фінальної оцінки якості моделі

    # =================== КІНЕЦЬ БЛОКУ НОРМАЛІЗАЦІЇ ===================

    # 3–4. Модель і MPC-контролер
    # Модель тепер буде працювати зі змасштабованими даними
    km = KernelModel(
        model_type=model_type,
        kernel=kernel,
        find_optimal_params=find_optimal_params
    )
    
    # ================== ПОЧАТОК ЗМІН ПАРАМЕТРІВ MPC ==================
    
    # 4a. Масштабуємо уставки та обмеження для MPC, щоб вони відповідали
    # масштабу виходів Y, на яких навчалась модель.
    # ref_point - це [conc_fe, conc_mass]
    ref_point_original = np.array([[ref_fe, ref_mass]]) # Створюємо 2D-масив
    ref_point_scaled = y_scaler.transform(ref_point_original)
    ref_fe_scaled = ref_point_scaled[0, 0]
    ref_mass_scaled = ref_point_scaled[0, 1] # Індекс 1, оскільки тільки 2 цільові

    y_max_original = np.array([[y_max_fe, y_max_mass]]) #
    y_max_scaled_full = y_scaler.transform(y_max_original)
    y_max_fe_scaled = y_max_scaled_full[0, 0]
    y_max_mass_scaled = y_max_scaled_full[0, 1] # Індекс 1
    
    # 4b. Створюємо об'єкт цілі з новими, масштабованими уставками
    obj = MaxIronMassTrackingObjective(
        λ=λ_obj, w_fe=w_fe, w_mass=w_mass, 
        ref_fe=ref_fe_scaled, 
        ref_mass=ref_mass_scaled, 
        K_I=K_I
    )
    
    # 4c. Створюємо контролер, передаючи йому скалер для вхідних даних
    mpc = MPCController(
        model=km,
        objective=obj,
        x_scaler=x_scaler,  
        y_scaler=y_scaler,
        n_targets=Y_train_scaled.shape[1], # Тепер Y_train_scaled матиме лише 2 стовпці
        horizon=Np,
        control_horizon=Nc,
        lag=lag,
        u_min=u_min, u_max=u_max,
        delta_u_max=delta_u_max,
        use_disturbance_estimator=use_disturbance_estimator,
        y_max=[y_max_fe_scaled, y_max_mass_scaled] if use_soft_constraints else None,
        rho_y=rho_y_val, 
        rho_delta_u=rho_du_val,
        rho_trust=rho_trust # Передаємо вагу регіону довіри
    )

    # =================== КІНЕЦЬ ЗМІН ПАРАМЕТРІВ MPC ===================

    # 5a. Навчаємо модель на МАСШТАБОВАНИХ даних
    cols_state = ['feed_fe_percent','ore_mass_flow','solid_feed_percent']
    # Історію в fit більше не передаємо, EKF буде керувати оцінкою стану
    mpc.fit(X_train_scaled, Y_train_scaled, None)
    
    # ================== БЛОК ОЦІНКИ МОДЕЛІ (залишається без змін) ==================
    print("\n" + "="*20 + " Оцінка якості моделі " + "="*20)
    y_val_pred_scaled = mpc.model.predict(X_val_scaled)
    y_val_pred_orig = y_scaler.inverse_transform(y_val_pred_scaled)
    Y_val_orig = y_scaler.inverse_transform(Y_val_scaled)
    val_mse = mean_squared_error(Y_val_orig, y_val_pred_orig)
    print(f"-> Загальна помилка моделі на валідаційних даних (MSE): {val_mse:.4f}")
    y_test_pred_scaled = mpc.model.predict(X_test_scaled)
    y_test_pred_orig = y_scaler.inverse_transform(y_test_pred_scaled)
    Y_test_orig = y_scaler.inverse_transform(Y_test_scaled)
    test_mse = mean_squared_error(Y_test_orig, y_test_pred_orig)
    print(f"-> Загальна помилка моделі на тестових даних (MSE): {test_mse:.4f}\n")
    print("--- Детальний аналіз помилок на ТЕСТОВИХ даних ---")
    # Оновлено: output_columns для оцінки моделі тепер тільки 2
    output_columns = ['conc_fe', 'conc_mass'] #
    # Y_test_orig також має лише 2 стовпці тепер.
    for i, col_name in enumerate(output_columns): #
        rmse = np.sqrt(mean_squared_error(Y_test_orig[:, i], y_test_pred_orig[:, i])) #
        mae = mean_absolute_error(Y_test_orig[:, i], y_test_pred_orig[:, i]) #
        units = "%" if "fe" in col_name else "т/год"
        print(f"-> {col_name}:\n     RMSE = {rmse:.3f} {units}\n     MAE  = {mae:.3f} {units}")
    print("="*64 + "\n")
    metrics = {'validation_mse_total': val_mse, 'test_mse_total': test_mse}
    # =================== КІНЕЦЬ БЛОКУ ОЦІНКИ МОДЕЛІ ===================
    
    # 5b. Визначаємо початок тестової ділянки та початкову історію
    n = X.shape[0]
    n_train = int(train_size * n)
    n_val = int(val_size * n)
    test_idx = (lag + 1) + n_train + n_val
    hist0_unscaled = df_true[cols_state].iloc[test_idx - (lag + 1): test_idx].values
    
    # ================== ІНІЦІАЛІЗАЦІЯ EKF ==================
    n_phys = (lag + 1) * 3
    n_dist = 2 # Тепер n_dist = 2, оскільки ми відстежуємо лише Fe і Mass концентрату

    # Початковий розширений стан: фізична частина (немасштабована) + нульові збурення
    x0_aug = np.hstack([hist0_unscaled.flatten(), np.zeros(n_dist)]) 

    # Початкова коваріація: висока невизначеність для збурень, низька для стану
    P0 = np.eye(n_phys + n_dist) * 1e-2
    P0[n_phys:, n_phys:] *= 1000 # Встановлюємо значно вищу невизначеність для збурень

    # Матриці шумів (ключові параметри для тюнінгу!)
    # Шум процесу: малий для фізичного стану, більший для збурень
    Q_phys = np.eye(n_phys) * 1e-6 
    Q_dist = np.eye(n_dist) * 1e-4 
    Q = np.block([[Q_phys, np.zeros((n_phys, n_dist))], [np.zeros((n_dist, n_phys)), Q_dist]])

    # Шум вимірювань (в масштабованому просторі)
    # --- ВИПРАВЛЕНО ЗГІДНО З РЕКОМЕНДАЦІЯМИ ---
    # R = np.eye(n_dist) * 0.1 # Старе значення

    # Використовуємо дисперсію масштабованих тренувальних даних
    # Y_train_scaled має бути доступний тут
    eta_R = 0.5 # Коефіцієнт для R
    R = np.diag(np.var(Y_train_scaled, axis=0)) * eta_R 
    # --- КІНЕЦЬ ВИПРАВЛЕННЯ ---

    ekf = ExtendedKalmanFilter(mpc.model, x_scaler, y_scaler, x0_aug, P0, Q, R, lag)
    
    # Встановлюємо історію в MPC один раз для коректної ініціалізації
    mpc.reset_history(hist0_unscaled)
    # =================== КІНЕЦЬ ІНІЦІАЛІЗАЦІЇ EKF ===================

    # 5c. Ініціалізація змінних симуляції
    df_run = df_true.iloc[test_idx:]
    d_all = df_run[['feed_fe_percent','ore_mass_flow']].values
    T_sim = len(df_run) - (lag + 1)

    alpha_input_filter = 0.1
    d_filtered_state = d_all[0, :].copy()
    
    records, u_applied, disturbance_history = [], [], []
    u_prev = float(hist0_unscaled[-1, 2])

    # 5d. Замкнений цикл MPC з використанням EKF
    for t in range(T_sim):
        if progress_callback:
            progress_callback(t, T_sim, f"Крок {t+1}/{T_sim}")

        # 1. КРОК ПРОГНОЗУ EKF
        # Прогнозуємо стан на поточний крок k, використовуючи керування з кроку k-1
        d_raw_current = d_all[t, :] 
        ekf.predict(u_prev, d_raw_current)
        
        # 2. ОТРИМУЄМО ОЦІНКИ СТАНУ ТА ЗБУРЕНЬ ВІД EKF
        x_est_phys_unscaled = ekf.x_hat[:n_phys].reshape(lag + 1, 3)
        d_est_scaled = ekf.x_hat[n_phys:]
        
        # Оновлюємо внутрішній стан MPC на основі оцінки EKF
        mpc.reset_history(x_est_phys_unscaled)
        mpc.d_hat = d_est_scaled # Передаємо оцінку збурення в MPC

        # Зберігаємо історію оцінки збурень (вони вже в масштабованому вигляді)
        disturbance_history.append(d_est_scaled.copy())

        # --- Каузальна фільтрація вхідних збурень для прогнозу (залишається) ---
        d_input_current_filtered = d_all[t+1, :]
        d_filtered_state = alpha_input_filter * d_input_current_filtered + (1 - alpha_input_filter) * d_filtered_state
        d_seq = np.repeat(d_filtered_state[None, :], Np, axis=0)
        
        # 3. РОЗРАХУНОК КЕРУЮЧОЇ ДІЇ
        u_seq = mpc.optimize(d_seq, u_prev)
        u_cur = u_prev if u_seq is None else float(u_seq[0])
        u_applied.append(u_cur)

        # 4. СИМУЛЯЦІЯ РЕАЛЬНОГО ПРОЦЕСУ
        # Використовуємо реальні (нефільтровані) збурення для отримання вимірювання
        inp = pd.DataFrame([[*d_all[t+1], u_cur]], columns=cols_state)
        y_pred_from_gen = true_gen._predict_outputs(inp)
        y_corr = true_gen._apply_mass_balance(inp, y_pred_from_gen)
        y_corr['ore_mass_flow'] = inp.ore_mass_flow.values
        y_corr['feed_fe_percent'] = inp.feed_fe_percent.values
        y_full = true_gen._derive(y_corr)
        
        # 5. КРОК КОРЕКЦІЇ EKF
        # Отримуємо реальне вимірювання і передаємо його в EKF для корекції (тепер тільки conc_fe, conc_mass)
        y_meas_unscaled = y_full[['concentrate_fe_percent', 'concentrate_mass_flow']].values.flatten() #
        ekf.update(y_meas_unscaled)

        # Зберігаємо результати кроку та оновлюємо u_prev
        records.append({
            'feed_fe_percent': inp.feed_fe_percent.iloc[0], 'ore_mass_flow': inp.ore_mass_flow.iloc[0],
            'solid_feed_percent': u_cur, 'conc_fe': y_full.concentrate_fe_percent.iloc[0],
            'tail_fe': y_full.tailings_fe_percent.iloc[0], 'conc_mass': y_full.concentrate_mass_flow.iloc[0],
            'tail_mass': y_full.tailings_mass_flow.iloc[0], 'mass_pull_pct': y_full.mass_pull_percent.iloc[0],
            'fe_recovery_percent': y_full.fe_recovery_percent.iloc[0],
        })
        u_prev = u_cur

    if progress_callback:
        progress_callback(T_sim, T_sim, "Симуляція завершена")

    # 6. Збір результатів і метрик
    results_df = pd.DataFrame(records)
    metrics['avg_iron_mass'] = (results_df.conc_fe * results_df.conc_mass / 100).mean()
    
    # 7. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ
    analize_errors(results_df, ref_fe, ref_mass)
    plot_control_and_disturbances(np.array(u_applied), d_all[1:1+len(u_applied)])
    plot_mpc_diagnostics(results_df, w_fe, w_mass, λ_obj)
        
    print("=" * 50)   
    return results_df, metrics

if __name__ == '__main__':
    def my_progress(step, total, msg):
        print(f"[{step}/{total}] {msg}")

    try:
        hist_df = pd.read_parquet('processed.parquet')
    except FileNotFoundError:
        print("Помилка: файл 'processed.parquet' не знайдено.")
        exit()
    
    # Запускаємо симуляцію з оновленими, більш стабільними параметрами
    res, mets = simulate_mpc(
        hist_df, 
        progress_callback=my_progress, 
        N_data=1000, 
        control_pts=100,
        seed=42,
        
        train_size = 0.7,
        val_size   = 0.15,
        test_size  = 0.15,

        noise_level='low',
        model_type = 'krr',
        kernel='rbf', 
        find_optimal_params=True,
        use_soft_constraints=True,

        # 1. Збільшуємо вагу регіону довіри (найважливіша зміна)
        rho_trust = 50.0, # Почніть звідси, можна збільшувати до 50-100

        # 2. Збалансовуємо ваги цілі для масштабованих даних
        w_fe = 1.0,
        w_mass = 1.0,

        # 3. Задаємо уставки та м'які обмеження
        ref_fe = 54.0,
        ref_mass = 58.2,
        y_max_fe = 55.0,
        y_max_mass = 60.0
    )
    
    print("\nРезультати симуляції:")
    print(res.tail())
    print("\nМетрики:")
    print(mets)
    res.to_parquet('mpc_simulation_results.parquet')