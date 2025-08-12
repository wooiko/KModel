# sim.py

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from collections import deque

from data_gen import StatefulDataGenerator
from model import KernelModel
from objectives import MaxIronMassTrackingObjective
from mpc import MPCController
from utils import (
    run_post_simulation_analysis_enhanced, diagnose_mpc_behavior, diagnose_ekf_detailed
    
)
from ekf import ExtendedKalmanFilter
from anomaly_detector import SignalAnomalyDetector
from maf import MovingAverageFilter
import json
from pathlib import Path
from typing import Optional, Dict
from evaluation_simple import evaluate_simulation, print_evaluation_report
from config_manager import (
    simulate_mpc_with_config,
    list_configs,
    create_default_configs,
    prompt_manual_adjustments,
    load_config,
    list_saved_results
)

# =============================================================================
# === БЛОК 1: ПІДГОТОВКА ДАНИХ ТА СКАЛЕРІВ ===
# =============================================================================

def prepare_simulation_data(
    reference_df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[StatefulDataGenerator, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Створює генератор, формує часовий ряд із аномаліями,
    ПРОМИВАЄ його SignalAnomalyDetector-ом, та будує
    лаговані матриці X, Y для подальшого навчання/симуляції.
    """
    print("Крок 1: Генерація симуляційних даних...")

    # 1. Ініціалізуємо генератор «plant»
    true_gen = StatefulDataGenerator(
        reference_df,
        ore_flow_var_pct=3.0,
        time_step_s=params['time_step_s'],
        time_constants_s=params['time_constants_s'],
        dead_times_s=params['dead_times_s'],
        true_model_type=params['plant_model_type'],
        seed=params['seed']
    )

    # 2. Конфігурація аномалій
    anomaly_cfg = StatefulDataGenerator.generate_anomaly_config(
        N_data=params['N_data'],
        train_frac=params['train_size'],
        val_frac=params['val_size'],
        test_frac=params['test_size'],
        seed=params['seed']
    )
    # 3. Генеруємо повний часовий ряд (з артефактами)
    df_true_orig = true_gen.generate(
        T=params['N_data'],
        control_pts=params['control_pts'],
        n_neighbors=params['n_neighbors'],
        noise_level=params['noise_level'],
        anomaly_config=anomaly_cfg
    )
    
    if params['enable_nonlinear']:
        # 4. Визначаємо, як ми хочемо посилити нелінійність
        nonlinear_config = params['nonlinear_config']
           
        # 5. Створюємо новий датасет з посиленою нелінійністю
        #    Можна передати той самий рівень шуму та конфігурацію аномалій
        df_true = true_gen.generate_nonlinear_variant(
            base_df=df_true_orig,
            non_linear_factors=nonlinear_config,
            noise_level='none',
            anomaly_config=None # або передати сюди конфігурацію аномалій
        )
    else:
        df_true=df_true_orig
    
    # 6. OFFLINE-ОЧИЩЕННЯ вхідних сигналів від аномалій
    #    Використовуємо ті самі налаштування, що й в online-циклі, або менш жорсткі.
    ad_config = params.get('anomaly_params', {})
    ad_feed_fe = SignalAnomalyDetector(**ad_config)
    ad_ore_flow = SignalAnomalyDetector(**ad_config)


    filtered_feed = []
    filtered_ore  = []
    for raw_fe, raw_ore in zip(df_true['feed_fe_percent'], df_true['ore_mass_flow']):
        filtered_feed.append(ad_feed_fe.update(raw_fe))
        filtered_ore.append(ad_ore_flow.update(raw_ore))

    # Підмінюємо «брудні» колонки на «очищені»
    df_true = df_true.copy()
    df_true['feed_fe_percent'] = filtered_feed
    df_true['ore_mass_flow']   = filtered_ore

    # 7. Лаговані вибірки для тренування/симуляції
    X, Y_full_np = StatefulDataGenerator.create_lagged_dataset(
        df_true,
        lags=params['lag']
    )
    # Вибираємо лише concentrate_fe та concentrate_mass колонки
    Y = Y_full_np[:, [0, 2]]

    return true_gen, df_true, X, Y

def split_and_scale_data(
    X: np.ndarray,
    Y: np.ndarray,
    params: Dict[str, Any]
) -> Tuple[Dict[str, np.ndarray], StandardScaler, StandardScaler]:
    """
    Розбиває дані на тренувальний/валідаційний/тестовий набори та масштабує їх.
    
    Args:
        X: Вхідні дані.
        Y: Вихідні дані.
        params: Словник з параметрами розбиття.

    Returns:
        Кортеж зі словником даних та навченими скалерами для X та Y.
    """
    n = X.shape[0]
    n_train = int(params['train_size'] * n)
    n_val = int(params['val_size'] * n)

    data_splits = {
        'X_train': X[:n_train], 'Y_train': Y[:n_train],
        'X_val': X[n_train:n_train + n_val], 'Y_val': Y[n_train:n_train + n_val],
        'X_test': X[n_train + n_val:], 'Y_test': Y[n_train + n_val:]
    }

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    data_splits['X_train_scaled'] = x_scaler.fit_transform(data_splits['X_train'])
    data_splits['Y_train_scaled'] = y_scaler.fit_transform(data_splits['Y_train'])
    data_splits['X_val_scaled'] = x_scaler.transform(data_splits['X_val'])
    data_splits['Y_val_scaled'] = y_scaler.transform(data_splits['Y_val'])
    data_splits['X_test_scaled'] = x_scaler.transform(data_splits['X_test'])
    data_splits['Y_test_scaled'] = y_scaler.transform(data_splits['Y_test'])
    
    return data_splits, x_scaler, y_scaler


# =============================================================================
# === БЛОК 2: ІНІЦІАЛІЗАЦІЯ КОМПОНЕНТІВ MPC та EKF ===
# =============================================================================


def train_and_evaluate_model(
    mpc: MPCController,
    data: Dict[str, np.ndarray],
    y_scaler: StandardScaler
) -> Dict[str, float]:
    """
    Навчає модель всередині MPC та оцінює її якість на тестових даних.
    """
    print("Крок 3: Навчання та оцінка моделі процесу...")
    mpc.fit(data['X_train_scaled'], data['Y_train_scaled'])

    y_pred_scaled = mpc.model.predict(data['X_test_scaled'])
    y_pred_orig = y_scaler.inverse_transform(y_pred_scaled)
    
    test_mse = mean_squared_error(data['Y_test'], y_pred_orig)
    print(f"-> Загальна помилка моделі на тестових даних (MSE): {test_mse:.4f}")
    
    metrics = {'test_mse_total': test_mse}
    output_columns = ['conc_fe', 'conc_mass']
    for i, col in enumerate(output_columns):
        rmse = np.sqrt(mean_squared_error(data['Y_test'][:, i], y_pred_orig[:, i]))
        metrics[f'test_rmse_{col}'] = rmse
        print(f"-> RMSE для {col}: {rmse:.3f}")
        
    return metrics

def initialize_mpc_controller_enhanced(
    params: Dict[str, Any],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler
) -> MPCController:
    """
    Ініціалізує покращений MPC контролер з адаптивним trust region.
    """
    print("Крок 2: Ініціалізація покращеного MPC контролера...")
    
    # Створення моделі процесу
    kernel_model = KernelModel(
        model_type=params['model_type'],
        kernel=params['kernel'],
        find_optimal_params=params['find_optimal_params']
    )
    
    # Масштабування уставок та обмежень
    ref_point_scaled = y_scaler.transform(np.array([[params['ref_fe'], params['ref_mass']]]))[0]
    y_max_scaled = y_scaler.transform(np.array([[params['y_max_fe'], params['y_max_mass']]]))[0]

    # Створення цільової функції
    objective = MaxIronMassTrackingObjective(
        λ=params['λ_obj'], w_fe=params['w_fe'], w_mass=params['w_mass'],
        ref_fe=ref_point_scaled[0], ref_mass=ref_point_scaled[1], K_I=params['K_I']
    )
    
    # Розрахунок ваг для м'яких обмежень
    avg_tracking_weight = (params['w_fe'] + params['w_mass']) / 2.
    rho_y_val = avg_tracking_weight * 1000
    rho_du_val = params['λ_obj'] * 100

    # Створення покращеного контролера з новими параметрами
    mpc = MPCController(
        model=kernel_model, 
        objective=objective, 
        x_scaler=x_scaler, 
        y_scaler=y_scaler,
        n_targets=2, 
        horizon=params['Np'], 
        control_horizon=params['Nc'], 
        lag=params['lag'],
        u_min=params['u_min'], 
        u_max=params['u_max'], 
        delta_u_max=params['delta_u_max'],
        use_disturbance_estimator=params['use_disturbance_estimator'],
        y_max=list(y_max_scaled) if params['use_soft_constraints'] else None,
        rho_y=rho_y_val, 
        rho_delta_u=rho_du_val, 
        rho_trust=params['rho_trust'],
        # === НОВІ ПАРАМЕТРИ ===
        adaptive_trust_region=params.get('adaptive_trust_region', True),
        initial_trust_radius=params.get('initial_trust_radius', 1.0),
        min_trust_radius=params.get('min_trust_radius', 0.1),
        max_trust_radius=params.get('max_trust_radius', 5.0),
        trust_decay_factor=params.get('trust_decay_factor', 0.8),
        linearization_check_enabled=params.get('linearization_check_enabled', True),
        max_linearization_distance=params.get('max_linearization_distance', 2.0)
    )
    return mpc

def run_simulation_loop_enhanced(
    true_gen: StatefulDataGenerator,
    mpc: MPCController,
    ekf: ExtendedKalmanFilter,
    df_true: pd.DataFrame,
    data: Dict[str, np.ndarray],
    scalers: Tuple[StandardScaler, StandardScaler],
    params: Dict[str, Any],
    progress_callback: Callable | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Покращений цикл симуляції з моніторингом trust region та якості лінеаризації.
    """
    print("Крок 5: Запуск покращеного циклу симуляції...")
    x_scaler, y_scaler = scalers

    # Початкова ініціалізація (як і раніше)
    n_total = len(df_true) - params['lag'] - 1
    n_train = int(params['train_size'] * n_total)
    n_val   = int(params['val_size'] * n_total)
    test_idx_start = params['lag'] + 1 + n_train + n_val

    hist0_unscaled = df_true[['feed_fe_percent',
                              'ore_mass_flow',
                              'solid_feed_percent']].iloc[
        test_idx_start - (params['lag'] + 1): test_idx_start
    ].values

    mpc.reset_history(hist0_unscaled)
    true_gen.reset_state(hist0_unscaled)

    df_run = df_true.iloc[test_idx_start:]
    d_all  = df_run[['feed_fe_percent', 'ore_mass_flow']].values
    T_sim  = len(df_run) - (params['lag'] + 1)

    # Службові змінні (розширені)
    records = []
    y_true_hist, x_hat_hist, P_hist, innov_hist, R_hist = [], [], [], [], []
    u_seq_hist = []
    d_hat_hist = []
    
    # ✅ ДОДАЄМО ЗМІННІ ДЛЯ ДІАГНОСТИКИ EKF:
    y_true_seq = []     # Реальні вимірювання
    y_pred_seq = []     # Передбачення моделі
    x_est_seq = []      # Оцінки стану EKF
    innovation_seq = [] # Інновації EKF
    
    trust_region_stats_hist = []  # НОВИЙ
    linearization_quality_hist = []  # НОВИЙ
    u_prev = float(hist0_unscaled[-1, 2])

    # Фільтри та детектори аномалій
    window_size = 4
    filt_feed = MovingAverageFilter(window_size)
    filt_ore  = MovingAverageFilter(window_size)

    retrain_cooldown_timer = 0

    # Буфери та налаштування для перенавчання
    if params['enable_retraining']:
        print(f"-> Динамічне перенавчання УВІМКНЕНО. "
              f"Вікно: {params['retrain_window_size']}, "
              f"Період перевірки: {params['retrain_period']}")
        retraining_buffer   = deque(maxlen=params['retrain_window_size'])
        initial_train_data  = list(zip(data['X_train_scaled'],
                                       data['Y_train_scaled']))
        retraining_buffer.extend(initial_train_data)
        innovation_monitor  = deque(maxlen=params['retrain_period'])

    # ONLINE-детектори аномалій
    ad_config = params.get('anomaly_params', {})
    ad_feed_fe = SignalAnomalyDetector(**ad_config)
    ad_ore_flow = SignalAnomalyDetector(**ad_config)

    # Головний цикл симуляції з покращеннями
    for t in range(T_sim):
        if progress_callback:
            progress_callback(t, T_sim, f"Крок симуляції {t + 1}/{T_sim}")

        # 1. Сирі вимірювання
        feed_fe_raw, ore_flow_raw = d_all[t, :]

        # 2. ONLINE-фільтрування аномалій
        feed_fe_filt_anom = ad_feed_fe.update(feed_fe_raw)
        ore_flow_filt_anom = ad_ore_flow.update(ore_flow_raw)

        # 3. Грубе згладжування
        d_filt = np.array([filt_feed.update(feed_fe_filt_anom),
                           filt_ore.update(ore_flow_filt_anom)])

        # 4. EKF: прогноз
        ekf.predict(u_prev, d_filt)

        # 5. Оновлення історії в MPC
        x_est_phys_unscaled = ekf.x_hat[:ekf.n_phys].reshape(params['lag'] + 1, 3)
        mpc.reset_history(x_est_phys_unscaled)
        mpc.d_hat = ekf.x_hat[ekf.n_phys:]

        # Беремо поточний стан і передбачаємо наступний вихід
        current_state = x_est_phys_unscaled.flatten().reshape(1, -1)
        current_state_scaled = x_scaler.transform(current_state)
        y_pred_scaled = mpc.model.predict(current_state_scaled)[0]
        y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled.reshape(1, -1))[0]

        # 6. Оптимізація MPC з покращеннями
        d_seq = np.repeat(d_filt.reshape(1, -1), params['Np'], axis=0)
        u_seq = mpc.optimize(d_seq, u_prev)
        u_cur = u_prev if u_seq is None else float(u_seq[0])

        # ДОДАЙ ДІАГНОСТИКУ ТУТ:
        if t % 10 == 0:  # Кожні 10 кроків
            diagnose_mpc_behavior(mpc, t, u_seq, u_prev, d_seq)
        
        u_cur = u_prev if u_seq is None else float(u_seq[0])

        # 7. Крок «реального» процесу
        y_full = true_gen.step(feed_fe_raw, ore_flow_raw, u_cur)

        # 8. EKF: корекція
        y_meas_unscaled = y_full[['concentrate_fe_percent',
                                  'concentrate_mass_flow']].values.flatten()
        ekf.update(y_meas_unscaled)

        # ✅ ЗБИРАЄМО ДАНІ ДЛЯ ДІАГНОСТИКИ EKF:
        y_true_seq.append(y_meas_unscaled.copy())
        
        # Передбачення моделі (зберігаємо перед update)
        # if hasattr(ekf, 'y_pred') and ekf.y_pred is not None:
        #     y_pred_seq.append(ekf.y_pred.copy())
        # else:
        #     y_pred_seq.append(np.zeros(2))
        y_pred_seq.append(y_pred_unscaled.copy())
        
        # Оцінка стану після update
        x_est_seq.append(ekf.x_hat.copy())
        
        # Інновації
        if hasattr(ekf, 'last_innovation') and ekf.last_innovation is not None:
            innovation_seq.append(ekf.last_innovation.copy())
        else:
            innovation_seq.append(np.zeros(2))

        # 9. Зменшуємо cooldown-таймер
        if retrain_cooldown_timer > 0:
            retrain_cooldown_timer -= 1

        # === НОВА ЛОГІКА: ЗБІР СТАТИСТИКИ TRUST REGION ===
        if hasattr(mpc, 'get_trust_region_stats'):
            trust_stats = mpc.get_trust_region_stats()
            trust_region_stats_hist.append(trust_stats)
            
            # ВИПРАВЛЕННЯ: зберігаємо якість лінеаризації правильно
            if hasattr(mpc, 'linearization_quality_history') and mpc.linearization_quality_history:
                # Зберігаємо останнє значення з історії
                if isinstance(mpc.linearization_quality_history[-1], dict):
                    linearization_quality_hist.append(mpc.linearization_quality_history[-1]['euclidean_distance'])
                else:
                    linearization_quality_hist.append(mpc.linearization_quality_history[-1])

        # 10. Буферизація та можливе перенавчання
        if params['enable_retraining']:
            new_x_unscaled = mpc.x_hist.flatten().reshape(1, -1)
            new_y_unscaled = y_meas_unscaled.reshape(1, -1)

            new_x_scaled = x_scaler.transform(new_x_unscaled)
            new_y_scaled = y_scaler.transform(new_y_unscaled)

            retraining_buffer.append((new_x_scaled[0], new_y_scaled[0]))

            if ekf.last_innovation is not None:
                innov_norm = np.linalg.norm(ekf.last_innovation)
                innovation_monitor.append(innov_norm)

            # Перевірка необхідності перенавчання
            if (t > 0 and
                t % params['retrain_period'] == 0 and
                len(innovation_monitor) == params['retrain_period'] and
                retrain_cooldown_timer == 0):

                avg_innov = float(np.mean(innovation_monitor))

                # === ПОКРАЩЕНА ЛОГІКА ПЕРЕНАВЧАННЯ ===
                # Додаткова перевірка: якість лінеаризації
                should_retrain = avg_innov > params['retrain_innov_threshold']
                
                if (hasattr(mpc, 'linearization_quality_history') and 
                    len(mpc.linearization_quality_history) > 10):
                    
                    # ВИПРАВЛЕННЯ: правильно витягуємо значення відстані
                    if isinstance(mpc.linearization_quality_history[-1], dict):
                        recent_distances = [h['euclidean_distance'] for h in mpc.linearization_quality_history[-10:]]
                    else:
                        recent_distances = mpc.linearization_quality_history[-10:]
                    
                    recent_lin_quality = np.mean(recent_distances)
                    lin_threshold = params.get('retrain_linearization_threshold', 1.5)
                    
                    if recent_lin_quality > lin_threshold:
                        print(f"  -> Додатковий тригер: погана якість лінеаризації ({recent_lin_quality:.3f} > {lin_threshold})")
                        should_retrain = True

                if should_retrain:
                    print(f"\n---> ТРИГЕР ПЕРЕНАВЧАННЯ на кроці {t}! "
                          f"Середня інновація: {avg_innov:.4f} > "
                          f"{params['retrain_innov_threshold']:.4f}")

                    retrain_data = list(retraining_buffer)
                    X_retrain = np.array([p[0] for p in retrain_data])
                    Y_retrain = np.array([p[1] for p in retrain_data])

                    print(f"--> mpc.fit() на {len(X_retrain)} семплах ...")
                    mpc.fit(X_retrain, Y_retrain)
                    print("--> Перенавчання завершено.")
                    
                    # Скидаємо trust region після перенавчання
                    if hasattr(mpc, 'reset_trust_region'):
                        mpc.reset_trust_region()
                        print("--> Trust region скинуто.\n")

                    innovation_monitor.clear()
                    retrain_cooldown_timer = params['retrain_period'] * 2

        # 11. Логування для візуалізації / метрик
        y_true_hist.append(y_meas_unscaled)
        x_hat_hist.append(ekf.x_hat.copy())
        P_hist.append(ekf.P.copy())
        R_hist.append(ekf.R.copy())
        innov_hist.append(
            ekf.last_innovation.copy()
            if ekf.last_innovation is not None
            else np.zeros(ekf.n_dist)
        )

        # Збереження планів MPC та оцінок збурень
        if u_seq is not None:
            u_seq_hist.append(u_seq)
        if mpc.d_hat is not None:
            d_hat_orig = y_scaler.inverse_transform(mpc.d_hat.reshape(1, -1))[0]
            d_hat_hist.append(d_hat_orig)

        y_meas = y_full.iloc[0]
        records.append({
            'feed_fe_percent':      y_meas.feed_fe_percent,
            'ore_mass_flow':        y_meas.ore_mass_flow,
            'solid_feed_percent':   u_cur,
            'conc_fe':              y_meas.concentrate_fe_percent,
            'tail_fe':              y_meas.tailings_fe_percent,
            'conc_mass':            y_meas.concentrate_mass_flow,
            'tail_mass':            y_meas.tailings_mass_flow,
            'mass_pull_pct':        y_meas.mass_pull_percent,
            'fe_recovery_percent':  y_meas.fe_recovery_percent,
        })

        u_prev = u_cur

    if progress_callback:
        progress_callback(T_sim, T_sim, "Симуляція завершена")

    # ✅ ДОДАЄМО ДІАГНОСТИКУ EKF:
    diagnose_ekf_detailed(ekf, y_true_seq, y_pred_seq, x_est_seq, innovation_seq)
        
    # Розширені дані для аналізу
    analysis_data = {
        "y_true": np.vstack(y_true_hist),
        "x_hat": np.vstack(x_hat_hist),
        "P": np.stack(P_hist),
        "innov": np.vstack(innov_hist),
        "R": np.stack(R_hist),
        "u_seq": u_seq_hist,
        "d_hat": np.vstack(d_hat_hist) if d_hat_hist else np.array([]),
        "trust_region_stats": trust_region_stats_hist,
        "linearization_quality": linearization_quality_hist,
        "y_true_seq": y_true_seq,
        "y_pred_seq": y_pred_seq,
        "x_est_seq": x_est_seq,
        "innovation_seq": innovation_seq,
    }

    return pd.DataFrame(records), analysis_data

def initialize_ekf(
    mpc: MPCController,
    scalers: Tuple[StandardScaler, StandardScaler],
    hist0_unscaled: np.ndarray,
    Y_train_scaled: np.ndarray,
    lag: int,
    params: Dict[str, Any]
) -> ExtendedKalmanFilter:
    """
    Ініціалізує розширений фільтр Калмана (EKF).
    """
    print("Крок 4: Ініціалізація фільтра Калмана (EKF)...")
       
    x_scaler, y_scaler = scalers
    n_phys, n_dist = (lag + 1) * 3, 2
    
    # x0_aug = np.hstack([hist0_unscaled.flatten(), np.zeros(n_dist)])
    # ✅ ВИПРАВЛЕННЯ: Розумна початкова оцінка збурень
    # Базуючись на систематичній помилці з попередніх результатів
    initial_disturbances = np.array([0.7, 0.0])  # Близько до Innovation mean: [0.71, 0.04]
    
    x0_aug = np.hstack([hist0_unscaled.flatten(), initial_disturbances])
    
    # P0 = np.eye(n_phys + n_dist) * params['P0']
    # P0[n_phys:, n_phys:] *= 1 
    P0 = np.eye(n_phys + n_dist) * params['P0'] * 1.5  # Було: * 1.0
    P0[n_phys:, n_phys:] *= 10  # Залишити як є

    Q_phys = np.eye(n_phys) * params['Q_phys']
    Q_dist = np.eye(n_dist) * params['Q_dist'] 
    Q = np.block([[Q_phys, np.zeros((n_phys, n_dist))], [np.zeros((n_dist, n_phys)), Q_dist]])
    
    # R = np.diag(np.var(Y_train_scaled, axis=0)) * params['R']
    R = np.diag(np.var(Y_train_scaled, axis=0)) * params['R'] * 0.5
    
    return ExtendedKalmanFilter(
        mpc.model, x_scaler, y_scaler, x0_aug, P0, Q, R, lag,
        beta_R=params.get('beta_R', 0.1), # .get для зворотної сумісності
        q_adaptive_enabled=params.get('q_adaptive_enabled', True),
        q_alpha=params.get('q_alpha', 0.995),
        q_nis_threshold=params.get('q_nis_threshold', 1.8)        
    )

   
# =============================================================================
# === ГОЛОВНА ФУНКЦІЯ-ОРКЕСТРАТОР ===
# =============================================================================

def simulate_mpc(
    reference_df: pd.DataFrame,
    N_data: int = 5000,
    control_pts : int = 1000,
    time_step_s : int = 5,
    dead_times_s : dict = {
        'concentrate_fe_percent': 20.0,
        'tailings_fe_percent': 25.0,
        'concentrate_mass_flow': 20.0,
        'tailings_mass_flow': 25.0
    },
    time_constants_s : dict = {
        'concentrate_fe_percent': 8.0,
        'tailings_fe_percent': 10.0,
        'concentrate_mass_flow': 5.0,
        'tailings_mass_flow': 7.0
    },
    lag: int = 2,
    Np: int = 6,
    Nc: int = 4,
    n_neighbors: int = 5,
    seed: int = 0,
    noise_level: str = 'none',
    model_type: str = 'krr',
    kernel: str = 'rbf',
    find_optimal_params: bool = True,
    λ_obj: float = 0.1,
    K_I: float = 0.01,
    w_fe: float = 7.0,
    w_mass: float = 1.0,
    ref_fe: float = 53.5,
    ref_mass: float = 57.0,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    u_min: float = 20.0,
    u_max: float = 40.0,
    delta_u_max: float = 1.0,
    use_disturbance_estimator: bool = True,
    y_max_fe: float = 54.5,
    y_max_mass: float = 58.0,
    rho_trust: float = 0.1,
    max_trust_radius: float = 5.0,
    adaptive_trust_region: bool = True,
    initial_trust_radius: float =  1.0,
    min_trust_radius: float =  0.5,
    trust_decay_factor: float =  0.8,
    linearization_check_enabled: bool = True,
    max_linearization_distance: float =  2.0,
    retrain_linearization_threshold: float =  1.5,
    use_soft_constraints: bool = True,
    plant_model_type: str = 'rf',
    enable_retraining: bool = True,
    retrain_period: int = 50,
    retrain_window_size: int = 1000,
    retrain_innov_threshold: float = 0.3,
    anomaly_params: dict = {
        'window': 25,
        'spike_z': 4.0,
        'drop_rel': 0.30,
        'freeze_len': 5,
        'enabled': True
    },
    nonlinear_config: dict = {
        'concentrate_fe_percent': ('pow', 2),
        'concentrate_mass_flow': ('pow', 1.5)
    },
    enable_nonlinear: bool =  False,
    run_analysis: bool = True,
    run_evaluation: bool = True,                # ✅ ІСНУЮЧИЙ ПАРАМЕТР
    show_evaluation_plots: bool = False,        # ✅ ДОДАТИ ЦЕЙ РЯДОК
    tolerance_fe_percent: float = 2.0,         # ✅ ДОДАТИ ЦЕЙ РЯДОК
    tolerance_mass_percent: float = 2.0,       # ✅ ДОДАТИ ЦЕЙ РЯДОК     
    P0: float = 1e-2,
    Q_phys: float = 1500,
    Q_dist: float = 1,
    R: float = 0.01,
    q_adaptive_enabled: bool = True,
    q_alpha:float = 0.99,
    q_nis_threshold:float = 1.5,
    progress_callback: Callable[[int, int, str], None] = None
):
    """
    Покращена версія головної функції-оркестратора.
    """
    # Збираємо всі параметри в один словник
    params = locals()
    # params.update(kwargs)  # Додаємо будь-які додаткові параметри
    
    # 1. Підготовка даних (без змін)
    true_gen, df_true, X, Y = prepare_simulation_data(reference_df, params)
    data, x_scaler, y_scaler = split_and_scale_data(X, Y, params)

    # 2. Ініціалізація покращеного MPC
    mpc = initialize_mpc_controller_enhanced(params, x_scaler, y_scaler)
    metrics = train_and_evaluate_model(mpc, data, y_scaler)
    
    # 3. Ініціалізація EKF (без змін)
    n_train_pts = len(data['X_train'])
    n_val_pts = len(data['X_val'])
    test_idx_start = params['lag'] + 1 + n_train_pts + n_val_pts
    hist0_unscaled = df_true[['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']].iloc[
        test_idx_start - (params['lag'] + 1): test_idx_start
    ].values
    
    ekf = initialize_ekf(mpc, (x_scaler, y_scaler), hist0_unscaled, data['Y_train_scaled'], params['lag'], params)

    # 4. Запуск покращеної симуляції
    results_df, analysis_data = run_simulation_loop_enhanced(
        true_gen, mpc, ekf, df_true, data, (x_scaler, y_scaler), params, 
        params.get('progress_callback')
    )
    
    # 5. Розширений аналіз результатів
    test_idx_start = params['lag'] + 1 + len(data['X_train']) + len(data['X_val'])
    analysis_data['d_all_test'] = df_true.iloc[test_idx_start:][['feed_fe_percent','ore_mass_flow']].values
    
    if params.get('run_analysis', True):
        run_post_simulation_analysis_enhanced(results_df, analysis_data, params)
    
    # ✅ НОВИЙ БЛОК: ОЦІНЮВАННЯ ЕФЕКТИВНОСТІ
    if params.get('run_evaluation', True):
        print("\n" + "="*60)
        print("🎯 ОЦІНЮВАННЯ ЕФЕКТИВНОСТІ MPC СИСТЕМИ")
        print("="*60)
        try:
            eval_results = evaluate_simulation(results_df, analysis_data, params)
            print_evaluation_report(eval_results, detailed=True)
            
            # ✅ ДОДАЄМО ВІЗУАЛІЗАЦІЮ
            if params.get('show_evaluation_plots', False):
                print("\n📊 Створення графіків оцінки...")
                try:
                    from evaluation_simple import create_evaluation_plots
                    create_evaluation_plots(results_df, eval_results, params)
                except Exception as plot_error:
                    print(f"⚠️ Помилка при створенні графіків: {plot_error}")
                    
        except Exception as e:
            print(f"⚠️ Помилка при оцінюванні: {e}")
            print("Продовжуємо без оцінювання...")
        print("="*60)
    
    return results_df, metrics

if __name__ == '__main__':
    
    def my_progress(step, total, msg):
        """Простий callback для виводу прогресу в консоль"""
        if step % 20 == 0 or step == total:
            print(f"[{step}/{total}] {msg}")

    # Завантажуємо дані
    try:
        hist_df = pd.read_parquet('processed.parquet')
        print("✅ Дані завантажено успішно")
    except FileNotFoundError:
        print("❌ Помилка: файл 'processed.parquet' не знайдено.")
        exit(1)
    
    # Створюємо стандартні конфігурації якщо їх немає
    available_configs = list_configs()
    if not available_configs:
        print("📁 Створюємо стандартні конфігурації...")
        create_default_configs()
        available_configs = list_configs()
    
    # Показуємо доступні конфігурації
    print(f"\n📋 Доступні конфігурації: {', '.join(available_configs)}")
    
    # Вибір базової конфігурації
    print(f"\nОберіть базову конфігурацію:")
    for i, config in enumerate(available_configs, 1):
        print(f"{i}. {config}")
    
    choice = input(f"Ваш вибір (1-{len(available_configs)}, за замовчуванням 1): ").strip()
    
    try:
        config_index = int(choice) - 1 if choice else 0
        if 0 <= config_index < len(available_configs):
            selected_config = available_configs[config_index]
        else:
            selected_config = available_configs[0]
    except (ValueError, IndexError):
        selected_config = available_configs[0]
    
    print(f"🎯 Обрано базову конфігурацію: {selected_config}")
    
    # Завантаження базової конфігурації для показу
    base_config = load_config(selected_config)
    
    # Показуємо поточні ключові параметри
    key_params = ['model_type', 'Np', 'Nc', 'ref_fe', 'ref_mass', 'w_fe', 'w_mass', 'λ_obj', 'N_data', 'control_pts']
    print(f"\n📊 Поточні ключові параметри:")
    for param in key_params:
        if param in base_config:
            print(f"   • {param}: {base_config[param]}")
    
    # Запитуємо про ручні корегування
    want_adjustments = input(f"\nХочете внести ручні корегування? (y/N): ").strip().lower()
    
    manual_overrides = {}
    if want_adjustments in ['y', 'yes', 'так', 'т']:
        manual_overrides = prompt_manual_adjustments(base_config)
        
        if manual_overrides:
            print(f"\n✅ Заплановано {len(manual_overrides)} корегувань")
        else:
            print("ℹ️ Корегування не внесено")
    
    # ✅ ДОДАНО: Запитуємо про оцінювання та візуалізацію
    want_evaluation = input(f"\nВключити оцінку ефективності? (Y/n): ").strip().lower()
    run_evaluation = want_evaluation not in ['n', 'no', 'ні']
    
    show_evaluation_plots = False
    if run_evaluation:
        want_plots = input(f"Показати графіки оцінки? (Y/n): ").strip().lower()
        show_evaluation_plots = want_plots not in ['n', 'no', 'ні']
    
    # Запуск симуляції
    print(f"\n🚀 Запуск симуляції...")
    print("=" * 50)
    
    try:
        result = simulate_mpc_with_config(
            hist_df,
            config_name=selected_config,
            manual_overrides=manual_overrides,
            progress_callback=my_progress,
            run_evaluation=run_evaluation,  # ✅ ПЕРЕДАЄМО ПАРАМЕТР
            show_evaluation_plots=show_evaluation_plots  # ✅ ПЕРЕДАЄМО ПАРАМЕТР ВІЗУАЛІЗАЦІЇ
        )
        
        if result is None:
            print("❌ simulate_mpc_with_config повернув None")
            exit(1)
        
        results_df, metrics = result
        
        # Виводимо результати
        print("\n📊 РЕЗУЛЬТАТИ СИМУЛЯЦІЇ:")
        print("=" * 40)
        print(f"📈 Оброблено кроків: {len(results_df)}")
        
        # Ключові метрики
        key_metrics = ['test_mse_total', 'test_rmse_conc_fe', 'test_rmse_conc_mass']
        for metric in key_metrics:
            if metric in metrics:
                print(f"📊 {metric}: {metrics[metric]:.6f}")
        
        # Показуємо збережені файли
        saved_results = list_saved_results()
        if saved_results:
            latest = saved_results[0]  # Останній збережений файл
            print(f"\n💾 Останній збережений файл:")
            print(f"   📁 {latest['file']}")
            print(f"   📊 Розмір: {latest['size_mb']:.2f} MB")
        
        print("\n✅ Симуляція завершена успішно!")
        
    except Exception as e:
        print(f"\n❌ Помилка під час симуляції: {e}")
        import traceback
        traceback.print_exc()