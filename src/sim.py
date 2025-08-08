# sim.py

import numpy as np
import pandas as pd
import inspect
import traceback  

from typing import Callable, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from collections import deque

from data_gen import StatefulDataGenerator
from model import KernelModel
from objectives import MaxIronMassTrackingObjective
from mpc import MPCController
# from conf_manager import MPCConfigManager
from utils import (
    run_post_simulation_analysis_enhanced,  diagnose_mpc_behavior, diagnose_ekf_detailed
)
from ekf import ExtendedKalmanFilter
from anomaly_detector import SignalAnomalyDetector
from maf import MovingAverageFilter
from benchmark import benchmark_model_training, benchmark_mpc_solve_time
from conf_manager import config_manager
from typing import Optional, List

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
        # ✅ ДОДАЄМО ДАНІ ДЛЯ ПОДАЛЬШОГО АНАЛІЗУ:
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

def collect_performance_metrics(
    mpc: MPCController,
    data: Dict[str, np.ndarray],
    model_config: Dict
) -> Dict[str, float]:
    """Збирає метрики продуктивності для статті"""
    
    print("📊 Збираю метрики продуктивності...")
    
    # 1. Бенчмарк навчання та прогнозу
    model_configs = [model_config]  # Поточна конфігурація
    training_metrics = benchmark_model_training(
        data['X_train_scaled'], 
        data['Y_train_scaled'], 
        model_configs
    )
    
    # 2. Бенчмарк MPC
    mpc_metrics = benchmark_mpc_solve_time(mpc, n_iterations=50)
    
    # 3. Загальний час циклу (приблизно)
    total_cycle_time = (
        training_metrics[f"{model_config['model_type']}-{model_config.get('kernel', 'default')}_predict_time"] +
        training_metrics[f"{model_config['model_type']}-{model_config.get('kernel', 'default')}_linearize_time"] +
        mpc_metrics["mpc_solve_mean"]
    )
    
    # 4. Об'єднуємо все
    all_metrics = {**training_metrics, **mpc_metrics}
    all_metrics["total_cycle_time"] = total_cycle_time
    
    return all_metrics
    
# ========================================  
# ОСНОВНА ФУНКЦІЯ СИМУЛЯЦІЇ (БЕЗ ЦИКЛІЧНОГО ВИКЛИКУ)  
# ========================================  

def simulate_mpc_core(  
    reference_df: pd.DataFrame,             # DataFrame з референсними даними  
    N_data: int = 5000,                     # Загальна кількість точок даних  
    control_pts: int = 1000,                # Кількість кроків MPC  
    time_step_s: int = 5,                   # Часовий крок  
    dead_times_s: dict = {  
        'concentrate_fe_percent': 20.0,  
        'tailings_fe_percent': 25.0,  
        'concentrate_mass_flow': 20.0,  
        'tailings_mass_flow': 25.0  
    },                                      # Транспортна затримка  
    time_constants_s: dict = {  
        'concentrate_fe_percent': 8.0,  
        'tailings_fe_percent': 10.0,  
        'concentrate_mass_flow': 5.0,  
        'tailings_mass_flow': 7.0  
    },                                      # Інерційність параметрів  
    lag: int = 2,                           # Кроки затримки моделі  
    Np: int = 6,                            # Горизонт прогнозування MPC  
    Nc: int = 4,                            # Горизонт керування MPC  
    n_neighbors: int = 5,                   # Кількість сусідів для KNN  
    seed: int = 0,                          # Зерно для RNG  
    noise_level: str = 'none',              # Рівень шуму  
    model_type: str = 'krr',                # Тип моделі MPC  
    kernel: str = 'rbf',                    # Тип ядра  
    linear_type: str = 'ridge',             # ols, ridge, lasso  
    poly_degree: int = 2,                   # Степінь поліному  
    alpha: float = 1.0,                     # Регуляризація  
    find_optimal_params: bool = True,       # Пошук оптимальних параметрів  
    λ_obj: float = 0.1,                     # Коефіцієнт згладжування  
    K_I: float = 0.01,                      # Інтегральний коефіцієнт  
    w_fe: float = 7.0,                      # Вага для Fe  
    w_mass: float = 1.0,                    # Вага для масової витрати  
    ref_fe: float = 53.5,                   # Референсне значення Fe  
    ref_mass: float = 57.0,                 # Референсне значення маси  
    train_size: float = 0.7,                # Розмір навчальної вибірки  
    val_size: float = 0.15,                 # Розмір валідаційної вибірки  
    test_size: float = 0.15,                # Розмір тестової вибірки  
    u_min: float = 20.0,                    # Мінімальне керування  
    u_max: float = 40.0,                    # Максимальне керування  
    delta_u_max: float = 1.0,               # Максимальна зміна керування  
    use_disturbance_estimator: bool = True, # Використання EKF  
    y_max_fe: float = 54.5,                 # Верхня межа Fe  
    y_max_mass: float = 58.0,               # Верхня межа маси  
    rho_trust: float = 0.1,                 # Коефіцієнт довіри  
    max_trust_radius: float = 5.0,          # Максимальний радіус довіри  
    adaptive_trust_region: bool = True,     # Адаптивна область довіри  
    initial_trust_radius: float = 1.0,      # Початковий радіус довіри  
    min_trust_radius: float = 0.5,          # Мінімальний радіус довіри  
    trust_decay_factor: float = 0.8,        # Фактор зменшення довіри  
    linearization_check_enabled: bool = True,           # Перевірка лінеаризації  
    max_linearization_distance: float = 2.0,            # Максимальна відстань лінеаризації  
    retrain_linearization_threshold: float = 1.5,       # Поріг перенавчання  
    use_soft_constraints: bool = True,      # М'які обмеження  
    plant_model_type: str = 'rf',           # Тип моделі об'єкта  
    enable_retraining: bool = True,         # Перенавчання моделі  
    retrain_period: int = 50,               # Періодичність перенавчання  
    retrain_window_size: int = 1000,        # Розмір вікна перенавчання  
    retrain_innov_threshold: float = 0.3,   # Поріг інновації EKF  
    anomaly_params: dict = {  
        'window': 25,  
        'spike_z': 4.0,  
        'drop_rel': 0.30,  
        'freeze_len': 5,  
        'enabled': True  
    },                                      # Параметри детектора аномалій  
    nonlinear_config: dict = {  
        'concentrate_fe_percent': ('pow', 2),  
        'concentrate_mass_flow': ('pow', 1.5)  
    },                                      # Нелінійна конфігурація  
    enable_nonlinear: bool = False,         # Нелінійний режим  
    run_analysis: bool = True,              # Аналіз результатів  
    P0: float = 1e-2,                      # Початкова коваріація EKF  
    Q_phys: float = 1500,                  # Коваріація процесу  
    Q_dist: float = 1,                     # Коваріація збурень  
    R: float = 0.01,                       # Коваріація вимірювань  
    q_adaptive_enabled: bool = True,        # Адаптивна Q матриця  
    q_alpha: float = 0.99,                 # Фактор згладжування Q  
    q_nis_threshold: float = 1.5,          # Поріг NIS  
    progress_callback: Callable[[int, int, str], None] = None  # Callback прогресу  
) -> Tuple[pd.DataFrame, Dict]:  
    """  
    Основна функція симуляції MPC для магнітної сепарації  
    
    Returns:  
        Tuple[pd.DataFrame, Dict]: (results_df, metrics)  
    """  
    
    # Збираємо всі параметри в словник (замість locals() щоб уникнути проблем)  
    params = {  
        'N_data': N_data, 'control_pts': control_pts, 'time_step_s': time_step_s,  
        'dead_times_s': dead_times_s, 'time_constants_s': time_constants_s,  
        'lag': lag, 'Np': Np, 'Nc': Nc, 'n_neighbors': n_neighbors,  
        'seed': seed, 'noise_level': noise_level, 'model_type': model_type,  
        'kernel': kernel, 'linear_type': linear_type, 'poly_degree': poly_degree,  
        'alpha': alpha, 'find_optimal_params': find_optimal_params,  
        'λ_obj': λ_obj, 'K_I': K_I, 'w_fe': w_fe, 'w_mass': w_mass,  
        'ref_fe': ref_fe, 'ref_mass': ref_mass, 'train_size': train_size,  
        'val_size': val_size, 'test_size': test_size, 'u_min': u_min,  
        'u_max': u_max, 'delta_u_max': delta_u_max,  
        'use_disturbance_estimator': use_disturbance_estimator,  
        'y_max_fe': y_max_fe, 'y_max_mass': y_max_mass,  
        'rho_trust': rho_trust, 'max_trust_radius': max_trust_radius,  
        'adaptive_trust_region': adaptive_trust_region,  
        'initial_trust_radius': initial_trust_radius,  
        'min_trust_radius': min_trust_radius,  
        'trust_decay_factor': trust_decay_factor,  
        'linearization_check_enabled': linearization_check_enabled,  
        'max_linearization_distance': max_linearization_distance,  
        'retrain_linearization_threshold': retrain_linearization_threshold,  
        'use_soft_constraints': use_soft_constraints,  
        'plant_model_type': plant_model_type,   
        'enable_retraining': enable_retraining,  
        'retrain_period': retrain_period,   
        'retrain_window_size': retrain_window_size,  
        'retrain_innov_threshold': retrain_innov_threshold,  
        'anomaly_params': anomaly_params,   
        'nonlinear_config': nonlinear_config,  
        'enable_nonlinear': enable_nonlinear,   
        'run_analysis': run_analysis,  
        'P0': P0, 'Q_phys': Q_phys, 'Q_dist': Q_dist, 'R': R,  
        'q_adaptive_enabled': q_adaptive_enabled,   
        'q_alpha': q_alpha,  
        'q_nis_threshold': q_nis_threshold,   
        'progress_callback': progress_callback  
    }  
    
    try:  
        # 1. Підготовка даних  
        true_gen, df_true, X, Y = prepare_simulation_data(reference_df, params)  
        data, x_scaler, y_scaler = split_and_scale_data(X, Y, params)  

        # 2. Ініціалізація MPC  
        mpc = initialize_mpc_controller_enhanced(params, x_scaler, y_scaler)  
        metrics = train_and_evaluate_model(mpc, data, y_scaler)  

        # 3. Бенчмарк метрики  
        perf_metrics = collect_performance_metrics(mpc, data, {  
            'model_type': params['model_type'],  
            'kernel': params.get('kernel', 'default'),  
            'linear_type': params.get('linear_type', 'default'),  
            'poly_degree': params.get('poly_degree', 1),  
            'find_optimal_params': params.get('find_optimal_params', False)  
        })  
        
        metrics.update(perf_metrics)  
        
        # 4. Ініціалізація EKF  
        n_train_pts = len(data['X_train'])  
        n_val_pts = len(data['X_val'])  
        test_idx_start = params['lag'] + 1 + n_train_pts + n_val_pts  
        hist0_unscaled = df_true[['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']].iloc[  
            test_idx_start - (params['lag'] + 1): test_idx_start  
        ].values  
        
        ekf = initialize_ekf(mpc, (x_scaler, y_scaler), hist0_unscaled, data['Y_train_scaled'], params['lag'], params)  

        # 5. Запуск симуляції  
        results_df, analysis_data = run_simulation_loop_enhanced(  
            true_gen, mpc, ekf, df_true, data, (x_scaler, y_scaler), params,   
            params.get('progress_callback')  
        )  
        
        # 6. Аналіз результатів  
        test_idx_start = params['lag'] + 1 + len(data['X_train']) + len(data['X_val'])  
        analysis_data['d_all_test'] = df_true.iloc[test_idx_start:][['feed_fe_percent','ore_mass_flow']].values  
        
        if params.get('run_analysis', True):  
            run_post_simulation_analysis_enhanced(results_df, analysis_data, params)  
        
        return results_df, metrics  
        
    except Exception as e:  
        print(f"❌ Помилка в simulate_mpc_core: {e}")  
        traceback.print_exc()  
        raise  


# ========================================
# WRAPPER ФУНКЦІЯ З КОНФІГУРАЦІЯМИ
# ========================================

def simulate_mpc_with_config(
    hist_df: pd.DataFrame, 
    config: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable] = None,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Wrapper функція з підтримкою конфігурацій + збереження конфігурації в результати
    """
    
    # 1. Збираємо повну конфігурацію (як і раніше)
    if config:
        print(f"📋 Завантажуємо профіль конфігурації: '{config}'")
        try:
            params = config_manager.load_config(config)
            print(f"   ✅ Профіль '{config}' завантажено успішно")
        except Exception as e:
            print(f"   ❌ Помилка завантаження: {e}")
            params = {}
    else:
        params = {}
    
    if config_overrides:
        print(f"🔧 Застосовуємо {len(config_overrides)} override параметрів")
        params.update(config_overrides)
    
    if kwargs:
        print(f"⚙️ Застосовуємо {len(kwargs)} додаткових параметрів")
        params.update(kwargs)
    
    # 🆕 ЗБЕРІГАЄМО ПОВНУ КОНФІГУРАЦІЮ ДЛЯ РЕЗУЛЬТАТІВ
    full_config_info = {
        'config_source': config if config else 'default',
        'config_overrides': config_overrides.copy() if config_overrides else {},
        'kwargs_applied': kwargs.copy(),
        'final_params': params.copy(),  # 📋 Повна конфігурація що використовувалася
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_params_count': len(params)
    }
    
    # Фільтруємо для simulate_mpc_core
    core_signature = inspect.signature(simulate_mpc_core)
    valid_params = set(core_signature.parameters.keys())
    sim_params = {k: v for k, v in params.items() if k in valid_params}
    
    if progress_callback:
        sim_params['progress_callback'] = progress_callback
    
    print(f"🚀 Передаємо {len(sim_params)} параметрів в simulate_mpc_core")
    
    try:
        results, metrics = simulate_mpc_core(hist_df, **sim_params)
        
        # 🆕 ДОДАЄМО КОНФІГУРАЦІЮ ДО РЕЗУЛЬТАТІВ DataFrame
        print("💾 Додаємо інформацію про конфігурацію до результатів...")
        
        # Створюємо колонки з конфігурацією (константні для всіх рядків)
        results['config_source'] = full_config_info['config_source']
        results['config_timestamp'] = full_config_info['timestamp']
        
        # Ключові параметри як окремі колонки для легкого аналізу
        key_params = ['model_type', 'kernel', 'linear_type', 'Np', 'Nc', 
                     'w_fe', 'w_mass', 'ref_fe', 'ref_mass', 'λ_obj']
        
        for param in key_params:
            if param in full_config_info['final_params']:
                results[f'cfg_{param}'] = full_config_info['final_params'][param]
        
        # 🆕 ДОДАЄМО КОНФІГУРАЦІЮ ДО МЕТРИК
        metrics['config_info'] = full_config_info
        metrics['config_summary'] = {
            'source': full_config_info['config_source'],
            'model_type': full_config_info['final_params'].get('model_type', 'unknown'),
            'kernel': full_config_info['final_params'].get('kernel', 'unknown'),
            'horizons': f"Np={full_config_info['final_params'].get('Np', '?')}, Nc={full_config_info['final_params'].get('Nc', '?')}",
            'weights': f"w_fe={full_config_info['final_params'].get('w_fe', '?')}, w_mass={full_config_info['final_params'].get('w_mass', '?')}"
        }
        
        print("✅ Конфігурацію додано до результатів")
        return results, metrics
        
    except Exception as e:
        print(f"❌ Помилка симуляції: {e}")
        traceback.print_exc()
        raise


# 🆕 ФУНКЦІЯ ДЛЯ АНАЛІЗУ КОНФІГУРАЦІЇ З РЕЗУЛЬТАТІВ
def analyze_results_config(results_df: pd.DataFrame, metrics: Dict = None) -> None:
    """
    Аналізує конфігурацію симуляції з результатів
    
    Args:
        results_df: DataFrame з результатами симуляції
        metrics: Словник метрик (опціонально)
    """
    
    print("="*60)
    print("📋 АНАЛІЗ КОНФІГУРАЦІЇ СИМУЛЯЦІЇ")
    print("="*60)
    
    # Базова інформація з DataFrame
    if 'config_source' in results_df.columns:
        config_source = results_df['config_source'].iloc[0]
        print(f"🎯 Джерело конфігурації: {config_source}")
    
    if 'config_timestamp' in results_df.columns:
        timestamp = results_df['config_timestamp'].iloc[0]
        print(f"🕒 Час симуляції: {timestamp}")
    
    # Ключові параметри з колонок
    config_columns = [col for col in results_df.columns if col.startswith('cfg_')]
    if config_columns:
        print(f"\n🔧 КЛЮЧОВІ ПАРАМЕТРИ ({len(config_columns)}):")
        for col in sorted(config_columns):
            param_name = col.replace('cfg_', '')
            value = results_df[col].iloc[0]
            print(f"   {param_name}: {value}")
    
    # Детальна інформація з metrics
    if metrics and 'config_info' in metrics:
        config_info = metrics['config_info']
        
        print(f"\n📊 СТАТИСТИКА КОНФІГУРАЦІЇ:")
        print(f"   Загалом параметрів: {config_info['total_params_count']}")
        print(f"   Override параметрів: {len(config_info['config_overrides'])}")
        print(f"   Kwargs параметрів: {len(config_info['kwargs_applied'])}")
        
        if config_info['config_overrides']:
            print(f"\n🔄 ПЕРЕВИЗНАЧЕНІ ПАРАМЕТРИ:")
            for key, value in config_info['config_overrides'].items():
                print(f"   {key}: {value}")
        
        if config_info['kwargs_applied']:
            print(f"\n⚙️ ДОДАТКОВІ ПАРАМЕТРИ (kwargs):")
            for key, value in config_info['kwargs_applied'].items():
                print(f"   {key}: {value}")
    
    # Аналіз результатів
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    non_config_cols = [col for col in numeric_cols if not col.startswith('cfg_')]
    
    if non_config_cols:
        print(f"\n📈 РЕЗУЛЬТАТИ СИМУЛЯЦІЇ:")
        print(f"   Кроків симуляції: {len(results_df)}")
        if 'y_fe_pred' in results_df.columns and 'y_mass_pred' in results_df.columns:
            fe_mean = results_df['y_fe_pred'].mean()
            mass_mean = results_df['y_mass_pred'].mean()
            print(f"   Середнє Fe: {fe_mean:.2f}")
            print(f"   Середня маса: {mass_mean:.2f}")
    
    print("="*60)


# 🆕 ФУНКЦІЯ ПОРІВНЯННЯ КОНФІГУРАЦІЙ
def compare_simulation_configs(*results_and_metrics_pairs) -> pd.DataFrame:
    """
    Порівнює конфігурації різних симуляцій
    
    Args:
        *results_and_metrics_pairs: Пари (results_df, metrics) для порівняння
    
    Returns:
        DataFrame з порівнянням конфігурацій
    """
    
    comparison_data = []
    
    for i, (results_df, metrics) in enumerate(results_and_metrics_pairs):
        row = {'simulation': f'Run_{i+1}'}
        
        # Базова інформація
        if 'config_source' in results_df.columns:
            row['config_source'] = results_df['config_source'].iloc[0]
        
        # Параметри з колонок
        config_cols = [col for col in results_df.columns if col.startswith('cfg_')]
        for col in config_cols:
            param_name = col.replace('cfg_', '')
            row[param_name] = results_df[col].iloc[0]
        
        # Метрики якості
        if isinstance(metrics, dict):
            for metric_key in ['rmse_fe', 'rmse_mass', 'r2_fe', 'r2_mass']:
                if metric_key in metrics:
                    row[metric_key] = metrics[metric_key]
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("📊 ПОРІВНЯННЯ КОНФІГУРАЦІЙ:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


# ========================================
# АЛИАС ДЛЯ ЗВОРОТНОЇ СУМІСНОСТІ
# ========================================

# Головна функція для використання
simulate_mpc = simulate_mpc_with_config

print("✅ simulate_mpc_with_config готовий до використання!")
print("🔧 Функція config_manager.load_config() буде викликатися при вказанні параметра 'config'")


# ========================================
# ПРИКЛАД ВИКОРИСТАННЯ В __main__
# ========================================

# if __name__ == '__main__':
    
#     def my_progress(step, total, msg):
#         print(f"[{step}/{total}] {msg}")

#     # Завантаження даних
#     try:
#         hist_df = pd.read_parquet('processed.parquet')
#         print(f"✅ Дані завантажено: {hist_df.shape}")
#     except FileNotFoundError:
#         try:
#             hist_df = pd.read_parquet('/content/KModel/src/processed.parquet')
#             print(f"✅ Дані завантажено з /content/: {hist_df.shape}")
#         except Exception as e:
#             print(f"❌ Помилка завантаження даних: {e}")
#             hist_df = None

#     if hist_df is not None:
#         print("\n🎯 ТЕСТУЄМО ЗАВАНТАЖЕННЯ КОНФІГУРАЦІЇ:")
        
#         # 🔥 ТУТ ВІДБУВАЄТЬСЯ ВИКЛИК config_manager.load_config():
#         try:
#             # Спочатку перевіряємо доступні конфігурації
#             if hasattr(config_manager, 'list_configs'):
#                 available_configs = config_manager.list_configs()
#                 print(f"📁 Доступні конфігурації: {available_configs}")
            
#             # Запускаємо симуляцію з конфігурацією
#             results, metrics = simulate_mpc(
#                 hist_df, 
#                 config='oleksandr_original',  # 🎯 ТУТ ВИКЛИКАЄТЬСЯ config_manager.load_config('oleksandr_original')
#                 config_overrides={
#                     'run_analysis':False
#                 },
#                 progress_callback=my_progress
#             )
            
#             print("\n📊 Результати симуляції:")
#             if isinstance(metrics, dict):
#                 for key in ['rmse_fe', 'rmse_mass', 'config_used']:
#                     if key in metrics:
#                         print(f"   {key}: {metrics[key]}")
            
#             # Зберігаємо результати
#             results.to_parquet('mpc_simulation_results.parquet')
#             print("💾 Результати збережено")

#             loaded_results = pd.read_parquet('mpc_simulation_results.parquet')
#             analyze_results_config(loaded_results, metrics)

#         except Exception as e:
#             print(f"❌ Помилка тестування: {e}")
#             traceback.print_exc()
#     else:
#         print("💥 Дані не завантажено")

# print("\n🎉 Модуль sim.py готовий!")

# ---- ЕКСПЕРИМЕНТ
''' ==================================='''

import numpy as np  
import pandas as pd  
import time  
import json  
import os  
from datetime import datetime

def run_model_comparison_experiment(  
    hist_df: pd.DataFrame,  
    base_config: str = 'oleksandr_original',  
    n_repeats: int = 5,  
    results_dir: str = 'model_comparison_experiment',  
    save_individual: bool = True  
) -> Dict[str, Dict]:  
    """  
    Запуск експерименту порівняння моделей з повтореннями  
    
    Returns:  
        Dict з результатами всіх експериментів  
    """  
    
    # Конфігурація експерименту  
    models_config = {  
        'KRR_RBF': {  
            'model_type': 'krr',  
            'kernel': 'rbf',  
            'find_optimal_params': True  
        },  
        'SVR_RBF': {  
            'model_type': 'svr',   
            'kernel': 'rbf',  
            'find_optimal_params': True  
        },  
        'GPR_RBF': {  
            'model_type': 'gpr',  
            'kernel': 'rbf',   
            'find_optimal_params': True  
        },  
        'LINEAR_RIDGE': {  
            'model_type': 'linear',  
            'linear_type': 'ridge',  
            'find_optimal_params': True  
        }  
    }  
    
    # Створюємо директорію  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
    full_results_dir = f"{results_dir}_{timestamp}"  
    os.makedirs(full_results_dir, exist_ok=True)  
    
    print("🔬 ПОЧАТОК ЕКСПЕРИМЕНТУ ПОРІВНЯННЯ МОДЕЛЕЙ")  
    print("="*70)  
    print(f"📊 Моделей: {len(models_config)}")  
    print(f"🔄 Повторень: {n_repeats}")  
    print(f"📁 Директорія результатів: {full_results_dir}")  
    print(f"⚡ Загалом запусків: {len(models_config) * n_repeats}")  
    print("="*70)  
    
    all_results = {}  
    experiment_summary = []  
    
    total_runs = len(models_config) * n_repeats  
    current_run = 0  
    
    for model_name, model_params in models_config.items():  
        print(f"\n🧪 МОДЕЛЬ: {model_name}")  
        print(f"   Параметри: {model_params}")  
        
        model_results = {  
            'runs': {},  
            'summary_stats': {},  
            'model_config': model_params  
        }  
        
        run_metrics = []  
        
        for repeat in range(n_repeats):  
            current_run += 1  
            run_id = f"{model_name}_run_{repeat+1}"  
            
            print(f"\n   🎯 Запуск {repeat+1}/{n_repeats} ({current_run}/{total_runs})")  
            
            try:  
                # Змінюємо seed для кожного повтору  
                run_start = time.time()  
                
                results_df, metrics = simulate_mpc(  
                    hist_df,  
                    config=base_config,  
                    config_overrides=model_params,  
                    seed=repeat * 42,  # Різні seeds для повторюваності  
                    run_analysis=False  
                )  
                
                run_time = time.time() - run_start  
                
                # Додаємо метаінформацію  
                results_df['experiment_model'] = model_name  
                results_df['experiment_run'] = repeat + 1  
                results_df['experiment_run_id'] = run_id  
                
                metrics['experiment_info'] = {  
                    'model_name': model_name,  
                    'run_number': repeat + 1,  
                    'run_id': run_id,  
                    'seed_used': repeat * 42,  
                    'run_time_seconds': run_time  
                }  
                
                # Зберігаємо індивідуальні результати  
                if save_individual:  
                    results_df.to_parquet(f"{full_results_dir}/{run_id}_results.parquet")  
                    
                    with open(f"{full_results_dir}/{run_id}_metrics.json", 'w') as f:  
                        json_metrics = {}  
                        for key, value in metrics.items():  
                            try:  
                                json.dumps(value)  # Перевіряємо JSON serializable  
                                json_metrics[key] = value  
                            except:  
                                json_metrics[key] = str(value)  
                        json.dump(json_metrics, f, indent=2)  
                
                # Збираємо метрики для статистики  
                run_summary = {  
                    'run_id': run_id,  
                    'model': model_name,  
                    'run_number': repeat + 1,  
                    'run_time': run_time  
                }  
                
                # Додаємо ключові метрики  
                key_metrics = ['rmse_fe', 'rmse_mass', 'mae_fe', 'mae_mass', 'r2_fe', 'r2_mass']  
                for metric in key_metrics:  
                    if metric in metrics:  
                        run_summary[metric] = metrics[metric]  
                
                run_metrics.append(run_summary)  
                model_results['runs'][run_id] = (results_df, metrics)  
                experiment_summary.append(run_summary)  
                
                print(f"      ✅ Завершено за {run_time:.1f}с")  
                if 'rmse_fe' in metrics:  
                    print(f"      📊 RMSE: Fe={metrics['rmse_fe']:.4f}, Mass={metrics.get('rmse_mass', 'N/A'):.4f}")  
                
            except Exception as e:  
                print(f"      ❌ ПОМИЛКА: {e}")  
                run_summary = {  
                    'run_id': run_id,  
                    'model': model_name,  
                    'run_number': repeat + 1,  
                    'error': str(e)  
                }  
                experiment_summary.append(run_summary)  
                continue  
        
        # Статистика по моделі  
        if run_metrics:  
            metrics_df = pd.DataFrame(run_metrics)  
            
            model_stats = {}  
            for metric in ['rmse_fe', 'rmse_mass', 'r2_fe', 'r2_mass', 'run_time']:  
                if metric in metrics_df.columns:  
                    model_stats[f'{metric}_mean'] = metrics_df[metric].mean()  
                    model_stats[f'{metric}_std'] = metrics_df[metric].std()  
                    model_stats[f'{metric}_min'] = metrics_df[metric].min()  
                    model_stats[f'{metric}_max'] = metrics_df[metric].max()  
            
            model_results['summary_stats'] = model_stats  
            
            print(f"   📈 СТАТИСТИКА {model_name}:")  
            print(f"      RMSE Fe: {model_stats.get('rmse_fe_mean', 0):.4f} ± {model_stats.get('rmse_fe_std', 0):.4f}")  
            print(f"      RMSE Mass: {model_stats.get('rmse_mass_mean', 0):.4f} ± {model_stats.get('rmse_mass_std', 0):.4f}")  
            print(f"      R² Fe: {model_stats.get('r2_fe_mean', 0):.4f} ± {model_stats.get('r2_fe_std', 0):.4f}")  
            print(f"      Час: {model_stats.get('run_time_mean', 0):.1f}с ± {model_stats.get('run_time_std', 0):.1f}с")  
        
        all_results[model_name] = model_results  
    
    # Зберігаємо загальну статистику  
    summary_df = pd.DataFrame(experiment_summary)  
    summary_df.to_csv(f"{full_results_dir}/experiment_summary.csv", index=False)  
    
    # Створюємо порівняльну таблицю  
    comparison_table = create_model_comparison_table(all_results)  
    comparison_table.to_csv(f"{full_results_dir}/model_comparison.csv", index=False)  
    
    print("\n" + "="*70)  
    print("🏁 ЕКСПЕРИМЕНТ ЗАВЕРШЕНО")  
    print(f"📁 Результати збережено в: {full_results_dir}")  
    print("="*70)  
    
    return all_results, comparison_table  


def create_model_comparison_table(all_results: Dict) -> pd.DataFrame:  
    """Створює таблицю порівняння моделей"""  
    
    comparison_data = []  
    
    for model_name, model_data in all_results.items():  
        if 'summary_stats' in model_data and model_data['summary_stats']:  
            stats = model_data['summary_stats']  
            
            row = {  
                'Model': model_name,  
                'RMSE_Fe_Mean': stats.get('rmse_fe_mean', np.nan),  
                'RMSE_Fe_Std': stats.get('rmse_fe_std', np.nan),  
                'RMSE_Mass_Mean': stats.get('rmse_mass_mean', np.nan),  
                'RMSE_Mass_Std': stats.get('rmse_mass_std', np.nan),  
                'R2_Fe_Mean': stats.get('r2_fe_mean', np.nan),  
                'R2_Fe_Std': stats.get('r2_fe_std', np.nan),  
                'R2_Mass_Mean': stats.get('r2_mass_mean', np.nan),  
                'R2_Mass_Std': stats.get('r2_mass_std', np.nan),  
                'Runtime_Mean': stats.get('run_time_mean', np.nan),  
                'Runtime_Std': stats.get('run_time_std', np.nan)  
            }  
            comparison_data.append(row)  
    
    comparison_df = pd.DataFrame(comparison_data)  
    
    # Сортуємо по RMSE Fe (найкращий зверху)  
    if 'RMSE_Fe_Mean' in comparison_df.columns:  
        comparison_df = comparison_df.sort_values('RMSE_Fe_Mean')  
    
    return comparison_df  


def analyze_experiment_results(results_dir: str) -> None:  
    """Детальний аналіз результатів експерименту"""  
    
    print("📊 АНАЛІЗ РЕЗУЛЬТАТІВ ЕКСПЕРИМЕНТУ")  
    print("="*70)  
    
    # Завантажуємо порівняльну таблицю  
    comparison_file = f"{results_dir}/model_comparison.csv"  
    if os.path.exists(comparison_file):  
        comparison_df = pd.read_csv(comparison_file)  
        
        print("🏆 РЕЙТИНГ МОДЕЛЕЙ (по RMSE Fe):")  
        print(comparison_df[['Model', 'RMSE_Fe_Mean', 'RMSE_Fe_Std', 'R2_Fe_Mean']].round(4))  
        
        print("\n📈 СТАТИСТИЧНА ЗНАЧУЩІСТЬ:")  
        # Простий аналіз статистичної значущості  
        best_model = comparison_df.iloc[0]  
        print(f"🥇 Найкращий: {best_model['Model']}")  
        print(f"   RMSE Fe: {best_model['RMSE_Fe_Mean']:.4f} ± {best_model['RMSE_Fe_Std']:.4f}")  
        print(f"   R² Fe: {best_model['R2_Fe_Mean']:.4f} ± {best_model['R2_Fe_Std']:.4f}")  
        
        if len(comparison_df) > 1:  
            second_model = comparison_df.iloc[1]  
            rmse_diff = second_model['RMSE_Fe_Mean'] - best_model['RMSE_Fe_Mean']  
            combined_std = np.sqrt(best_model['RMSE_Fe_Std']**2 + second_model['RMSE_Fe_Std']**2)  
            
            print(f"\n🥈 Другий: {second_model['Model']}")  
            print(f"   Різниця RMSE: +{rmse_diff:.4f}")  
            print(f"   Статистична значущість: {'Так' if rmse_diff > 2*combined_std else 'Сумнівно'}")  
    
    # Завантажуємо детальну статистику  
    summary_file = f"{results_dir}/experiment_summary.csv"  
    if os.path.exists(summary_file):  
        summary_df = pd.read_csv(summary_file)  
        
        print(f"\n📋 ДЕТАЛІ ЕКСПЕРИМЕНТУ:")  
        print(f"   Загалом запусків: {len(summary_df)}")  
        print(f"   Успішних: {len(summary_df[~summary_df.get('error', pd.Series()).notna()])}")  
        
        # Статистика по моделях  
        if 'model' in summary_df.columns:  
            model_stats = summary_df.groupby('model').agg({  
                'rmse_fe': ['count', 'mean', 'std'],  
                'run_time': ['mean', 'std']  
            }).round(4)  
            print(f"\n📊 ДЕТАЛЬНА СТАТИСТИКА ПО МОДЕЛЯХ:")  
            print(model_stats)  


# 🚀 ЗАПУСК ЕКСПЕРИМЕНТУ  
def main_experiment():  
    """Головна функція експерименту"""  
    
    # Завантажуємо дані  
    try:  
        hist_df = pd.read_parquet('processed.parquet')  
        print(f"✅ Дані завантажено: {hist_df.shape}")  
    except FileNotFoundError:  
        try:  
            hist_df = pd.read_parquet('/content/KModel/src/processed.parquet')  
            print(f"✅ Дані завантажено з /content/: {hist_df.shape}")  
        except Exception as e:  
            print(f"❌ Помилка завантаження даних: {e}")  
            return  
    
    # Callback для прогресу (показуємо тільки ключові моменти)  
    def progress_callback(step, total, msg):  
        if step % 100 == 0 or step == total or 'завершен' in msg.lower():  
            print(f"      [{step}/{total}] {msg}")  
    
    # Запускаємо експеримент  
    print("🚀 ПОЧИНАЄМО ЕКСПЕРИМЕНТ...")  
    start_time = time.time()  
    
    all_results, comparison_table = run_model_comparison_experiment(  
        hist_df,  
        base_config='oleksandr_original',  
        n_repeats=5,  
        results_dir='model_comparison_experiment'  
    )  
    
    total_time = time.time() - start_time  
    
    print(f"\n⏱️ ЗАГАЛЬНИЙ ЧАС ЕКСПЕРИМЕНТУ: {total_time/60:.1f} хв")  
    print(f"📊 РЕЗУЛЬТАТИ:")  
    print(comparison_table.round(4))  
    
    # Визначаємо директорію результатів  
    # Визначаємо директорію результатів  
    results_dirs = [d for d in os.listdir('.') if d.startswith('model_comparison_experiment_')]  
    if results_dirs:  
        latest_dir = max(results_dirs)  # Останній за алфавітом (найновіший timestamp)  
        print(f"\n🔍 Детальний аналіз результатів...")  
        analyze_experiment_results(latest_dir)  
        
        # Створюємо візуалізацію результатів  
        create_experiment_visualizations(latest_dir)  
    
    return all_results, comparison_table  


def create_experiment_visualizations(results_dir: str) -> None:  
    """Створює графіки для аналізу експерименту"""  
    
    try:  
        import matplotlib.pyplot as plt  
        import seaborn as sns  
        
        print("\n📈 СТВОРЕННЯ ВІЗУАЛІЗАЦІЙ...")  
        
        # Завантажуємо дані  
        comparison_df = pd.read_csv(f"{results_dir}/model_comparison.csv")  
        summary_df = pd.read_csv(f"{results_dir}/experiment_summary.csv")  
        
        # Налаштування стилю  
        plt.style.use('default')  
        sns.set_palette("husl")  
        
        # Створюємо фігуру з підграфіками  
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))  
        fig.suptitle('🔬 Результати експерименту порівняння моделей MPC', fontsize=16, fontweight='bold')  
        
        # 1. Bar plot RMSE Fe  
        ax1 = axes[0, 0]  
        bars1 = ax1.bar(comparison_df['Model'], comparison_df['RMSE_Fe_Mean'],   
                       yerr=comparison_df['RMSE_Fe_Std'], capsize=5, alpha=0.8)  
        ax1.set_title('🎯 RMSE Fe (нижче = краще)', fontweight='bold')  
        ax1.set_ylabel('RMSE Fe')  
        ax1.tick_params(axis='x', rotation=45)  
        ax1.grid(True, alpha=0.3)  
        
        # Підсвічуємо найкращий результат  
        min_idx = comparison_df['RMSE_Fe_Mean'].idxmin()  
        bars1[min_idx].set_color('gold')  
        bars1[min_idx].set_edgecolor('orange')  
        bars1[min_idx].set_linewidth(2)  
        
        # 2. Bar plot RMSE Mass  
        ax2 = axes[0, 1]  
        bars2 = ax2.bar(comparison_df['Model'], comparison_df['RMSE_Mass_Mean'],  
                       yerr=comparison_df['RMSE_Mass_Std'], capsize=5, alpha=0.8)  
        ax2.set_title('⚖️ RMSE Mass (нижче = краще)', fontweight='bold')  
        ax2.set_ylabel('RMSE Mass')  
        ax2.tick_params(axis='x', rotation=45)  
        ax2.grid(True, alpha=0.3)  
        
        # 3. R² Fe  
        ax3 = axes[0, 2]  
        bars3 = ax3.bar(comparison_df['Model'], comparison_df['R2_Fe_Mean'],  
                       yerr=comparison_df['R2_Fe_Std'], capsize=5, alpha=0.8)  
        ax3.set_title('📈 R² Fe (вище = краще)', fontweight='bold')  
        ax3.set_ylabel('R² Fe')  
        ax3.tick_params(axis='x', rotation=45)  
        ax3.grid(True, alpha=0.3)  
        ax3.set_ylim(0, 1)  
        
        # Підсвічуємо найкращий R²  
        max_r2_idx = comparison_df['R2_Fe_Mean'].idxmax()  
        bars3[max_r2_idx].set_color('gold')  
        bars3[max_r2_idx].set_edgecolor('orange')  
        bars3[max_r2_idx].set_linewidth(2)  
        
        # 4. Box plot RMSE по моделях (якщо є детальні дані)  
        ax4 = axes[1, 0]  
        if 'rmse_fe' in summary_df.columns and 'model' in summary_df.columns:  
            summary_df.boxplot(column='rmse_fe', by='model', ax=ax4)  
            ax4.set_title('📦 Розподіл RMSE Fe по моделях', fontweight='bold')  
            ax4.set_xlabel('Модель')  
            ax4.set_ylabel('RMSE Fe')  
        else:  
            ax4.text(0.5, 0.5, 'Детальні дані\nнедоступні', ha='center', va='center',   
                    transform=ax4.transAxes, fontsize=12)  
            ax4.set_title('📦 Розподіл RMSE Fe')  
        
        # 5. Час виконання  
        ax5 = axes[1, 1]  
        bars5 = ax5.bar(comparison_df['Model'], comparison_df['Runtime_Mean'],  
                       yerr=comparison_df['Runtime_Std'], capsize=5, alpha=0.8, color='lightcoral')  
        ax5.set_title('⏱️ Час виконання (сек)', fontweight='bold')  
        ax5.set_ylabel('Час (сек)')  
        ax5.tick_params(axis='x', rotation=45)  
        ax5.grid(True, alpha=0.3)  
        
        # 6. Scatter RMSE vs R²  
        ax6 = axes[1, 2]  
        scatter = ax6.scatter(comparison_df['RMSE_Fe_Mean'], comparison_df['R2_Fe_Mean'],  
                             s=100, alpha=0.7, c=range(len(comparison_df)), cmap='viridis')  
        ax6.set_xlabel('RMSE Fe')  
        ax6.set_ylabel('R² Fe')  
        ax6.set_title('🎯 RMSE vs R² (ліво-верх = краще)', fontweight='bold')  
        ax6.grid(True, alpha=0.3)  
        
        # Додаємо підписи точок  
        for i, model in enumerate(comparison_df['Model']):  
            ax6.annotate(model,   
                        (comparison_df.iloc[i]['RMSE_Fe_Mean'], comparison_df.iloc[i]['R2_Fe_Mean']),  
                        xytext=(5, 5), textcoords='offset points', fontsize=8)  
        
        plt.tight_layout()  
        plt.savefig(f"{results_dir}/experiment_analysis.png", dpi=300, bbox_inches='tight')  
        plt.savefig(f"{results_dir}/experiment_analysis.pdf", bbox_inches='tight')  
        
        print(f"   ✅ Графіки збережено: {results_dir}/experiment_analysis.png")  
        
        # Додатковий графік: детальний аналіз кожної моделі  
        if 'rmse_fe' in summary_df.columns:  
            create_detailed_model_analysis(summary_df, results_dir)  
        
        plt.show()  
        
    except ImportError:  
        print("   ⚠️ matplotlib/seaborn недоступні для візуалізації")  
    except Exception as e:  
        print(f"   ❌ Помилка створення графіків: {e}")  


def create_detailed_model_analysis(summary_df: pd.DataFrame, results_dir: str) -> None:  
    """Детальний аналіз результатів по моделях"""  
    
    try:  
        import matplotlib.pyplot as plt  
        import seaborn as sns  
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))  
        fig.suptitle('🔍 Детальний аналіз стабільності моделей', fontsize=14, fontweight='bold')  
        
        # 1. Violin plot RMSE Fe  
        ax1 = axes[0, 0]  
        if len(summary_df['model'].unique()) > 1:  
            sns.violinplot(data=summary_df, x='model', y='rmse_fe', ax=ax1)  
            ax1.set_title('🎻 Розподіл RMSE Fe')  
            ax1.tick_params(axis='x', rotation=45)  
        
        # 2. Violin plot R² Fe  
        ax2 = axes[0, 1]  
        if 'r2_fe' in summary_df.columns:  
            sns.violinplot(data=summary_df, x='model', y='r2_fe', ax=ax2)  
            ax2.set_title('🎻 Розподіл R² Fe')  
            ax2.tick_params(axis='x', rotation=45)  
        
        # 3. Кореляція метрик  
        ax3 = axes[1, 0]  
        if 'r2_fe' in summary_df.columns:  
            ax3.scatter(summary_df['rmse_fe'], summary_df['r2_fe'], alpha=0.6)  
            ax3.set_xlabel('RMSE Fe')  
            ax3.set_ylabel('R² Fe')  
            ax3.set_title('📊 Кореляція RMSE vs R²')  
            ax3.grid(True, alpha=0.3)  
        
        # 4. Стабільність по запускам  
        ax4 = axes[1, 1]  
        if 'run_number' in summary_df.columns:  
            for model in summary_df['model'].unique():  
                model_data = summary_df[summary_df['model'] == model]  
                ax4.plot(model_data['run_number'], model_data['rmse_fe'],   
                        'o-', label=model, alpha=0.7)  
            ax4.set_xlabel('Номер запуску')  
            ax4.set_ylabel('RMSE Fe')  
            ax4.set_title('📈 Стабільність по запускам')  
            ax4.legend()  
            ax4.grid(True, alpha=0.3)  
        
        plt.tight_layout()  
        plt.savefig(f"{results_dir}/detailed_analysis.png", dpi=300, bbox_inches='tight')  
        print(f"   ✅ Детальний аналіз збережено: {results_dir}/detailed_analysis.png")  
        
    except Exception as e:  
        print(f"   ❌ Помилка детального аналізу: {e}")  


def statistical_significance_test(results_dir: str) -> None:  
    """Статистичний тест значущості різниць між моделями"""  
    
    try:  
        from scipy import stats  
        
        print("\n🧮 СТАТИСТИЧНИЙ АНАЛІЗ ЗНАЧУЩОСТІ")  
        print("="*50)  
        
        summary_df = pd.read_csv(f"{results_dir}/experiment_summary.csv")  
        
        if 'rmse_fe' not in summary_df.columns or 'model' not in summary_df.columns:  
            print("❌ Недостатньо даних для статистичного тесту")  
            return  
        
        models = summary_df['model'].unique()  
        if len(models) < 2:  
            print("❌ Потрібно принаймні 2 моделі для порівняння")  
            return  
        
        # Парні t-тести між моделями  
        print("🔬 ПАРНІ T-ТЕСТИ (RMSE Fe):")  
        results_matrix = []  
        
        for i, model1 in enumerate(models):  
            row = []  
            for j, model2 in enumerate(models):  
                if i == j:  
                    row.append("-")  
                else:  
                    data1 = summary_df[summary_df['model'] == model1]['rmse_fe']  
                    data2 = summary_df[summary_df['model'] == model2]['rmse_fe']  
                    
                    if len(data1) >= 2 and len(data2) >= 2:  
                        t_stat, p_value = stats.ttest_ind(data1, data2)  
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"  
                        row.append(f"{p_value:.4f} {significance}")  
                    else:  
                        row.append("N/A")  
            results_matrix.append(row)  
        
        # Створюємо таблицю результатів  
        results_df = pd.DataFrame(results_matrix, index=models, columns=models)  
        print(results_df)  
        
        print("\n📊 Позначення: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")  
        
        # Зберігаємо статистичний аналіз  
        results_df.to_csv(f"{results_dir}/statistical_analysis.csv")  
        
    except ImportError:  
        print("⚠️ scipy недоступна для статистичного аналізу")  
    except Exception as e:  
        print(f"❌ Помилка статистичного аналізу: {e}")  


# # 🎯 ГОЛОВНА ФУНКЦІЯ ЗАПУСКУ  
# if __name__ == '__main__':  
#     print("🚀 ЗАПУСК ЕКСПЕРИМЕНТУ ПОРІВНЯННЯ МОДЕЛЕЙ MPC")  
#     print("="*70)  
    
#     # Запускаємо експеримент  
#     try:  
#         all_results, comparison_table = main_experiment()  
        
#         # Знаходимо директорію з результатами  
#         results_dirs = [d for d in os.listdir('.') if d.startswith('model_comparison_experiment_')]  
#         if results_dirs:  
#             latest_dir = max(results_dirs)  
            
#             print(f"\n📊 ФІНАЛЬНІ РЕЗУЛЬТАТИ:")  
#             print("="*70)  
            
#             # Показуємо порівняльну таблицю  
#             print("🏆 РЕЙТИНГ МОДЕЛЕЙ:")  
#             print(comparison_table[['Model', 'RMSE_Fe_Mean', 'RMSE_Fe_Std', 'R2_Fe_Mean', 'Runtime_Mean']].round(4))  
            
#             # Статистичний аналіз  
#             statistical_significance_test(latest_dir)  
            
#             # Рекомендації  
#             print(f"\n💡 РЕКОМЕНДАЦІЇ:")  
#             best_model = comparison_table.iloc[0]['Model']  
#             best_rmse = comparison_table.iloc[0]['RMSE_Fe_Mean']  
#             best_r2 = comparison_table.iloc[0]['R2_Fe_Mean']  
            
#             print(f"🥇 Найкращий результат: {best_model}")  
#             print(f"   RMSE Fe: {best_rmse:.4f}")  
#             print(f"   R² Fe: {best_r2:.4f}")  
            
#             if len(comparison_table) > 1:  
#                 second_best = comparison_table.iloc[1]  
#                 improvement = ((second_best['RMSE_Fe_Mean'] - best_rmse) / second_best['RMSE_Fe_Mean'] * 100)  
#                 print(f"   Покращення відносно другого місця: {improvement:.2f}%")  
            
#             print(f"\n📁 Всі результати збережено в: {latest_dir}")  
#             print(f"📊 Файли:")  
#             print(f"   - model_comparison.csv (порівняльна таблиця)")  
#             print(f"   - experiment_summary.csv (детальні результати)")  
#             print(f"   - experiment_analysis.png (графіки)")  
#             print(f"   - statistical_analysis.csv (статистичний аналіз)")  
#             print(f"   - individual *_results.parquet (результати кожного запуску)")  
            
#     except Exception as e:  
#         print(f"❌ КРИТИЧНА ПОМИЛКА: {e}")
#         import traceback
#         traceback.print_exc()
        
#     print("\n🎉 ЕКСПЕРИМЕНТ ЗАВЕРШЕНО!")
#     print("="*70)


# 🔧 ДОДАТКОВІ УТИЛІТИ ДЛЯ АНАЛІЗУ РЕЗУЛЬТАТІВ

def load_and_compare_experiments(experiment_dirs: List[str]) -> pd.DataFrame:
    """Порівнює результати кількох експериментів"""
    
    print("🔍 ПОРІВНЯННЯ ЕКСПЕРИМЕНТІВ")
    print("="*50)
    
    all_comparisons = []
    
    for exp_dir in experiment_dirs:
        try:
            comparison_file = f"{exp_dir}/model_comparison.csv"
            if os.path.exists(comparison_file):
                df = pd.read_csv(comparison_file)
                df['Experiment'] = exp_dir
                all_comparisons.append(df)
                print(f"✅ Завантажено: {exp_dir}")
            else:
                print(f"❌ Не знайдено: {comparison_file}")
        except Exception as e:
            print(f"❌ Помилка завантаження {exp_dir}: {e}")
    
    if all_comparisons:
        combined_df = pd.concat(all_comparisons, ignore_index=True)
        
        # Створюємо зведену таблицю
        pivot_table = combined_df.pivot_table(
            index='Model',
            columns='Experiment', 
            values='RMSE_Fe_Mean',
            aggfunc='mean'
        )
        
        print(f"\n📊 ЗВЕДЕНА ТАБЛИЦЯ RMSE Fe:")
        print(pivot_table.round(4))
        
        return combined_df
    else:
        print("❌ Немає даних для порівняння")
        return pd.DataFrame()


def generate_experiment_report(results_dir: str, output_file: str = None) -> str:
    """Генерує детальний звіт про експеримент"""
    
    if output_file is None:
        output_file = f"{results_dir}/experiment_report.md"
    
    try:
        comparison_df = pd.read_csv(f"{results_dir}/model_comparison.csv")
        summary_df = pd.read_csv(f"{results_dir}/experiment_summary.csv")
        
        report = f"""# 🔬 Звіт про експеримент порівняння моделей MPC

## 📊 Загальна інформація
- **Дата експерименту:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Кількість моделей:** {len(comparison_df)}
- **Повторень на модель:** {summary_df['model'].value_counts().iloc[0] if len(summary_df) > 0 else 'N/A'}
- **Загалом запусків:** {len(summary_df)}

## 🏆 Результати

### Рейтинг моделей (по RMSE Fe):
"""
        
        for idx, row in comparison_df.iterrows():
            rank = idx + 1
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            
            report += f"""
{medal} **{row['Model']}**
- RMSE Fe: {row['RMSE_Fe_Mean']:.4f} ± {row['RMSE_Fe_Std']:.4f}
- RMSE Mass: {row['RMSE_Mass_Mean']:.4f} ± {row['RMSE_Mass_Std']:.4f}
- R² Fe: {row['R2_Fe_Mean']:.4f} ± {row['R2_Fe_Std']:.4f}
- R² Mass: {row['R2_Mass_Mean']:.4f} ± {row['R2_Mass_Std']:.4f}
- Час виконання: {row['Runtime_Mean']:.1f}с ± {row['Runtime_Std']:.1f}с
"""
        
        # Аналіз стабільності
        report += f"""
## 📈 Аналіз стабільності

### Коефіцієнти варіації (CV = std/mean):
"""
        
        for idx, row in comparison_df.iterrows():
            cv_rmse = (row['RMSE_Fe_Std'] / row['RMSE_Fe_Mean']) * 100
            cv_r2 = (row['R2_Fe_Std'] / row['R2_Fe_Mean']) * 100 if row['R2_Fe_Mean'] > 0 else float('inf')
            
            stability = "Дуже стабільна" if cv_rmse < 5 else "Стабільна" if cv_rmse < 10 else "Нестабільна"
            
            report += f"- **{row['Model']}**: CV_RMSE = {cv_rmse:.2f}% ({stability})\n"
        
        # Рекомендації
        best_model = comparison_df.iloc[0]
        report += f"""
## 💡 Рекомендації

### 🎯 Найкращий результат: {best_model['Model']}
- Найнижчий RMSE Fe: {best_model['RMSE_Fe_Mean']:.4f}
- Високий R²: {best_model['R2_Fe_Mean']:.4f}
- {'Швидкий' if best_model['Runtime_Mean'] < 60 else 'Повільний'} час виконання: {best_model['Runtime_Mean']:.1f}с

### 📊 Порівняння з іншими моделями:
"""
        
        if len(comparison_df) > 1:
            for idx in range(1, min(4, len(comparison_df))):  # Топ-3 після найкращої
                model = comparison_df.iloc[idx]
                improvement = ((model['RMSE_Fe_Mean'] - best_model['RMSE_Fe_Mean']) / model['RMSE_Fe_Mean'] * 100)
                report += f"- Краще за {model['Model']} на {improvement:.2f}%\n"
        
        # Технічні деталі
        report += f"""
## 🔧 Технічні деталі

### Конфігурація експерименту:
- Базова конфігурація: `oleksandr_original`
- Seeds використані: 0, 42, 84, 126, 168 (для відтворюваності)
- Аналіз відключений для швидкості

### Файли результатів:
- `model_comparison.csv` - порівняльна таблиця
- `experiment_summary.csv` - детальні результати всіх запусків
- `experiment_analysis.png` - візуалізація результатів
- `statistical_analysis.csv` - статистичний аналіз значущості
- `*_results.parquet` - результати окремих симуляцій
- `*_metrics.json` - метрики окремих симуляцій

## 📝 Висновки

1. **Найкращий вибір:** {best_model['Model']} показує найкращі результати по точності
2. **Стабільність:** Всі моделі показують {'стабільні' if comparison_df['RMSE_Fe_Std'].max() < 0.1 else 'варіативні'} результати
3. **Швидкодія:** {'GPR та SVR' if any('GPR' in m or 'SVR' in m for m in comparison_df['Model']) else 'Лінійні моделі'} потребують більше часу для навчання
4. **Рекомендація:** Використовувати {best_model['Model']} для продакшн системи

---
*Звіт згенерований автоматично {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        # Зберігаємо звіт
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Звіт збережено: {output_file}")
        return report
        
    except Exception as e:
        print(f"❌ Помилка генерації звіту: {e}")
        return ""

# 🎯 ШВИДКИЙ ЗАПУСК (ДЛЯ ТЕСТУВАННЯ)
def quick_test_experiment():
    """Швидкий тест експерименту з меншою кількістю даних"""
    
    print("⚡ ШВИДКИЙ ТЕСТ ЕКСПЕРИМЕНТУ")
    
    try:
        hist_df = pd.read_parquet('processed.parquet')
    except:
        try:
            hist_df = pd.read_parquet('/content/KModel/src/processed.parquet')
        except Exception as e:
            print(f"❌ Не можу завантажити дані: {e}")
            return
    
    # Тестуємо тільки 2 моделі по 2 рази з меншими даними
    models_config = {
        'KRR_RBF_TEST': {
            'model_type': 'krr',
            'kernel': 'rbf',
            'N_data': 2000,
            'control_pts': 200,
            'find_optimal_params': True
        },
        'LINEAR_TEST': {
            'model_type': 'linear',
            'linear_type': 'ridge',
            'N_data': 2000,
            'control_pts': 200,
            'find_optimal_params': True
        }
    }
    
    results = {}
    for model_name, params in models_config.items():
        print(f"\n🧪 Тестуємо {model_name}...")
        
        try:
            results_df, metrics = simulate_mpc(
                hist_df,
                config='oleksandr_original',
                config_overrides=params,
                run_analysis=False
            )
            
            results[model_name] = {
                'rmse_fe': metrics.get('rmse_fe', 0),
                'rmse_mass': metrics.get('rmse_mass', 0),
                'r2_fe': metrics.get('r2_fe', 0)
            }
            
            print(f"   ✅ RMSE Fe: {metrics.get('rmse_fe', 0):.4f}")
            
        except Exception as e:
            print(f"   ❌ Помилка: {e}")
    
    print(f"\n📊 РЕЗУЛЬТАТИ ШВИДКОГО ТЕСТУ:")
    for model, metrics in results.items():
        print(f"   {model}: RMSE_Fe={metrics['rmse_fe']:.4f}, R²={metrics['r2_fe']:.4f}")
    
    return results

# ---- АНАЛІЗ ТА ТЕСТУВАННЯ РЕЗУЛЬТАТІВ

# 🔍 ШВИДКА ПЕРЕВІРКА ПОВЕРНЕННЯ МЕТРИК
def quick_metrics_check():
    """Швидка перевірка чи метрики взагалі повертаються з simulate_mpc"""
    
    try:
        hist_df = pd.read_parquet('processed.parquet')
    except:
        hist_df = pd.read_parquet('/content/KModel/src/processed.parquet')
    
    print("🔍 ШВИДКА ПЕРЕВІРКА МЕТРИК")
    print("="*40)
    
    # Мінімальна конфігурація
    config = {
        'model_type': 'krr',
        'kernel': 'rbf',
        'N_data': 500,
        'control_pts': 20,
        'find_optimal_params': False
    }
    
    # Викликаємо simulate_mpc
    results_df, metrics = simulate_mpc(
        hist_df,
        config='oleksandr_original',
        config_overrides=config,
        run_analysis=False
    )
    
    print(f"📊 РЕЗУЛЬТАТ ВИКЛИКУ:")
    print(f"   results_df: {type(results_df)} shape={getattr(results_df, 'shape', 'N/A')}")
    print(f"   metrics: {type(metrics)}")
    
    if isinstance(metrics, dict):
        print(f"   metrics має {len(metrics)} ключів")
        
        # Показуємо всі ключі і значення
        print(f"\n📋 ВСІ МЕТРИКИ:")
        for key, value in metrics.items():
            print(f"   {key}: {value} ({type(value)})")
        
        # Перевіряємо чи є RMSE
        rmse_keys = [k for k in metrics.keys() if 'rmse' in k.lower()]
        if rmse_keys:
            print(f"\n🎯 ЗНАЙДЕНІ RMSE КЛЮЧІ:")
            for key in rmse_keys:
                print(f"   {key} = {metrics[key]}")
                return metrics[key]  # Повертаємо перше RMSE значення
        else:
            print(f"\n❌ RMSE КЛЮЧІ НЕ ЗНАЙДЕНІ!")
            print(f"Доступні ключі: {list(metrics.keys())}")
    
    else:
        print(f"   ❌ metrics не є словником!")
    
    return metrics

# СПОЧАТКУ ЗАПУСКАЙ ЦЕ
if __name__ == '__main__':
    # quick_test_experiment()
    quick_metrics_check()