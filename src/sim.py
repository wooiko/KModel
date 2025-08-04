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
    analize_errors, plot_control_and_disturbances, 
    evaluate_ekf_performance, plot_fact_vs_mpc_plans,
    plot_disturbance_estimation, control_aggressiveness_metrics,
    plot_delta_u_histogram, plot_trust_region_evolution, plot_linearization_quality,
    run_post_simulation_analysis_enhanced
)
from ekf import ExtendedKalmanFilter
from anomaly_detector import SignalAnomalyDetector
from maf import MovingAverageFilter

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

    # ✅ ДОДАЙ ДІАГНОСТИКУ SVR ТУТ:
    if hasattr(mpc.model, '_impl') and hasattr(mpc.model._impl, 'models'):  # Перевіряємо чи це SVR
        print("\n🔍 ЗАПУСК ДІАГНОСТИКИ SVR...")
        svr_diagnostics = diagnose_svr_quality(
            svr_model=mpc.model._impl,  # ВАЖЛИВО: ._impl для доступу до SVR
            X_train=data['X_train_scaled'], 
            y_train=data['Y_train_scaled'],
            X_test=data['X_test_scaled'],
            y_test=data['Y_test_scaled'],
            x_scaler=None,  # Дані вже масштабовані
            y_scaler=y_scaler
        )

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
    
    x0_aug = np.hstack([hist0_unscaled.flatten(), np.zeros(n_dist)])
    
    P0 = np.eye(n_phys + n_dist) * params['P0']
    P0[n_phys:, n_phys:] *= 1 

    Q_phys = np.eye(n_phys) * params['Q_phys']
    Q_dist = np.eye(n_dist) * params['Q_dist'] 
    Q = np.block([[Q_phys, np.zeros((n_phys, n_dist))], [np.zeros((n_dist, n_phys)), Q_dist]])
    
    R = np.diag(np.var(Y_train_scaled, axis=0)) * params['R']
    
    return ExtendedKalmanFilter(
        mpc.model, x_scaler, y_scaler, x0_aug, P0, Q, R, lag,
        beta_R=params.get('beta_R', 0.1), # .get для зворотної сумісності
        q_adaptive_enabled=params.get('q_adaptive_enabled', True),
        q_alpha=params.get('q_alpha', 0.995),
        q_nis_threshold=params.get('q_nis_threshold', 1.8)        
    )

# =============================================================================
# === БЛОК 3: ОСНОВНИЙ ЦИКЛ СИМУЛЯЦІЇ ===
# =============================================================================


    
# =============================================================================
# === ГОЛОВНА ФУНКЦІЯ-ОРКЕСТРАТОР ===
# =============================================================================


def simulate_mpc(
    reference_df: pd.DataFrame,             # DataFrame, що містить референсні дані для генерації даних симуляції.
    N_data: int = 5000,                     # Загальна кількість точок даних, що генеруються для симуляції.
    control_pts : int = 1000,               # Кількість точок (кроків) симуляції, на яких відбувається керування MPC.
    time_step_s : int = 5,                  # Часовий крок виконання
    dead_times_s : dict = 
    {
        'concentrate_fe_percent': 20.0,
        'tailings_fe_percent': 25.0,
        'concentrate_mass_flow': 20.0,
        'tailings_mass_flow': 25.0
    },                                      # Транспортна затримка вихідних параметрів
    time_constants_s : dict = 
    {
        'concentrate_fe_percent': 8.0,
        'tailings_fe_percent': 10.0,
        'concentrate_mass_flow': 5.0,
        'tailings_mass_flow': 7.0
    },                                      # Інерційність вихідних параметрів
    lag: int = 2,                           # Кількість кроків затримки (lag) для моделі, впливає на розмір вектора стану.
    Np: int = 6,                            # Горизонт прогнозування (Prediction Horizon) MPC. Кількість майбутніх кроків, які модель прогнозує.
    Nc: int = 4,                            # Горизонт керування (Control Horizon) MPC. Кількість майбутніх змін керування, які MPC розраховує.
    n_neighbors: int = 5,                   # Кількість сусідів для KNN регресора, якщо використовується (наразі не використовується в `KernelModel`).
    seed: int = 0,                          # Зерно для генератора випадкових чисел, для відтворюваності симуляції.
    noise_level: str = 'none',              # Рівень шуму, який додається до вимірювань 'none', 'low', 'medium', 'high'. Визначає відсоток похибки.
    model_type: str = 'krr',                # Тип моделі, що використовується в MPC: 'krr' (Kernel Ridge Regression), 'gpr' (Gaussian Process Regressor), 'svr' (Support-Vector Regression).
    kernel: str = 'rbf',                    # Тип ядра для KernelModel ('linear', 'poly', 'rbf').
    find_optimal_params: bool = True,       # Чи потрібно шукати оптимальні гіперпараметри моделі за допомогою RandomizedSearchCV.
    λ_obj: float = 0.1,                     # Коефіцієнт ваги для терму згладжування керування (lambda) в цільовій функції MPC.
    K_I: float = 0.01,                      # Інтегральний коефіцієнт для інтегрального контролера (якщо використовується). Наразі не застосовується явно в MPC.
    w_fe: float = 7.0,                      # Вага для помилки прогнозування концентрації заліза (Fe) в цільовій функції MPC.
    w_mass: float = 1.0,                    # Вага для помилки прогнозування масової витрати концентрату в цільовій функції MPC.
    ref_fe: float = 53.5,                   # Бажане (референсне) значення концентрації заліза (Fe) в концентраті.
    ref_mass: float = 57.0,                 # Бажане (референсне) значення масової витрати концентрату.
    train_size: float = 0.7,                # Частка даних, що використовуються для навчання моделі MPC.
    val_size: float = 0.15,                 # Частка даних, що використовуються для валідації моделі (якщо `find_optimal_params=True`).
    test_size: float = 0.15,                # Частка даних, що використовуються для тестування моделі (зазвичай не використовується безпосередньо в циклі MPC).
    u_min: float = 20.0,                    # Мінімальне допустиме значення для керуючої змінної `u` (ore_flow_rate_target).
    u_max: float = 40.0,                    # Максимальне допустиме значення для керуючої змінної `u` (ore_flow_rate_target).
    delta_u_max: float = 1.0,               # Максимальне допустиме абсолютне значення зміни керуючої змінної `u` між послідовними кроками.
    use_disturbance_estimator: bool = True, # Чи використовувати оцінювач збурень (Extended Kalman Filter) в циклі MPC.
    y_max_fe: float = 54.5,                 # Верхня межа для концентрації заліза (Fe) в концентраті (жорстке або м'яке обмеження).
    y_max_mass: float = 58.0,               # Верхня межа для масової витрати концентрату (жорстке або м'яке обмеження).
    rho_trust: float = 0.1,                 # Коефіцієнт штрафу (rho) для терму довіри в цільовій функції MPC, що використовується для регуляризації.
    max_trust_radius: float = 5.0,
    adaptive_trust_region: bool = True,
    initial_trust_radius: float =  1.0,
    min_trust_radius: float =  0.5,
    trust_decay_factor: float =  0.8,
    linearization_check_enabled: bool = True,
    max_linearization_distance: float =  2.0,
    retrain_linearization_threshold: float =  1.5,
    use_soft_constraints: bool = True,      # Чи використовувати м'які обмеження для виходів (y) та зміни керування (delta_u).
    plant_model_type: str = 'rf',           # Тип моделі, що імітує "реальний об'єкт" (plant) для генерації даних: 'rf' (Random Forest) або 'nn' (Neural Network).
    enable_retraining: bool = True,         # Ввімкнути/вимкнути функціонал перенавчання моделі MPC під час симуляції.
    retrain_period: int = 50,               # Як часто перевіряти необхідність перенавчання (кожні N кроків).
    retrain_window_size: int = 1000,        # Розмір буфера даних для перенавчання (використовуються останні `retrain_window_size` точок).
    retrain_innov_threshold: float = 0.3,   # Поріг для середньої нормованої інновації EKF. Якщо NIS перевищує цей поріг, ініціюється перенавчання.
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
    },                                      # Нелінійна конфігунація
    enable_nonlinear: bool =  False,        # Використовувати нелінійну конфігурацію
    run_analysis: bool = True,              # Показати візуалізацію результатів роботи симулятора
    P0: float = 1e-2,
    Q_phys: float = 1500,
    Q_dist: float = 1,
    R: float = 0.01,
    q_adaptive_enabled: bool = True,
    q_alpha:float = 0.99,
    q_nis_threshold:float = 1.5,
    progress_callback: Callable[[int, int, str], None] = None # Функція зворотного виклику для відстеження прогресу симуляції. Приймає поточний крок, загальну кількість кроків та повідомлення.
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
    
    return results_df, metrics

def diagnose_svr_quality(svr_model, X_train, y_train, X_test, y_test, x_scaler, y_scaler):
    """Діагностика якості SVR моделі"""
    
    print("\n" + "="*50)
    print("=== ДІАГНОСТИКА SVR МОДЕЛІ ===")
    print("="*50)
    
    # 1. Перевір support vectors
    print("\n--- Support Vectors Analysis ---")
    for i, svr in enumerate(svr_model.models):
        sv_count = len(svr.support_vectors_)
        sv_ratio = sv_count / len(X_train)
        print(f"Вихід {i}: Support Vectors = {sv_count}/{len(X_train)} ({sv_ratio:.1%})")
        print(f"  C = {svr.C:.3f}, gamma = {getattr(svr, 'gamma', 'N/A')}, epsilon = {svr.epsilon:.3f}")
        
        if sv_ratio > 0.8:
            print(f"  ❌ ПЕРЕНАВЧАННЯ! {sv_ratio:.1%} > 80% зразків є support vectors")
        elif sv_ratio < 0.1:
            print(f"  ❌ НЕДОНАВЧАННЯ! {sv_ratio:.1%} < 10% support vectors")
        else:
            print(f"  ✅ Нормальна кількість support vectors")
    
    # 2. Прямий тест SVR (без EKF)
    print("\n--- Direct SVR Performance ---")
    y_pred_train = svr_model.predict(X_train)
    y_pred_test = svr_model.predict(X_test)
    
    # Train error
    rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2, axis=0))
    nrmse_train = rmse_train / (np.max(y_train, axis=0) - np.min(y_train, axis=0))
    
    # Test error  
    rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2, axis=0))
    nrmse_test = rmse_test / (np.max(y_test, axis=0) - np.min(y_test, axis=0))
    
    print(f"SVR Train RMSE: {rmse_train}")
    print(f"SVR Train NRMSE: {nrmse_train}")
    print(f"SVR Test RMSE: {rmse_test}")
    print(f"SVR Test NRMSE: {nrmse_test}")
    
    # Overfitting check
    overfitting = nrmse_test / nrmse_train
    print(f"Overfitting ratio (test/train NRMSE): {overfitting}")
    
    for i in range(len(nrmse_test)):
        if nrmse_test[i] > 5.0:
            print(f"  ❌ Вихід {i}: SVR МОДЕЛЬ ПОГАНА! NRMSE = {nrmse_test[i]:.2f} > 5.0")
        elif nrmse_test[i] > 2.0:
            print(f"  ⚠️  Вихід {i}: SVR якість низька. NRMSE = {nrmse_test[i]:.2f}")
        else:
            print(f"  ✅ Вихід {i}: SVR якість прийнятна. NRMSE = {nrmse_test[i]:.2f}")
            
        if overfitting[i] > 2.0:
            print(f"  ❌ Вихід {i}: Сильне перенавчання! Test/Train = {overfitting[i]:.2f}")
    
    # 3. Перевір діапазони даних
    print("\n--- Data Range Analysis ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_train range: min={np.min(X_train, axis=0)}, max={np.max(X_train, axis=0)}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train range: min={np.min(y_train, axis=0)}, max={np.max(y_train, axis=0)}")
    
    # 4. ✅ ВИПРАВЛЕНИЙ Linearization check
    print("\n--- Linearization Analysis ---")
    try:
        # ✅ ВИКОРИСТОВУЄМО ТІЛЬКИ ОДИН ЗРАЗОК:
        X_single = X_test[:1]  # (1, 9) - ПРАВИЛЬНА ФОРМА!
        W, b = svr_model.linearize(X_single)
        print(f"Jacobian W shape: {W.shape}")
        print(f"Bias b shape: {b.shape}")
        print(f"W range: min={np.min(W):.3f}, max={np.max(W):.3f}")
        print(f"b range: min={np.min(b):.3f}, max={np.max(b):.3f}")
        
        # Check for extreme values
        if np.any(np.abs(W) > 100) or np.any(np.abs(b) > 100):
            print("  ❌ ПОПЕРЕДЖЕННЯ: Екстремальні значення в лінеаризації!")
        else:
            print("  ✅ Лінеаризація в нормальних межах")
            
        # ✅ ПЕРЕВІРКА НА НУЛЬОВИЙ ЯКОБІАН:
        W_norm = np.linalg.norm(W)
        print(f"Jacobian norm: {W_norm:.4f}")
        if W_norm < 0.01:
            print("  ❌ КРИТИЧНО: Якобіан майже нульовий! MPC не зможе керувати!")
        else:
            print("  ✅ Якобіан має достатню норму для керування")
            
    except Exception as e:
        print(f"  ❌ ПОМИЛКА лінеаризації: {e}")
    
    print("="*50)
    
    return {
        'train_nrmse': nrmse_train,
        'test_nrmse': nrmse_test,
        'overfitting_ratio': overfitting,
        'support_vector_ratios': [len(m.support_vectors_)/len(X_train) for m in svr_model.models]
    }

def diagnose_mpc_behavior(mpc, step, u_optimal, u_prev, d_seq):
    """Діагностика поведінки MPC"""
    
    print(f"\n--- MPC ДІАГНОСТИКА (крок {step}) ---")
    print(f"u_prev: {u_prev:.4f}")
    
    if u_optimal is not None and len(u_optimal) > 0:
        print(f"u_optimal: {u_optimal[0]:.4f}")
        print(f"delta_u: {u_optimal[0] - u_prev:.4f}")
    else:
        print("u_optimal: None")
        
    print(f"trust_radius: {mpc.trust_region_radius:.4f}")
    print(f"problem_status: {mpc.problem.status}")
    print(f"problem_value: {mpc.problem.value}")
    
    # Перевіряємо лінеаризацію
    if 'W' in mpc.parameters and mpc.parameters['W'].value is not None:
        W = mpc.parameters['W'].value
        print(f"Якобіан W: min={np.min(W):.4f}, max={np.max(W):.4f}")
        print(f"Якобіан W norm: {np.linalg.norm(W):.4f}")
        
        if np.all(np.abs(W) < 0.01):
            print("❌ ПРОБЛЕМА: Якобіан майже нульовий! MPC не бачить впливу u на y")
        else:
            print("✅ Якобіан має нормальні значення")
    
    # Перевіряємо збурення
    if d_seq is not None:
        print(f"d_seq mean: {np.mean(d_seq, axis=0)}")
        print(f"d_seq std: {np.std(d_seq, axis=0)}")

def diagnose_ekf_detailed(ekf, y_true_seq, y_pred_seq, x_est_seq, innovation_seq):
    """Детальна діагностика EKF"""
    
    print("\n" + "="*60)
    print("=== ДЕТАЛЬНА ДІАГНОСТИКА EKF ===")
    print("="*60)
    
    print("\n--- ДІАГНОСТИКА ДАНИХ ---")
    y_pred = np.array(y_pred_seq[1:])
    y_true = np.array(y_true_seq[1:])
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_true shape: {y_true.shape}")
    print(f"y_pred sample: {y_pred[0]}")
    print(f"y_true sample: {y_true[0]}")
    model_error = y_true - y_pred
    print(f"model_error sample: {model_error[0]}")
    print(f"model_error mean: {np.mean(model_error, axis=0)}")
    print(f"model_error std: {np.std(model_error, axis=0)}")

    # 1. Аналіз інновацій
    innovations = np.array(innovation_seq[1:])  # Skip first step
    print("\n--- Аналіз інновацій ---")
    print(f"Innovation mean: {np.mean(innovations, axis=0)}")
    print(f"Innovation std: {np.std(innovations, axis=0)}")
    print(f"Innovation max abs: {np.max(np.abs(innovations), axis=0)}")
    
    # Перевірка на систематичні помилки
    if np.any(np.abs(np.mean(innovations, axis=0)) > 0.5):
        print("❌ СИСТЕМАТИЧНА ПОМИЛКА: Інновації мають велике зміщення!")
    else:
        print("✅ Інновації центровані")
    
    # 2. Аналіз передбачень моделі vs реальності  
    print("\n--- Якість передбачень моделі ---")
    model_rmse = np.sqrt(np.mean(model_error**2, axis=0))
    print(f"Model prediction error RMSE: {model_rmse}")
    
    # ✅ ПРАВИЛЬНА НОРМАЛІЗАЦІЯ:
    y_mean = np.mean(y_true, axis=0)
    model_nrmse = model_rmse / y_mean * 100  # В відсотках
    print(f"Model prediction NRMSE: FE={model_nrmse[0]:.2f}%, Mass={model_nrmse[1]:.2f}%")
    
    if np.any(model_nrmse > 10):  # 10%
        print("❌ МОДЕЛЬ ПОГАНА: NRMSE передбачень > 10%")
    else:
        print("✅ Модель передбачає добре")
    
    # 3. Аналіз коваріанс EKF
    print("\n--- Аналіз коваріанс EKF ---")
    P_diag_history = []
    if hasattr(ekf, 'P_history') and ekf.P_history:
        for P in ekf.P_history[-10:]:  # Last 10 steps
            P_diag_history.append(np.diag(P)[:2])  # Only output states
        
        P_diag_mean = np.mean(P_diag_history, axis=0)
        print(f"EKF covariance diagonal (outputs): {P_diag_mean}")
        
        # Перевірка чи не занадто великі коваріанси
        if np.any(P_diag_mean > 100):
            print("❌ EKF коваріанси ЗАНАДТО ВЕЛИКІ! Фільтр не впевнений")
        elif np.any(P_diag_mean < 0.01):
            print("❌ EKF коваріанси ЗАНАДТО МАЛІ! Фільтр переоцінює точність")
        else:
            print("✅ EKF коваріанси в розумних межах")
    else:
        print("⚠️ EKF covariance history недоступна")
    
    # 4. Порівняння різних джерел помилок
    print("\n--- Декомпозиція помилок ---")
    
    # Помилка через модель
    model_contribution = np.std(model_error, axis=0)
    
    # Помилка через фільтрацію (використовуємо інновації)
    innovations_array = np.array(innovation_seq[1:])
    innovation_contribution = np.std(innovations_array, axis=0)
    
    print(f"Model error contribution: {model_contribution}")
    print(f"Innovation contribution: {innovation_contribution}")
    
    # Що більше впливає?
    ratio = innovation_contribution / (model_contribution + 1e-8)
    print(f"Innovation/Model error ratio: {ratio}")
    
    if np.any(ratio > 2):
        print("❌ ФІЛЬТР ПСУЄ ОЦІНКИ! Проблема в EKF налаштуваннях")
    elif np.any(ratio < 0.5):
        print("❌ МОДЕЛЬ ДОМІНУЄ! Проблема в якості моделі")
    else:
        print("✅ Збалансовані помилки моделі та фільтра")

    # ✅ ДОДАТКОВА ДІАГНОСТИКА ДЛЯ ПЕРЕВІРКИ RMSE = 34:
    print(f"\n--- ДОДАТКОВА ДІАГНОСТИКА ---")
    print(f"y_true mean: {np.mean(y_true, axis=0)}")
    print(f"y_true std: {np.std(y_true, axis=0)}")
    print(f"Model RMSE/std ratio: {model_rmse / np.std(y_true, axis=0)}")
    print(f"Sample model errors (перші 5): {model_error[:5]}")
       
    print("="*60)

if __name__ == '__main__':
    
    def my_progress(step, total, msg):
        # Простий callback для виводу прогресу в консоль
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
        
        # ---- Блок даних
        N_data=4000, 
        control_pts=1000,
        seed=42,
        
        plant_model_type='rf',
        
        train_size=0.85,
        val_size=0.13,
        test_size=0.02,
    
        # ---- Налаштування моделі
        noise_level='low',
        model_type='svr',
        kernel='rbf', 
        find_optimal_params=True,
        use_soft_constraints=True,
        
        # ---- Налаштування EKF
        P0=1e-2,
        Q_phys=600, #1000,
        Q_dist=1,
        R=2.0, # 0.18
        q_adaptive_enabled=False,
        q_alpha = 0.999,
        q_nis_threshold = 1.8,

        # Адаптивний Trust Region
        adaptive_trust_region=True,           # Увімкнути адаптацію
        initial_trust_radius=3.0,             # Початковий розмір
        min_trust_radius=0.5,                 # Мінімальний розмір  
        max_trust_radius=2.0,                 # Максимальний розмір
        trust_decay_factor=0.9,               # Затухання по горизонту
        rho_trust=0.5,
        
        # Контроль лінеаризації
        linearization_check_enabled=True,     # Увімкнути перевірку
        max_linearization_distance=0.8,       # Поріг відстані
        retrain_linearization_threshold=1.0,  # Поріг для перенавчання

        # ---- Налантування аномалій
        anomaly_params = 
        {
            'window': 25,
            'spike_z': 4.0,
            'drop_rel': 0.30,
            'freeze_len': 5,
            'enabled': True
        },

        # ---- Нелінійні параметри
        nonlinear_config = 
        {
            'concentrate_fe_percent': ('pow', 2),
            'concentrate_mass_flow': ('pow', 1.5)
        },
        enable_nonlinear =  True, 

        # ---- Параметри затримки, чавові параметри
        time_step_s = 1800,
        dead_times_s = 
        {
            'concentrate_fe_percent': 20.0,
            'tailings_fe_percent': 25.0,
            'concentrate_mass_flow': 20.0,
            'tailings_mass_flow': 25.0
        },
                time_constants_s = 
        {
            'concentrate_fe_percent': 8.0,
            'tailings_fe_percent': 10.0,
            'concentrate_mass_flow': 5.0,
            'tailings_mass_flow': 7.0
        },
        
        # ---- Обмеження моделі
        delta_u_max = 0.6,
        λ_obj=0.2,
        
        # ---- MPC горизонти
        Nc=6, #8
        Np=8, #12
        lag=2, #2
        
        # ---- Цільові параметри/ваги
        w_fe=1.0,
        w_mass=1.0,
        ref_fe=54.5,
        ref_mass=57.0,
        y_max_fe=55.0,
        y_max_mass=60.0,
        
        # ---- Блок перенавчання
        enable_retraining=True,          # Ввімкнути/вимкнути функціонал перенавчання
        retrain_period=50,                 # Як часто перевіряти необхідність перенавчання (кожні 50 кроків)
        retrain_window_size=1000,          # Розмір буфера даних для перенавчання (останні 1000 точок)
        retrain_innov_threshold=0.25,     # Поріг для середньої нормованої інновації EKF
        
        run_analysis=True
    )
    
    print("\nФінальні метрики:")
    print(mets)
    res.to_parquet('mpc_simulation_results.parquet')



