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
    plot_delta_u_histogram
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

def initialize_mpc_controller(
    params: Dict[str, Any],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler
) -> MPCController:
    """
    Ініціалізує та налаштовує MPC контролер.

    Args:
        params: Словник з параметрами MPC.
        x_scaler: Навчений скалер для вхідних даних.
        y_scaler: Навчений скалер для вихідних даних.

    Returns:
        Налаштований екземпляр MPCController.
    """
    print("Крок 2: Ініціалізація MPC контролера...")
    # Створення моделі процесу
    kernel_model = KernelModel(
        model_type=params['model_type'],
        kernel=params['kernel'],
        find_optimal_params=params['find_optimal_params']
    )
    
    # Масштабування уставк та обмежень
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

    # Створення контролера
    mpc = MPCController(
        model=kernel_model, objective=objective, x_scaler=x_scaler, y_scaler=y_scaler,
        n_targets=2, horizon=params['Np'], control_horizon=params['Nc'], lag=params['lag'],
        u_min=params['u_min'], u_max=params['u_max'], delta_u_max=params['delta_u_max'],
        use_disturbance_estimator=params['use_disturbance_estimator'],
        y_max=list(y_max_scaled) if params['use_soft_constraints'] else None,
        rho_y=rho_y_val, rho_delta_u=rho_du_val, rho_trust=params['rho_trust']
    )
    return mpc


def train_and_evaluate_model(
    mpc: MPCController,
    data: Dict[str, np.ndarray],
    y_scaler: StandardScaler
) -> Dict[str, float]:
    """
    Навчає модель всередині MPC та оцінює її якість на тестових даних.

    Args:
        mpc: Екземпляр MPC контролера з ненавченою моделлю.
        data: Словник з розбитими та масштабованими даними.
        y_scaler: Навчений скалер для вихідних даних.
        
    Returns:
        Словник з метриками якості моделі.
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


# ---- New Block
# sim.py - Додаткові функції для інтеграції покращень

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

        # 6. Оптимізація MPC з покращеннями
        d_seq = np.repeat(d_filt.reshape(1, -1), params['Np'], axis=0)
        u_seq = mpc.optimize(d_seq, u_prev)
        u_cur = u_prev if u_seq is None else float(u_seq[0])

        # 7. Крок «реального» процесу
        y_full = true_gen.step(feed_fe_raw, ore_flow_raw, u_cur)

        # 8. EKF: корекція
        y_meas_unscaled = y_full[['concentrate_fe_percent',
                                  'concentrate_mass_flow']].values.flatten()
        ekf.update(y_meas_unscaled)

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
        
    # Розширені дані для аналізу
    analysis_data = {
        "y_true": np.vstack(y_true_hist),
        "x_hat": np.vstack(x_hat_hist),
        "P": np.stack(P_hist),
        "innov": np.vstack(innov_hist),
        "R": np.stack(R_hist),
        "u_seq": u_seq_hist,
        "d_hat": np.vstack(d_hat_hist) if d_hat_hist else np.array([]),
        "trust_region_stats": trust_region_stats_hist,  # НОВИЙ
        "linearization_quality": linearization_quality_hist,  # НОВИЙ
    }

    return pd.DataFrame(records), analysis_data

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


# Модифікована головна функція симулятора
def simulate_mpc_enhanced(
    reference_df: pd.DataFrame,
    # ... всі існуючі параметри ...
    # === НОВІ ПАРАМЕТРИ ===
    adaptive_trust_region: bool = True,
    initial_trust_radius: float = 1.0,
    min_trust_radius: float = 0.1,
    max_trust_radius: float = 5.0,
    trust_decay_factor: float = 0.8,
    linearization_check_enabled: bool = True,
    max_linearization_distance: float = 2.0,
    retrain_linearization_threshold: float = 1.5,
    **kwargs  # Для зворотної сумісності
):
    """
    Покращена версія головної функції-оркестратора.
    """
    # Збираємо всі параметри в один словник
    params = locals()
    params.update(kwargs)  # Додаємо будь-які додаткові параметри
    
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

# ---- End New Block


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

def run_simulation_loop(
    true_gen: StatefulDataGenerator,
    mpc: MPCController,
    ekf: ExtendedKalmanFilter,
    df_true: pd.DataFrame,
    # >>> НОВІ АРГУМЕНТИ <<<
    data: Dict[str, np.ndarray],
    scalers: Tuple[StandardScaler, StandardScaler],
    params: Dict[str, Any],
    progress_callback: Callable | None = None,
) -> pd.DataFrame:
    """
    Виконує основний замкнений цикл симуляції MPC з динамічним перенавчанням
    та online-фільтрацією spike / drift / drop / freeze аномалій.
    """
    print("Крок 5: Запуск основного циклу симуляції з логікою динамічного перенавчання...")
    x_scaler, y_scaler = scalers

    # ---------------------------------------------------------------------
    # 0. Початкова ініціалізація (як і раніше)
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 1. Службові змінні
    # ---------------------------------------------------------------------
    records = []
    y_true_hist, x_hat_hist, P_hist, innov_hist, R_hist = [], [], [], [], []
    u_seq_hist = [] # <<< НОВИЙ СПИСОК для планів MPC
    d_hat_hist = [] # <<< НОВИЙ СПИСОК для оцінки збурень
    u_prev = float(hist0_unscaled[-1, 2])

    y_true_hist, x_hat_hist, P_hist, innov_hist, R_hist = [], [], [], [], []

    window_size = 4
    filt_feed = MovingAverageFilter(window_size)
    filt_ore  = MovingAverageFilter(window_size)

    retrain_cooldown_timer = 0

    # ---------------------------------------------------------------------
    # 2. Буфери та налаштування для перенавчання (як і раніше)
    # ---------------------------------------------------------------------
    if params['enable_retraining']:
        print(f"-> Динамічне перенавчання УВІМКНЕНО. "
              f"Вікно: {params['retrain_window_size']}, "
              f"Період перевірки: {params['retrain_period']}")
        retraining_buffer   = deque(maxlen=params['retrain_window_size'])
        initial_train_data  = list(zip(data['X_train_scaled'],
                                       data['Y_train_scaled']))
        retraining_buffer.extend(initial_train_data)
        innovation_monitor  = deque(maxlen=params['retrain_period'])

    # ---------------------------------------------------------------------
    # 3. ONLINE-ДЕТЕКТОРИ АНОМАЛІЙ (нове)
    # ---------------------------------------------------------------------
    ad_config = params.get('anomaly_params', {})
    ad_feed_fe = SignalAnomalyDetector(**ad_config)
    ad_ore_flow = SignalAnomalyDetector(**ad_config)

    
    # ---------------------------------------------------------------------
    # 4. Головний цикл симуляції
    # ---------------------------------------------------------------------
    for t in range(T_sim):
        if progress_callback:
            progress_callback(t, T_sim, f"Крок симуляції {t + 1}/{T_sim}")

        # ------------------------------------------------- 4.1 Сирі вимірювання
        feed_fe_raw, ore_flow_raw = d_all[t, :]

        # -------------------- 4.2 ONLINE-фільтрування аномалій ----------------
        feed_fe_filt_anom = ad_feed_fe.update(feed_fe_raw)
        ore_flow_filt_anom = ad_ore_flow.update(ore_flow_raw)
        # ---------------------------------------------------------------------

        # 4.3 Грубе згладжування (як і раніше, після видалення аномалій)
        d_filt = np.array([filt_feed.update(feed_fe_filt_anom),
                           filt_ore.update(ore_flow_filt_anom)])

        # 4.4 EKF: прогноз
        ekf.predict(u_prev, d_filt)

        # 4.5 Оновлення історії в MPC
        x_est_phys_unscaled = ekf.x_hat[:ekf.n_phys].reshape(params['lag'] + 1, 3)
        mpc.reset_history(x_est_phys_unscaled)
        mpc.d_hat = ekf.x_hat[ekf.n_phys:]

        # 4.6 Оптимізація MPC
        d_seq = np.repeat(d_filt.reshape(1, -1), params['Np'], axis=0)
        u_seq = mpc.optimize(d_seq, u_prev)
        u_cur = u_prev if u_seq is None else float(u_seq[0])

        # 4.7 Крок «реального» процесу
        y_full = true_gen.step(feed_fe_raw, ore_flow_raw, u_cur)

        # 4.8 EKF: корекція
        y_meas_unscaled = y_full[['concentrate_fe_percent',
                                  'concentrate_mass_flow']].values.flatten()
        ekf.update(y_meas_unscaled)

        # 4.9 Зменшуємо cooldown-таймер
        if retrain_cooldown_timer > 0:
            retrain_cooldown_timer -= 1

        # ---------------- 4.10 Буферизація та можливе перенавчання -----------
        if params['enable_retraining']:
            new_x_unscaled = mpc.x_hist.flatten().reshape(1, -1)
            new_y_unscaled = y_meas_unscaled.reshape(1, -1)

            new_x_scaled = x_scaler.transform(new_x_unscaled)
            new_y_scaled = y_scaler.transform(new_y_unscaled)

            retraining_buffer.append((new_x_scaled[0], new_y_scaled[0]))

            if ekf.last_innovation is not None:
                innov_norm = np.linalg.norm(ekf.last_innovation)
                innovation_monitor.append(innov_norm)

            if (t > 0 and
                t % params['retrain_period'] == 0 and
                len(innovation_monitor) == params['retrain_period'] and
                retrain_cooldown_timer == 0):

                avg_innov = float(np.mean(innovation_monitor))

                if avg_innov > params['retrain_innov_threshold']:
                    print(f"\n---> ТРИГЕР ПЕРЕНАВЧАННЯ на кроці {t}! "
                          f"Середня інновація: {avg_innov:.4f} > "
                          f"{params['retrain_innov_threshold']:.4f}")

                    retrain_data = list(retraining_buffer)
                    X_retrain = np.array([p[0] for p in retrain_data])
                    Y_retrain = np.array([p[1] for p in retrain_data])

                    print(f"--> mpc.fit() на {len(X_retrain)} семплах ...")
                    mpc.fit(X_retrain, Y_retrain)
                    print("--> Перенавчання завершено.\n")

                    innovation_monitor.clear()
                    retrain_cooldown_timer = params['retrain_period'] * 2
        # ---------------------------------------------------------------------

        # 4.11 Логування для візуалізації / метрик
        y_true_hist.append(y_meas_unscaled)
        x_hat_hist.append(ekf.x_hat.copy())
        P_hist.append(ekf.P.copy())
        R_hist.append(ekf.R.copy())
        innov_hist.append(
            ekf.last_innovation.copy()
            if ekf.last_innovation is not None
            else np.zeros(ekf.n_dist)
        )

        # <<< ДОДАЄМО ЗБЕРЕЖЕННЯ ДАНИХ >>>
        if u_seq is not None:
            u_seq_hist.append(u_seq)
        if mpc.d_hat is not None:
            # Зберігаємо d_hat в оригінальному масштабі для кращої інтерпретації
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

    # ---------------------------------------------------------------------
    if progress_callback:
        progress_callback(T_sim, T_sim, "Симуляція завершена")
        
    analysis_data = {
        "y_true": np.vstack(y_true_hist),
        "x_hat": np.vstack(x_hat_hist),
        "P": np.stack(P_hist),
        "innov": np.vstack(innov_hist),
        "R": np.stack(R_hist),
        "u_seq": u_seq_hist,
        "d_hat": np.vstack(d_hat_hist) if d_hat_hist else np.array([]),
    }

    return pd.DataFrame(records), analysis_data

def run_post_simulation_analysis(results_df, analysis_data, params):
    """Виконує повний аналіз результатів симуляції та будує графіки."""
    print("\n" + "="*20 + " АНАЛІЗ РЕЗУЛЬТАТІВ " + "="*20)
    
    u_applied = results_df['solid_feed_percent'].values
    d_all = analysis_data['d_all_test'] # Потрібно передати d_all_test в analysis_data
    
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
    if not analysis_data['y_true'] is None:
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
        
    print("="*60 + "\n")
    
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
    min_trust_radius: float =  0.1,
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
    Головна функція-оркестратор для запуску симуляції MPC.
    """
    # Збираємо всі параметри в один словник для зручності передачі
    params = locals()
    
    # 1. Підготовка даних
    true_gen, df_true, X, Y = prepare_simulation_data(reference_df, params)
    data, x_scaler, y_scaler = split_and_scale_data(X, Y, params)

    # 2. Ініціалізація та навчання
    mpc = initialize_mpc_controller_enhanced(params, x_scaler, y_scaler)
    metrics = train_and_evaluate_model(mpc, data, y_scaler)
    
    # Визначення початкової історії для EKF
    n_train_pts = len(data['X_train'])
    n_val_pts = len(data['X_val'])
    test_idx_start = lag + 1 + n_train_pts + n_val_pts
    hist0_unscaled = df_true[['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']].iloc[test_idx_start - (lag + 1): test_idx_start].values
    
    ekf = initialize_ekf(mpc, (x_scaler, y_scaler), hist0_unscaled, data['Y_train_scaled'], lag, params)

    # 3. Запуск симуляції
    results_df, analysis_data = run_simulation_loop_enhanced(true_gen, mpc, ekf, df_true, data, (x_scaler, y_scaler), params, progress_callback) # <<< Передаємо більше даних
    
    # 4. Аналіз результатів
    # Додаємо тестові збурення в словник для аналізу
    test_idx_start = lag + 1 + len(data['X_train']) + len(data['X_val'])
    analysis_data['d_all_test'] = df_true.iloc[test_idx_start:][['feed_fe_percent','ore_mass_flow']].values
    
    if run_analysis:
        run_post_simulation_analysis(results_df, analysis_data, params)
    
    return results_df, metrics



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
        N_data=1000, 
        control_pts=200,
        seed=42,
        
        plant_model_type='rf',
        
        train_size=0.65,
        val_size=0.2,
        test_size=0.15,
    
        # ---- Налаштування моделі
        noise_level='low',
        model_type='svr',
        kernel='rbf', 
        find_optimal_params=True,
        use_soft_constraints=True,
        
        # ---- Налаштування EKF
        P0=1e-2,
        Q_phys=1000, #1000,
        Q_dist=1,
        R=0.18, # 0.18
        q_adaptive_enabled=True,
        q_alpha = 0.995,
        q_nis_threshold = 1.8,

        # Адаптивний Trust Region
        adaptive_trust_region=True,           # Увімкнути адаптацію
        initial_trust_radius=0.3,             # Початковий розмір
        min_trust_radius=0.05,                 # Мінімальний розмір  
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
        delta_u_max = 0.3,
        λ_obj=5.0,
        
        # ---- MPC горизонти
        Nc=4, #8
        Np=6, #12
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



