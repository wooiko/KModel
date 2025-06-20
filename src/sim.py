# sim.py

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_gen import StatefulDataGenerator
from model import KernelModel
from objectives import MaxIronMassTrackingObjective
from mpc import MPCController
from utils import (analize_errors, plot_control_and_disturbances, plot_mpc_diagnostics)
from ekf import ExtendedKalmanFilter

# =============================================================================
# === БЛОК 1: ПІДГОТОВКА ДАНИХ ТА СКАЛЕРІВ ===
# =============================================================================

def prepare_simulation_data(
    reference_df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[StatefulDataGenerator, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Створює генератор, генерує часовий ряд та створює лаговані датасети.
    
    Args:
        reference_df: DataFrame з референсними даними.
        params: Словник з параметрами симуляції.

    Returns:
        Кортеж з генератором, повним DataFrame, вхідними (X) та вихідними (Y) даними.
    """
    print("Крок 1: Генерація симуляційних даних...")
    true_gen = StatefulDataGenerator(
        reference_df,
        ore_flow_var_pct=3.0,
        time_step_s=5.0,
        time_constant_s=8.0,
        dead_time_s=20.0,
        true_model_type=params['plant_model_type']
    )
    
    df_true = true_gen.generate(
        params['N_data'],
        params['control_pts'],
        params['n_neighbors'],
        noise_level=params['noise_level']
    )
    
    X, Y_full_np = StatefulDataGenerator.create_lagged_dataset(df_true, lags=params['lag'])
    Y = Y_full_np[:, [0, 2]]  # Вибираємо лише Fe концентрату та масу концентрату
    
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


def initialize_ekf(
    mpc: MPCController,
    scalers: Tuple[StandardScaler, StandardScaler],
    hist0_unscaled: np.ndarray,
    Y_train_scaled: np.ndarray,
    lag: int
) -> ExtendedKalmanFilter:
    """
    Ініціалізує розширений фільтр Калмана (EKF).
    """
    print("Крок 4: Ініціалізація фільтра Калмана (EKF)...")
    x_scaler, y_scaler = scalers
    n_phys, n_dist = (lag + 1) * 3, 2
    
    x0_aug = np.hstack([hist0_unscaled.flatten(), np.zeros(n_dist)])
    
    P0 = np.eye(n_phys + n_dist) * 1e-2
    P0[n_phys:, n_phys:] *= 1000

    Q_phys = np.eye(n_phys) * 1e-6
    Q_dist = np.eye(n_dist) * 1e-4
    Q = np.block([[Q_phys, np.zeros((n_phys, n_dist))], [np.zeros((n_dist, n_phys)), Q_dist]])
    
    R = np.diag(np.var(Y_train_scaled, axis=0)) * 0.5

    return ExtendedKalmanFilter(mpc.model, x_scaler, y_scaler, x0_aug, P0, Q, R, lag)


# =============================================================================
# === БЛОК 3: ОСНОВНИЙ ЦИКЛ СИМУЛЯЦІЇ ===
# =============================================================================

def run_simulation_loop(
    true_gen: StatefulDataGenerator,
    mpc: MPCController,
    ekf: ExtendedKalmanFilter,
    df_true: pd.DataFrame,
    params: Dict[str, Any],
    progress_callback: Callable
) -> pd.DataFrame:
    """
    Виконує основний замкнений цикл симуляції MPC.
    """
    print("Крок 5: Запуск основного циклу симуляції...")
    
    # Визначення початкових умов для тестової вибірки
    n_total = len(df_true) - params['lag'] - 1
    n_train = int(params['train_size'] * n_total)
    n_val = int(params['val_size'] * n_total)
    test_idx_start = params['lag'] + 1 + n_train + n_val
    
    hist0_unscaled = df_true[['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']].iloc[test_idx_start - (params['lag'] + 1): test_idx_start].values
    
    # Ініціалізація станів MPC, EKF та симулятора
    mpc.reset_history(hist0_unscaled)
    true_gen.reset_state(hist0_unscaled)
    
    # Налаштування змінних для циклу
    df_run = df_true.iloc[test_idx_start:]
    d_all = df_run[['feed_fe_percent', 'ore_mass_flow']].values
    T_sim = len(df_run) - (params['lag'] + 1)
    
    records = []
    u_prev = float(hist0_unscaled[-1, 2])
    d_filtered_state = d_all[0, :].copy()

    for t in range(T_sim):
        if progress_callback:
            progress_callback(t, T_sim, f"Крок симуляції {t+1}/{T_sim}")

        # 1. Прогноз EKF
        ekf.predict(u_prev, d_all[t, :])
        
        # 2. Оцінка стану та збурень
        x_est_phys_unscaled = ekf.x_hat[:ekf.n_phys].reshape(params['lag'] + 1, 3)
        mpc.reset_history(x_est_phys_unscaled)
        mpc.d_hat = ekf.x_hat[ekf.n_phys:]

        # 3. Розрахунок керуючої дії
        d_filtered_state = 0.1 * d_all[t+1, :] + 0.9 * d_filtered_state
        d_seq = np.repeat(d_filtered_state[None, :], params['Np'], axis=0)
        u_seq = mpc.optimize(d_seq, u_prev)
        u_cur = u_prev if u_seq is None else float(u_seq[0])

        # 4. Крок симуляції реального процесу
        feed_fe_next, ore_flow_next = d_all[t+1]
        y_full = true_gen.step(feed_fe_next, ore_flow_next, u_cur)
        
        # 5. Корекція EKF
        y_meas_unscaled = y_full[['concentrate_fe_percent', 'concentrate_mass_flow']].values.flatten()
        ekf.update(y_meas_unscaled)

        # ===> ЗМІНА: ЯВНЕ СТВОРЕННЯ СЛОВНИКА З ПЕРЕЙМЕНУВАННЯМ <===
        # 6. Збереження результатів
        # Замість y_full.iloc[0].to_dict() використовуємо явне створення словника,
        # щоб перейменувати стовпці для сумісності з функціями аналізу.
        y_meas = y_full.iloc[0]
        records.append({
            'feed_fe_percent': y_meas.feed_fe_percent,
            'ore_mass_flow': y_meas.ore_mass_flow,
            'solid_feed_percent': u_cur,
            'conc_fe': y_meas.concentrate_fe_percent,    # Перейменовано
            'tail_fe': y_meas.tailings_fe_percent,
            'conc_mass': y_meas.concentrate_mass_flow,  # Перейменовано
            'tail_mass': y_meas.tailings_mass_flow,
            'mass_pull_pct': y_meas.mass_pull_percent,
            'fe_recovery_percent': y_meas.fe_recovery_percent,
        })
        # ===> КІНЕЦЬ ЗМІНИ <===
        
        u_prev = u_cur

    if progress_callback:
        progress_callback(T_sim, T_sim, "Симуляція завершена")
        
    return pd.DataFrame(records)


# =============================================================================
# === ГОЛОВНА ФУНКЦІЯ-ОРКЕСТРАТОР ===
# =============================================================================

def simulate_mpc(
    reference_df: pd.DataFrame,
    N_data: int = 5000, control_pts: int = 1000, lag: int = 2, Np: int = 6, Nc: int = 4,
    n_neighbors: int = 5, seed: int = 0, noise_level: str = 'none', model_type: str = 'krr',
    kernel: str = 'rbf', find_optimal_params: bool = True, λ_obj: float = 0.1, K_I: float = 0.01,
    w_fe: float = 7.0, w_mass: float = 1.0, ref_fe: float = 53.5, ref_mass: float = 57.0,
    train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15,
    u_min: float = 20.0, u_max: float = 40.0, delta_u_max: float = 1.0,
    use_disturbance_estimator: bool = True, y_max_fe: float = 54.5, y_max_mass: float = 58.0,
    rho_trust: float = 0.1, use_soft_constraints: bool = True, plant_model_type: str = 'rf',
    progress_callback: Callable[[int, int, str], None] = None
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
    mpc = initialize_mpc_controller(params, x_scaler, y_scaler)
    metrics = train_and_evaluate_model(mpc, data, y_scaler)
    
    # Визначення початкової історії для EKF
    n_train_pts = len(data['X_train'])
    n_val_pts = len(data['X_val'])
    test_idx_start = lag + 1 + n_train_pts + n_val_pts
    hist0_unscaled = df_true[['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']].iloc[test_idx_start - (lag + 1): test_idx_start].values
    
    ekf = initialize_ekf(mpc, (x_scaler, y_scaler), hist0_unscaled, data['Y_train_scaled'], lag)

    # 3. Запуск симуляції
    results_df = run_simulation_loop(true_gen, mpc, ekf, df_true, params, progress_callback)
    
    # 4. Аналіз результатів
    print("\nАналіз результатів симуляції:")
    analize_errors(results_df, ref_fe, ref_mass)
    u_applied = results_df['solid_feed_percent'].values
    d_all_test = df_true.iloc[test_idx_start:][['feed_fe_percent','ore_mass_flow']].values
    plot_control_and_disturbances(u_applied, d_all_test[1:1+len(u_applied)])
    plot_mpc_diagnostics(results_df, w_fe, w_mass, λ_obj)
    
    final_avg_iron_mass = (results_df.conc_fe * results_df.conc_mass / 100).mean()
    metrics['avg_iron_mass'] = final_avg_iron_mass
    print(f"Середня маса заліза в концентраті: {final_avg_iron_mass:.2f} т/год")
    
    print("=" * 50)
    return results_df, metrics


# =============================================================================
# === ТОЧКА ВХОДУ (ЯКЩО ФАЙЛ ЗАПУСКАЄТЬСЯ НАПРЯМУ) ===
# =============================================================================

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
        N_data=1000, 
        control_pts=100,
        seed=42,
        
        train_size = 0.7,
        val_size   = 0.15,
        test_size  = 0.15,

        noise_level='low',
        model_type = 'krr',
        kernel='linear', 
        find_optimal_params=True,
        use_soft_constraints=True,

        # 1. Збільшуємо вагу регіону довіри (найважливіша зміна)
        rho_trust = 50.0,

        # 2. Збалансовуємо ваги цілі для масштабованих даних
        w_fe = 1.0,
        w_mass = 1.0,

        # 3. Задаємо уставки та м'які обмеження
        ref_fe = 54.0,
        ref_mass = 58.2,
        y_max_fe = 55.0,
        y_max_mass = 60.0
    )
    
    print("\nРезультати симуляції (останні 5 кроків):")
    print(res.tail())
    print("\nФінальні метрики:")
    print(mets)
    res.to_parquet('mpc_simulation_results.parquet')