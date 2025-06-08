# sim.py

import numpy as np
import pandas as pd
from typing import Callable

from data_gen import DataGenerator
from model import KernelModel
from objectives import MaxIronMassTrackingObjective
from mpc import MPCController
from anomaly_detector import AnomalyDetector
from kalman_observer import DisturbanceObserverKalman
from utils import (compute_metrics, train_val_test_time_series, analyze_sensitivity, 
                   analize_errors, plot_control_and_disturbances, plot_historical_data, 
                   plot_fact_vs_mpc_plans, visualize_comparison)

def simulate_mpc(
    reference_df: pd.DataFrame,
    N_data: int = 5000,
    control_pts: int = 1000,
    noise_level: str = 'medium',  # none, low, medium, high
    lag: int = 2,
    Np: int = 6,       # prediction horizon
    Nc: int = 3,       # control horizon, Nc <= Np
    n_neighbors: int = 5,
    model_type: str = 'krr',
    kernel: str = 'linear',
    alpha: float = 1.0,
    gamma: float = None,
    λ_obj: float = 0.1,
    K_I: float = 0.01,
    w_fe: float = 7.0,
    w_mass: float = 1.0,
    ref_fe: float = 54.5,
    ref_mass: float = 58.0,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    u_min: float = 20.0,
    u_max: float = 40.0,
    delta_u_max: float = 1.0,
    progress_callback: Callable[[int, int, str], None] = None
):
    # 1. «Справжній» генератор даних
    true_gen = DataGenerator(reference_df, ore_flow_var_pct=3.0)
    anomaly_config = DataGenerator.generate_anomaly_config(
        N_data, train_frac=train_size,
        val_frac=val_size, test_frac=test_size,
        seed=42
    )
    anomaly_config = None
    df_true = true_gen.generate(N_data, control_pts, n_neighbors, noise_level=noise_level, anomaly_config=anomaly_config)
    
    # (Опційно) можна візуалізувати сирі дані, якщо потрібно:
    # visualize_comparison(df_true=df_true, df_filtered=df_true)
    
    # 2. Створення лагованого набору X, Y та розбиття на train/val/test
    X, Y = DataGenerator.create_lagged_dataset(df_true, lags=lag)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = \
        train_val_test_time_series(X, Y, train_size, val_size, test_size)
    
    # 3–4. Створення прогнозної моделі та MPC-контролера
    km = KernelModel(model_type=model_type, kernel=kernel, alpha=alpha, gamma=gamma)
    obj = MaxIronMassTrackingObjective(
        λ=λ_obj, w_fe=w_fe, w_mass=w_mass,
        ref_fe=ref_fe, ref_mass=ref_mass, K_I=K_I
    )
    mpc = MPCController(
        model=km, objective=obj,
        horizon=Np, control_horizon=Nc,
        lag=lag, u_min=u_min, u_max=u_max,
        delta_u_max=delta_u_max
    )
    
    # Ініціалізація Калманівських спостерігачів для компенсації зсуву виходів
    d_obs_fe = DisturbanceObserverKalman()
    d_obs_mass = DisturbanceObserverKalman()
    mpc.d_est = np.array([0.0, 0.0])
    
    # --- Ініціалізація засобів для очищення вхідних сигналів в режимі онлайн ---
    # Цей набір використовується в циклі симуляції «на льоту».
    ad_feed_online = AnomalyDetector(window=5, z_thresh=3.0)
    ad_ore_online  = AnomalyDetector(window=5, z_thresh=3.0)
   
    A_d = 1.0
    C_d = 1.0
    Q     = 1e-7
    R     = 1e2
    P     = 500.0
    d_est = 1.1    

    d_obs_feed_online = DisturbanceObserverKalman(
        A_d   = 1.0,
        C_d   = 1.0,
        Q     = 1e-1,  # Підняли Q для кращого відстеження змін
        R     = 1e-3,  # Трохи зменшили R
        P     = 15.0,
        d_est = 0.0
    )
    
    d_obs_ore_online = DisturbanceObserverKalman(
        A_d   = 1.0,
        C_d   = 1.0,
        Q     = 1e-1,
        R     = 1e-3,
        P     = 15.0,
        d_est = 0.0
    )
    
    # 5a. Навчання моделі на train та ініціалізація історії
    cols_state = ['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']
    hist_train = df_true[cols_state].iloc[:lag+1].values
    mpc.fit(X_train, Y_train, hist_train)
    
    # 5b. Визначення початку тестової ділянки
    n = X.shape[0]
    n_train = int(train_size * n)
    n_val = int(val_size * n)
    test_idx = (lag + 1) + n_train + n_val
    
    # Історія для симуляції: останні (lag+1) точок перед початком тестового сегмента
    hist0 = df_true[cols_state].iloc[test_idx - (lag + 1): test_idx].values
    
    # Тестові дані: використовуємо сирі дані (df_true), очищення буде виконуватись онлайн
    df_run = df_true.iloc[test_idx:]
    d_all = df_run[['feed_fe_percent', 'ore_mass_flow']].values
    T_sim = len(df_run) - (lag + 1)
    
    records, pred_records = [], []
    all_u_sequences, control_steps = [], []
    u_applied = []
    u_prev = float(hist0[-1, 2])
    

    # Перед циклом ініціалізуємо «попередні» відфільтровані та прогнозовані значення
    prev_feed_pred = hist0[-1, 0]
    prev_ore_pred  = hist0[-1, 1]
    u_prev         = hist0[-1, 2]
    
    for t in range(T_sim):
        if progress_callback:
            progress_callback(t, T_sim, f"Крок {t+1}/{T_sim}")
    
        # 1) Сирі вимірювання на кроці t+1
        raw_feed, raw_ore = d_all[t+1]
    
        # 2) Первинна корекція аномалій
        corr_feed = ad_feed_online.correct(raw_feed)
        corr_ore  = ad_ore_online.correct(raw_ore)
    
        # 3) Оцінка збурення для поточного кроку та отримання відфільтрованих значень
        d_feed_est = d_obs_feed_online.update(corr_feed, prev_feed_pred)
        d_ore_est  = d_obs_ore_online.update(corr_ore,  prev_ore_pred)
        feed_filt  = corr_feed - d_feed_est
        ore_filt   = corr_ore  - d_ore_est
    
        # Запам'ятовуємо цей результат як перший елемент горизонту
        d_seq_clean = [[feed_filt, ore_filt]]
    
        # Оновлюємо попередні прогнози під Kalman’ом
        prev_feed_pred = feed_filt
        prev_ore_pred  = ore_filt
    
        # 4) Формуємо та очищуємо решту горизонту прогнозу Np
        d_seq_raw = d_all[t+1 : t+1 + Np]
        if len(d_seq_raw) < Np:
            pad = np.repeat(d_seq_raw[-1][None, :], Np - len(d_seq_raw), axis=0)
            d_seq_raw = np.vstack([d_seq_raw, pad])
    
        for k in range(1, Np):
            raw_f_k, raw_o_k = d_seq_raw[k]
    
            # детектор аномалій
            cf = ad_feed_online.correct(raw_f_k)
            co = ad_ore_online.correct(raw_o_k)
    
            # Kalman-спостереження
            d_f_k = d_obs_feed_online.update(cf, prev_feed_pred)
            d_o_k = d_obs_ore_online.update(co,  prev_ore_pred)
    
            # відфільтроване значення
            f_filt_k = cf - d_f_k
            o_filt_k = co - d_o_k
    
            d_seq_clean.append([f_filt_k, o_filt_k])
    
            # оновлюємо «попереднє» для наступного кроку горизонту
            prev_feed_pred = f_filt_k
            prev_ore_pred  = o_filt_k
    
        d_seq_clean = np.array(d_seq_clean)
    
        # 5) Оптимізація MPC та застосування першого керуючого впливу
        u_seq = mpc.optimize(d_seq_clean, u_prev)
        u_cur = float(u_seq[0])
        all_u_sequences.append(u_seq)
        control_steps.append(t)
        u_applied.append(u_cur)
    
        # 6) Прогнозування і коригування виходів моделі
        inp    = pd.DataFrame([[feed_filt, ore_filt, u_cur]], columns=cols_state)
        y_pred = true_gen._predict_outputs(inp)
        y_corr = true_gen._apply_mass_balance(inp, y_pred)
    
        # Збираємо передбачення для метрик
        pred_records.append({
            'conc_fe':   y_pred.concentrate_fe_percent.iloc[0],
            'tail_fe':   y_pred.tailings_fe_percent.iloc[0],
            'conc_mass': y_pred.concentrate_mass_flow.iloc[0],
            'tail_mass': y_pred.tailings_mass_flow.iloc[0],
        })
    
        # 7) Остаточний розрахунок виходу
        y_corr['ore_mass_flow']   = inp.ore_mass_flow.values
        y_corr['feed_fe_percent'] = inp.feed_fe_percent.values
        y_full = true_gen._derive(y_corr)
    
        records.append({
            'feed_fe_percent':  feed_filt,
            'ore_mass_flow':    ore_filt,
            'solid_feed_percent': u_cur,
            'conc_fe':         y_full.concentrate_fe_percent.iloc[0],
            'tail_fe':         y_full.tailings_fe_percent.iloc[0],
            'conc_mass':       y_full.concentrate_mass_flow.iloc[0],
            'tail_mass':       y_full.tailings_mass_flow.iloc[0],
            'mass_pull_pct':   y_full.mass_pull_percent.iloc[0],
            'fe_recovery_pct': y_full.fe_recovery_percent.iloc[0],
        })
    
        # 8) Оновлення спостерігачів для компенсації зсуву виходів
        new_d_fe   = d_obs_fe.update(y_full.concentrate_fe_percent.iloc[0],
                                     y_pred.concentrate_fe_percent.iloc[0])
        new_d_mass = d_obs_mass.update(y_full.concentrate_mass_flow.iloc[0],
                                       y_pred.concentrate_mass_flow.iloc[0])
        mpc.d_est = np.array([new_d_fe, new_d_mass])
    
        # 9) Оновлюємо історію стану та u_prev
        xk = np.roll(mpc.x_hist, -1, axis=0)
        xk[-1] = [feed_filt, ore_filt, u_cur]
        mpc.x_hist = xk
        u_prev = u_cur
    
    if progress_callback:
        progress_callback(T_sim, T_sim, "Симуляція завершена")
    
    # Збір результатів і розрахунок метрик
    results_df = pd.DataFrame(records)
    preds_df = pd.DataFrame(pred_records)
    metrics = {
        'mae': compute_metrics(
            results_df[['conc_fe', 'tail_fe', 'conc_mass', 'tail_mass']],
            preds_df),
        'avg_iron_mass': (results_df.conc_fe * results_df.conc_mass / 100).mean()
    }
    
    # (Опційно) побудова діагностичних графіків
    # plot_historical_data(results_df, columns=['conc_fe', 'conc_mass'])
    # analize_errors(results_df, ref_fe, ref_mass)
    # plot_fact_vs_mpc_plans(results_df, all_u_sequences, control_steps)
    # plot_control_and_disturbances(np.array(u_applied), d_all[1:1+len(u_applied)])
    # Приклад обрізання для візуалізації:
    T_sim = results_df.shape[0]
    df_true_vis = df_true.iloc[test_idx:test_idx + T_sim].copy()
    visualize_comparison(df_true=df_true_vis, df_filtered=results_df)    
    return results_df, metrics


if __name__ == '__main__':
    def my_progress(step, total, msg):
        print(f"[{step}/{total}] {msg}")
    
    hist_df = pd.read_parquet('processed.parquet')
    
    N_data = 400
    control_pts = int(N_data * 0.1)
    
    res, mets = simulate_mpc(hist_df, progress_callback=my_progress, N_data=N_data, control_pts=control_pts)
    
    print("=" * 50)
    res.to_parquet('mpc_simulation_results.parquet')