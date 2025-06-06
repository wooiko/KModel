# sim.py

import numpy as np
import pandas as pd
from typing import Callable

from data_gen import DataGenerator
from model import KernelModel
from objectives import  MaxIronMassTrackingObjective
from mpc import MPCController
from utils import compute_metrics, train_val_test_time_series, analyze_sensitivity, analize_errors, plot_control_and_disturbances, plot_historical_data, plot_fact_vs_mpc_plans

def simulate_mpc(
    reference_df: pd.DataFrame,
    N_data: int = 5000,
    control_pts: int = 1000,
    lag: int = 2,
    # horizon: int = 6,
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
    val_size: float   = 0.15,
    test_size: float  = 0.15,
    u_min: float  = 20.0, 
    u_max: float  = 40.0, 
    delta_u_max: float  = 1.0,
    progress_callback: Callable[[int, int, str], None] = None
):
    # 1. «справжній» генератор
    true_gen = DataGenerator(reference_df, ore_flow_var_pct=3.0)
    df_true  = true_gen.generate(N_data, control_pts, n_neighbors)

    # 2. Лаговані X, Y і послідовне розбиття на train/val/test
    X, Y = DataGenerator.create_lagged_dataset(df_true, lags=lag)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = \
        train_val_test_time_series(X, Y, train_size, val_size, test_size)

    # 3–4. Модель і MPC-контролер
    km = KernelModel(model_type=model_type,
                     kernel=kernel,
                     alpha=alpha,
                     gamma=gamma)
    
    obj = MaxIronMassTrackingObjective(
        λ=λ_obj, 
        w_fe=w_fe, 
        w_mass=w_mass,
        ref_fe=ref_fe, 
        ref_mass=ref_mass,
        K_I=K_I
    )
    
    mpc = MPCController(
        model=km,
        objective=obj,
        horizon=Np,                # раніше prediction_horizon
        control_horizon=Nc,        # новий параметр
        lag=lag,
        u_min=u_min, u_max=u_max,
        delta_u_max=delta_u_max
    )

    # 5a. Навчаємо модель на train і ініціалізуємо історію
    cols_state = ['feed_fe_percent','ore_mass_flow','solid_feed_percent']
    hist_train = df_true[cols_state].iloc[:lag+1].values
    mpc.fit(X_train, Y_train, hist_train)

    # 5b. Визначаємо початок тестової ділянки у df_true
    n        = X.shape[0]
    n_train  = int(train_size * n)
    n_val    = int(val_size * n)
    test_idx = (lag + 1) + n_train + n_val

    # Історія для симуляції (lag+1 точок перед тестом)
    hist0  = df_true[cols_state]\
                .iloc[test_idx - (lag + 1): test_idx]\
                .values

    # Самі «тестові» дані
    df_run = df_true.iloc[test_idx:]
    d_all  = df_run[['feed_fe_percent','ore_mass_flow']].values
    T_sim  = len(df_run) - (lag + 1)

    # 5c–5f. Замкнений цикл MPC тільки на тесті
    records, pred_records = [], []
    all_u_sequences, control_steps = [], []
    u_applied = []
    u_prev = float(hist0[-1, 2])

    for t in range(T_sim):
        if progress_callback:
            progress_callback(t, T_sim, f"Крок {t+1}/{T_sim}")

        # формуємо d_seq
        d_seq = d_all[t+1 : t+1 + Np]                         # ❸
        if len(d_seq) < Np:
            pad   = np.repeat(d_seq[-1][None, :], Np - len(d_seq), axis=0)
            d_seq = np.vstack([d_seq, pad])

        # оптимізація та реальний крок
        u_seq = mpc.optimize(d_seq, u_prev)
        all_u_sequences.append(u_seq)
        control_steps.append(t)   
        u_cur = float(u_seq[0])
        u_applied.append(u_cur)

        inp    = pd.DataFrame([[ *d_all[t+1], u_cur ]], columns=cols_state)
        y_pred = true_gen._predict_outputs(inp)
        y_corr = true_gen._apply_mass_balance(inp, y_pred)

        pred_records.append({
            'conc_fe':   y_pred.concentrate_fe_percent.iloc[0],
            'tail_fe':   y_pred.tailings_fe_percent.iloc[0],
            'conc_mass': y_pred.concentrate_mass_flow.iloc[0],
            'tail_mass': y_pred.tailings_mass_flow.iloc[0],
        })

        # повний вихід
        y_corr['ore_mass_flow']   = inp.ore_mass_flow.values
        y_corr['feed_fe_percent'] = inp.feed_fe_percent.values
        y_full = true_gen._derive(y_corr)

        records.append({
            'feed_fe_percent':    inp.feed_fe_percent.iloc[0],
            'ore_mass_flow':      inp.ore_mass_flow.iloc[0],
            'solid_feed_percent': u_cur,
            'conc_fe':            y_full.concentrate_fe_percent.iloc[0],
            'tail_fe':            y_full.tailings_fe_percent.iloc[0],
            'conc_mass':          y_full.concentrate_mass_flow.iloc[0],
            'tail_mass':          y_full.tailings_mass_flow.iloc[0],
            'mass_pull_pct':      y_full.mass_pull_percent.iloc[0],
            'fe_recovery_pct':    y_full.fe_recovery_percent.iloc[0],
        })

        # оновлюємо історію
        new_state = np.array([
            records[-1]['feed_fe_percent'],
            records[-1]['ore_mass_flow'],
            records[-1]['solid_feed_percent']
        ])
        xk = np.roll(mpc.x_hist, -1, axis=0)
        xk[-1] = new_state
        mpc.x_hist = xk
        u_prev     = u_cur

    if progress_callback:
        progress_callback(T_sim, T_sim, "Симуляція завершена")

    # 6. Збір результатів і метрик
    results_df = pd.DataFrame(records)
    preds_df   = pd.DataFrame(pred_records)
    metrics = {
        'mae': compute_metrics(
            results_df[['conc_fe','tail_fe','conc_mass','tail_mass']],
            preds_df),
        'avg_iron_mass': (results_df.conc_fe * results_df.conc_mass / 100).mean()
    }


    columns=['conc_fe', 'conc_mass']
    plot_historical_data(results_df, columns= columns)
    
    analize_errors(results_df, ref_fe, ref_mass)
    
    # 1) порівняльний графік планів MPC vs факт
    # plot_fact_vs_mpc_plans(results_df, all_u_sequences, control_steps)

    # 2) фактичний applied-u + збурення в стилі step
    plot_control_and_disturbances(
        np.array(u_applied),
        d_all[1:1+len(u_applied)]
    )

    print("=" * 50)   
    
    return results_df, metrics

if __name__ == '__main__':
    def my_progress(step, total, msg):
        print(f"[{step}/{total}] {msg}")

    hist_df = pd.read_parquet('processed.parquet')
    
    N_data = 500 
    control_pts = int(N_data * 0.2) 
    
    res, mets = simulate_mpc(hist_df, progress_callback=my_progress, N_data=N_data, control_pts=control_pts)
    
    print("=" * 50)
    # print("Метрики:", mets)
    res.to_parquet('mpc_simulation_results.parquet')