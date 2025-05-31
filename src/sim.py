# sim.py

import numpy as np
import pandas as pd

from data_gen import DataGenerator
from model import KernelModel
from objectives import MaxIronMassObjective
from mpc import MPCController
from utils import train_val_test, compute_metrics


def simulate_mpc(
    reference_df: pd.DataFrame,
    N_data: int = 1000,
    control_pts: int = 200,
    lag: int = 2,
    horizon: int = 6,
    n_neighbors: int = 5,
    model_type: str = 'krr',
    kernel: str = 'linear',
    alpha: float = 1.0,
    gamma: float = None,
    λ_obj: float = 0.1,
    w_fe: float = 1.0,
    w_mass: float = 1.0
):
    """
    Повна симуляція MPC у закритому циклі:
      1) Генеруємо «справжні» дані через DataGenerator.
      2) Будуємо лаговані X, Y і ділимо на train/val/test.
      3) Навчаємо KernelModel.
      4) Ініціалізуємо MPCController з MaxIronMassObjective.
      5) Закритий цикл довжини T_sim:
         - для кожного кроку формуємо d_seq, оптимізуємо QP, отримуємо u_k,
         - робимо «справжній» крок через true_gen, оновлюємо історію,
         - зберігаємо результати.
      6) Обчислюємо метрики.

    Повертає:
      results_df: DataFrame з колонками
        ['feed_fe_percent','ore_mass_flow','solid_feed_percent',
         'conc_fe','tail_fe','conc_mass','tail_mass',
         'mass_pull_pct','fe_recovery_pct']
      metrics: dict з { 'mae': DataFrame, 'avg_iron_mass': float }
    """
    # 1. «Справжній» генератор
    true_gen = DataGenerator(reference_df, ore_flow_var_pct=3.0)
    df_true  = true_gen.generate(N_data, control_pts, n_neighbors)

    # 2. Лаговані X, Y і train/val/test
    X, Y = DataGenerator.create_lagged_dataset(df_true, lags=lag)
    (X_train, Y_train,
     X_val,   Y_val,
     X_test,  Y_test) = train_val_test(
         X, Y, train_size=0.7, val_size=0.15, test_size=0.15
    )

    # 3–4. Створюємо модель і контролер
    km = KernelModel(model_type=model_type,
                     kernel=kernel,
                     alpha=alpha,
                     gamma=gamma)

    objective = MaxIronMassObjective(λ=λ_obj,
                                     w_fe=w_fe,
                                     w_mass=w_mass)

    mpc = MPCController(model=km,
                        objective=objective,
                        horizon=horizon,
                        lag=lag,
                        u_min=25.0,
                        u_max=35.0,
                        delta_u_max=1.0)

    # 5a. Навчаємо модель і ініціалізуємо історію
    cols_state = ['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']
    hist0 = df_true[cols_state].iloc[: lag + 1].values  # (lag+1,3)
    mpc.fit(X_train, Y_train, hist0)

    # Підготовка до циклу
    d_all        = df_true[['feed_fe_percent', 'ore_mass_flow']].values
    T_sim        = len(df_true) - (lag + 1)
    records      = []
    pred_records = []  # для збереження прогнозів
    u_prev       = float(hist0[-1, 2])

    for t in range(T_sim):
        # 5b. Формуємо d_seq довжини horizon
        d_seq = d_all[t+1: t+1 + horizon]
        if d_seq.shape[0] < horizon:
            pad   = np.repeat(d_seq[-1][None, :], horizon - d_seq.shape[0], axis=0)
            d_seq = np.vstack([d_seq, pad])

        # 5c. Оптимізація MPC → u_seq
        u_seq = mpc.optimize(d_seq, u_prev)
        u_cur = float(u_seq[0])

        # 5d. «Справжній» крок процесу
        inp    = pd.DataFrame(
            [[d_all[t+1, 0], d_all[t+1, 1], u_cur]],
            columns=cols_state
        )
        y_pred = true_gen._predict_outputs(inp)
        y_corr = true_gen._apply_mass_balance(inp, y_pred)

        # зберігаємо прогнозні значення для compute_metrics
        pred_records.append({
            'conc_fe':   y_pred.concentrate_fe_percent.iloc[0],
            'tail_fe':   y_pred.tailings_fe_percent.iloc[0],
            'conc_mass': y_pred.concentrate_mass_flow.iloc[0],
            'tail_mass': y_pred.tailings_mass_flow.iloc[0],
        })

        # додаємо необхідні стовпці для _derive
        y_corr['ore_mass_flow']   = inp['ore_mass_flow'].values
        y_corr['feed_fe_percent'] = inp['feed_fe_percent'].values

        y_full = true_gen._derive(y_corr)

        # 5e. Зберігаємо результати «справжніх» виходів
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

        # 5f. Оновлюємо історію у контролері
        new_state = np.array([
            records[-1]['feed_fe_percent'],
            records[-1]['ore_mass_flow'],
            records[-1]['solid_feed_percent']
        ])
        xk = np.roll(mpc.x_hist, -1, axis=0)
        xk[-1, :] = new_state
        mpc.x_hist = xk
        u_prev     = u_cur

    # 6. Конвертуємо списки в DataFrame
    results_df = pd.DataFrame(records)
    preds_df   = pd.DataFrame(pred_records)

    # Обчислюємо метрики за справжніми та прогнозованими
    mae = compute_metrics(
        results_df[['conc_fe','tail_fe','conc_mass','tail_mass']],
        preds_df
    )

    # Середня маса заліза в концентраті
    iron_mass     = results_df.conc_fe * results_df.conc_mass / 100
    avg_iron_mass = iron_mass.mean()

    metrics = {
        'mae': mae,
        'avg_iron_mass': avg_iron_mass
    }

    return results_df, metrics


if __name__ == '__main__':
    hist_df = pd.read_parquet('processed.parquet')
    results, metrics = simulate_mpc(hist_df)
    print("Метрики:", metrics)
    results.to_parquet('mpc_simulation_results.parquet')