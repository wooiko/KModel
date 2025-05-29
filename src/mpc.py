# mpc.py

import numpy as np
import cvxpy as cp
from objectives import ControlObjective

class MPCController:
    """
    MPC-контролер з прогнозом на H кроків та буфером останніх L кроків стану.
    Використовує KernelModel для прогнозу та ControlObjective для формування цільової функції.
    """

    def __init__(self,
                 model,
                 objective: ControlObjective,
                 horizon: int = 6,
                 lag: int = 2,
                 u_min: float = 25.0,
                 u_max: float = 35.0,
                 delta_u_max: float = 1.0):
        """
        Args:
            model: об'єкт з методом predict(X) → np.ndarray (n_samples,4)
            objective: інстанс ControlObjective для формування вартості
            horizon: H, число кроків прогнозу
            lag: L, глибина лагів (історія має L+1 станів)
            u_min, u_max: межі для керування solid_feed_percent
            delta_u_max: максимальна зміна u між кроками (від’ємне та додатнє)
        """
        self.model      = model
        self.objective  = objective
        self.H          = horizon
        self.L          = lag
        self.u_min      = u_min
        self.u_max      = u_max
        self.delta_u_max= delta_u_max
        self.x_hist     = None  # Масив форми (L+1, 3): [feed_fe, ore_mass_flow, solid_feed]

    def reset_history(self, initial_history: np.ndarray):
        """
        Ініціалізує буфер історії.
        initial_history.shape == (L+1, 3), колонки: [feed_fe_percent, ore_mass_flow, solid_feed_percent]
        """
        if initial_history.shape != (self.L+1, 3):
            raise ValueError(f"initial_history має форму {(self.L+1,3)}, отримано {initial_history.shape}")
        self.x_hist = initial_history.copy()

    def optimize(self,
                 d_seq: np.ndarray,
                 u_prev: float) -> np.ndarray:
        """
        Розв'язує задачу MPC на горизонт H, повертає оптимальну послідовність u[0:H].

        Args:
            d_seq: масив збурень форми (H, 2): стовпці [feed_fe_percent, ore_mass_flow]
            u_prev: попереднє значення solid_feed_percent (скаляр)

        Returns:
            np.ndarray форми (H,) з керуванням на кожен крок
        """
        if self.x_hist is None:
            raise RuntimeError("Буфер історії не ініціалізовано. Викличте reset_history().")
        if d_seq.shape != (self.H, 2):
            raise ValueError(f"d_seq має форму {(self.H,2)}, отримано {d_seq.shape}")

        # Змінні оптимізації: u[0],…,u[H-1]
        u = cp.Variable(self.H)
        constraints = [
            u >= self.u_min,
            u <= self.u_max
        ]
        # Обмеження на зміну керування
        for k in range(self.H):
            prev = u_prev if k == 0 else u[k-1]
            constraints.append(cp.abs(u[k] - prev) <= self.delta_u_max)

        cost_terms = []
        # Локальна копія історії для прогнозу
        xk = self.x_hist.copy()

        for k in range(self.H):
            # Формуємо вхід для моделі: послідовність L+1 кроків history
            Xk = xk.flatten().reshape(1, -1)  # shape (1, 3*(L+1))
            # Прогноз вихідних: [conc_fe, tail_fe, conc_mass, tail_mass]
            yk = self.model.predict(Xk)[0]

            # Додаємо терм цільової функції через об'єкт-стратегію
            uk = u[k]
            prev = u_prev if k == 0 else u[k-1]
            cost_terms.append(self.objective.cost_term(yk, uk, prev))

            # Оновлюємо історію: зсуваємо на 1 крок і додаємо новий стан
            # Новий стан: [feed_fe, ore_mass_flow, solid_feed]
            feed_fe, ore_flow = d_seq[k]
            # append row [feed_fe, ore_flow, u_k]
            xk = np.roll(xk, -1, axis=0)
            xk[-1, :] = [feed_fe, ore_flow, uk]

        # Сформуємо і вирішимо оптимізаційну задачу
        problem = cp.Problem(cp.Minimize(cp.sum(cost_terms)), constraints)
        problem.solve(solver=cp.OSQP)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC optimization failed: {problem.status}")

        return u.value