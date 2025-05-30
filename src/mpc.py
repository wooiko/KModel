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

    def optimize(self, d_seq: np.ndarray, u_prev: float) -> np.ndarray:
        """
        Розв'язує задачу MPC на горизонт H, повертає np.ndarray(H,) з керуванням.
        Args:
            d_seq: масив збурень форми (H, 2): стовпці [feed_fe_percent, ore_mass_flow]
            u_prev: попереднє значення solid_feed_percent (скаляр)
        Returns:
            np.ndarray форми (H,) з керуванням на кожен крок
        """
        # Перевірки вхідних даних
        if self.x_hist is None:
            raise RuntimeError("Буфер історії не ініціалізовано. Викличте reset_history().")
        if d_seq.shape != (self.H, 2):
            raise ValueError(f"d_seq має форму {(self.H,2)}, отримано {d_seq.shape}")
    
        # Змінні оптимізації
        u = cp.Variable(self.H)
        constraints = [
            u >= self.u_min,
            u <= self.u_max
        ]
        # Обмеження на зміну керування
        for k in range(self.H):
            prev_u = u_prev if k == 0 else u[k-1]
            constraints.append(cp.abs(u[k] - prev_u) <= self.delta_u_max)
    
        # Ініціалізуємо історію як список списків: [[f,o,s], ...]
        xk_list = [row.tolist() for row in self.x_hist]
    
        cost_terms = []
        # Основний цикл формування цільової функції
        for k in range(self.H):
            # 1) Flatten історії в один список
            flat = []
            for row in xk_list:
                flat.extend(row)
            Xk = [flat]  # батч з одного рядка
    
            # 2) Прогноз виходів моделі
            yk = self.model.predict(Xk)[0]
    
            # 3) Додаємо компоненту вартості
            uk = u[k]
            prev_u = u_prev if k == 0 else u[k-1]
            cost_terms.append(self.objective.cost_term(yk, uk, prev_u))
    
            # 4) Оновлюємо FIFO-історію: видаляємо перший і додаємо новий
            feed_fe, ore_flow = d_seq[k]
            xk_list.pop(0)
            xk_list.append([feed_fe, ore_flow, uk])
    
        # Задача оптимізації
        problem = cp.Problem(cp.Minimize(cp.sum(cost_terms)), constraints)
        problem.solve(solver=cp.OSQP)
    
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC optimization failed: {problem.status}")
    
        return u.value