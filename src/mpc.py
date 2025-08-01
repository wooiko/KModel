# mpc.py
from __future__ import annotations

import warnings
import numpy as np
import cvxpy as cp
# Змінено: KernelModel тепер єдине джерело моделей
from model import KernelModel, _BaseKernelModel
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod


class BaseMPC(ABC):
    """
    Абстрактний базовий клас для MPC контролерів.
    Визначає спільний інтерфейс для використання в симуляторі.
    """
    def __init__(self, model: _BaseKernelModel, objective, x_scaler: StandardScaler, y_scaler: StandardScaler,
                 n_targets: int, horizon: int, control_horizon: int, lag: int,
                 u_min: float, u_max: float, delta_u_max: float,
                 use_disturbance_estimator: bool, **kwargs):
        self.model = model
        self.objective = objective
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.L = lag
        self.n_inputs = (lag + 1) * 3
        self.n_targets = n_targets
        self.Np = horizon
        self.Nc = control_horizon or horizon

        self.u_min, self.u_max = u_min, u_max
        self.delta_u_max = delta_u_max

        self.use_disturbance_estimator = use_disturbance_estimator
        self.d_hat = np.zeros(self.n_targets)
        self.x_hist: np.ndarray | None = None

    @abstractmethod
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, **kwargs):
        """Навчає внутрішню модель контролера."""
        pass

    @abstractmethod
    def optimize(self, d_seq: np.ndarray, u_prev: float) -> np.ndarray:
        """Виконує крок оптимізації та повертає послідовність керування."""
        pass

    def reset_history(self, initial_history: np.ndarray):
        expected = (self.L + 1, 3)
        if initial_history.shape != expected:
            raise ValueError(f"initial_history має форму {expected}, отримано {initial_history.shape}")
        self.x_hist = initial_history.copy()


class KMPCController(BaseMPC):
    def __init__(self,
                 model: KernelModel, # Явно вказуємо тип
                 objective,
                 x_scaler: StandardScaler,
                 y_scaler: StandardScaler,
                 n_targets: int,
                 horizon: int = 6,
                 control_horizon: int = None,
                 lag: int = 2,
                 u_min: float = 25.0,
                 u_max: float = 35.0,
                 delta_u_max: float = 1.0,
                 use_disturbance_estimator: bool = True,
                 y_max: list = None,
                 y_min: list = None,
                 rho_y: float = 1e6,
                 rho_delta_u: float = 1e4,
                 rho_trust: float = 0.1
                ):
        # Використовуємо super() для ініціалізації базового класу
        super().__init__(
            model=model, objective=objective, x_scaler=x_scaler, y_scaler=y_scaler,
            n_targets=n_targets, horizon=horizon, control_horizon=control_horizon, lag=lag,
            u_min=u_min, u_max=u_max, delta_u_max=delta_u_max,
            use_disturbance_estimator=use_disturbance_estimator
        )

        if self.Nc > self.Np:
            raise ValueError("control_horizon (Nc) не може бути більше за horizon (Np)")

        self.y_max = np.array(y_max) if y_max is not None else None
        self.y_min = np.array(y_min) if y_min is not None else None
        self.rho_y = rho_y
        self.rho_delta_u = rho_delta_u
        self.rho_trust = rho_trust

        self.problem = None
        self.variables = {}
        self.parameters = {}

        self._setup_optimization_problem()

        warnings.filterwarnings("ignore",
            message="The problem includes expressions that don't support CPP backend.",
            category=UserWarning,
            module="cvxpy.reductions.solvers.solving_chain_utils")

    def _setup_optimization_problem(self):
        # ... (код без змін)
        # 1. Змінні оптимізації
        u_var = cp.Variable(self.Nc, name="u_seq")
        # Змінено: окремі змінні для верхніх та нижніх порушень дельти u
        eps_delta_u_upper = cp.Variable(self.Nc, nonneg=True, name="eps_delta_u_upper") #
        eps_delta_u_lower = cp.Variable(self.Nc, nonneg=True, name="eps_delta_u_lower") #
        eps_y_upper = cp.Variable((self.Np, 2), nonneg=True, name="eps_y_upper") if self.y_max is not None else None
        eps_y_lower = cp.Variable((self.Np, 2), nonneg=True, name="eps_y_lower") if self.y_min is not None else None
        self.variables = {
            'u': u_var,
            'eps_delta_u_upper': eps_delta_u_upper, #
            'eps_delta_u_lower': eps_delta_u_lower, #
            'eps_y_upper': eps_y_upper, 'eps_y_lower': eps_y_lower
        }

        # 2. Параметри, що будуть оновлюватись на кожному кроці
        W_param = cp.Parameter((self.n_inputs, self.n_targets), name="W_local")
        b_param = cp.Parameter(self.n_targets, name="b_local")
        x_hist_param = cp.Parameter(self.n_inputs, name="x_hist_flat")
        u_prev_param = cp.Parameter(name="u_prev")
        d_hat_param = cp.Parameter(self.n_targets, name="d_hat")
        d_seq_param = cp.Parameter((self.Np, 2), name="d_seq")
        x0_scaled_param = cp.Parameter(self.n_inputs, name="x0_scaled") # Для регіону довіри
        self.parameters = {
            'W': W_param, 'b': b_param, 'x_hist': x_hist_param,
            'u_prev': u_prev_param, 'd_hat': d_hat_param, 'd_seq': d_seq_param,
            'x0_scaled': x0_scaled_param
        }

        # Константи для масштабування
        mean_c = cp.Constant(self.x_scaler.mean_)
        scale_c = cp.Constant(self.x_scaler.scale_)

        # 3. Побудова прогнозу на горизонті Np
        pred_fe, pred_mass = [], []
        trust_region_cost = 0

        # Початковий стан з параметра
        xk_unscaled_list = [x_hist_param[i*3:(i+1)*3] for i in range(self.L + 1)]

        for k in range(self.Np):
            uk = u_var[k] if k < self.Nc else u_var[self.Nc - 1]

            # Формуємо вектор стану Xk в ОРИГІНАЛЬНОМУ масштабі
            Xk_unscaled = cp.hstack(xk_unscaled_list)

            # КОРЕКТНЕ МАСШТАБУВАННЯ ВЕКТОРА СТАНУ
            Xk_scaled = (Xk_unscaled - mean_c) / scale_c

            # Прогноз за локальною лінійною моделлю
            yk = Xk_scaled @ W_param + b_param + d_hat_param
            pred_fe.append(yk[0])
            pred_mass.append(yk[1]) # Індекс 1 для concentrate_mass_flow, оскільки n_targets = 2

            # ШТРАФ РЕГІОНУ ДОВІРИ (TRUST REGION)
            # Штрафуємо за відхилення прогнозованого стану від точки лінеаризації ТІЛЬКИ на першому кроці
            if k == 0 and self.model.kernel != 'linear': #
                 trust_region_cost += self.rho_trust * cp.sum_squares(Xk_scaled - x0_scaled_param) #

            # Оновлюємо стан для наступного кроку
            feed_fe, ore_flow = d_seq_param[k, 0], d_seq_param[k, 1]
            xk_unscaled_list.pop(0)
            xk_unscaled_list.append(cp.hstack([feed_fe, ore_flow, uk]))

        conc_fe_preds = cp.hstack(pred_fe)
        conc_mass_preds = cp.hstack(pred_mass)

        # 4. Формування обмежень
        cons = [u_var >= self.u_min, u_var <= self.u_max]
        du0 = u_var[0] - u_prev_param
        du_rest = u_var[1:] - u_var[:-1] if self.Nc > 1 else []
        if self.Nc > 1:
            Du_ext = cp.hstack([du0, du_rest])
        else:
            Du_ext = cp.hstack([du0])

        # Обмеження на Du_ext з м'якими змінними (виправлено)
        cons.extend([
            Du_ext <= self.delta_u_max + eps_delta_u_upper, #
            Du_ext >= -self.delta_u_max - eps_delta_u_lower #
        ])

        y_preds_stacked = cp.vstack([conc_fe_preds, conc_mass_preds]).T
        if eps_y_upper is not None:
            cons.append(y_preds_stacked <= self.y_max + eps_y_upper)
        if eps_y_lower is not None:
            cons.append(y_preds_stacked >= self.y_min - eps_y_lower)

        # 5. Формування цільової функції
        base_cost = self.objective.cost_full(
            conc_fe_preds=conc_fe_preds, conc_mass_preds=conc_mass_preds,
            u_seq=u_var, u_prev=u_prev_param
        )
        # Штраф за порушення Du (виправлено)
        penalty_cost = self.rho_delta_u * (cp.sum_squares(eps_delta_u_upper) + cp.sum_squares(eps_delta_u_lower)) #
        if eps_y_upper is not None:
            penalty_cost += self.rho_y * cp.sum_squares(eps_y_upper)
        if eps_y_lower is not None:
            penalty_cost += self.rho_y * cp.sum_squares(eps_y_lower)

        total_cost = base_cost + penalty_cost + trust_region_cost

        # 6. Створення об'єкту задачі
        self.problem = cp.Problem(cp.Minimize(total_cost), cons)


    def optimize(self, d_seq: np.ndarray, u_prev: float) -> np.ndarray:
        # ... (код без змін)
        if self.x_hist is None:
            raise RuntimeError("Історія стану не ініціалізована. Викличте MPCController.reset_history().")
        if self.problem is None:
            raise RuntimeError("Задача оптимізації не була налаштована. Перевірте конструктор.")

        # 1. Лінеаризація моделі в поточній робочій точці
        X0_current_unscaled = self.x_hist.flatten().reshape(1, -1)
        X0_current_scaled = self.x_scaler.transform(X0_current_unscaled)
        W_local, b_local = self.model.linearize(X0_current_scaled)

        # 2. Оновлення значень параметрів у задачі CVXPY
        self.parameters['W'].value = W_local
        self.parameters['b'].value = b_local
        self.parameters['x_hist'].value = X0_current_unscaled.flatten()
        self.parameters['u_prev'].value = u_prev
        self.parameters['d_seq'].value = d_seq
        self.parameters['x0_scaled'].value = X0_current_scaled.flatten()

        d_hat_val = self.d_hat if self.use_disturbance_estimator and self.d_hat is not None else np.zeros(self.n_targets)
        self.parameters['d_hat'].value = d_hat_val

        # 3. Розв'язання задачі
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True)
        except cp.error.SolverError:
            print("ПОПЕРЕДЖЕННЯ: Помилка солвера. Використовується попереднє керування.")
            return np.array([u_prev] * self.Nc)

        # 4. Діагностика та повернення результату
        if self.problem.status not in ["infeasible", "unbounded"]:
            u_optimal = self.variables['u'].value
            # Діагностичний вивід (виправлено для двох eps змінних)
            if (self.variables['eps_y_upper'] is not None and
                np.any(self.variables['eps_y_upper'].value > 1e-4)):
                print(f"  -> УВАГА: Порушено верхнє обмеження Y! ε_y_upper = {np.round(self.variables['eps_y_upper'].value.flatten(), 3)}")

            if (self.variables['eps_delta_u_upper'] is not None and np.any(self.variables['eps_delta_u_upper'].value > 1e-4)) or \
               (self.variables['eps_delta_u_lower'] is not None and np.any(self.variables['eps_delta_u_lower'].value > 1e-4)):
                print(f"  -> УВАГА: Порушено обмеження Δu! ε_Δu_upper = {np.round(self.variables['eps_delta_u_upper'].value, 3)}, ε_Δu_lower = {np.round(self.variables['eps_delta_u_lower'].value, 3)}") #
        else:
            print(f"ПОПЕРЕДЖЕННЯ: Задача оптимізації не має розв'язку (статус: {self.problem.status}). Використовується попереднє керування.")
            return np.array([u_prev] * self.Nc)

        return u_optimal


    def fit(self,
            X_train: np.ndarray,
            Y_train: np.ndarray,
            **kwargs):
        """
        Навчає KernelModel. Дані мають бути вже масштабовані.
        """
        self.model.fit(X_train, Y_train)


# ВИДАЛЕНО КЛАС LinPredictor


# ======================================================================
#                   ОНОВЛЕНИЙ LMPC CONTROLLER
# ======================================================================
class LMPCController(BaseMPC):
    def __init__(self, **kwargs):
        # Тепер LMPC приймає `model` як і KMPC.
        # Всі параметри передаються в базовий клас.
        super().__init__(**kwargs)

        self.problem: cp.Problem | None = None
        self._vars: dict[str, cp.Variable] = {}
        self._pars: dict[str, cp.Parameter] = {}

        # Параметри для м'яких обмежень, як у K-MPC, для сумісності
        self.rho_y = kwargs.get('rho_y', 1e6)
        self.rho_delta_u = kwargs.get('rho_delta_u', 1e4)
        self.y_max = kwargs.get('y_max')
        self.y_min = kwargs.get('y_min')

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, **kwargs):
        """
        Навчає внутрішню лінійну модель та налаштовує QP-задачу.
        Дані мають бути вже масштабовані.
        """
        # 1. Просто навчаємо модель (вся логіка тепер в model.py)
        self.model.fit(X_train, Y_train)

        # 2. Налаштовуємо QP-задачу один раз після навчання
        self._setup_optim_problem()

    def _setup_optim_problem(self):
        """Створює QP-задачу з фіксованою лінійною моделлю."""
        # 1. Змінні оптимізації
        u_var = cp.Variable(self.Nc, name="u")
        eps_delta_u_upper = cp.Variable(self.Nc, nonneg=True, name="eps_delta_u_upper")
        eps_delta_u_lower = cp.Variable(self.Nc, nonneg=True, name="eps_delta_u_lower")
        eps_y_upper = cp.Variable((self.Np, self.n_targets), nonneg=True) if self.y_max is not None else None
        eps_y_lower = cp.Variable((self.Np, self.n_targets), nonneg=True) if self.y_min is not None else None
        self._vars = {
            'u': u_var, 'eps_delta_u_upper': eps_delta_u_upper,
            'eps_delta_u_lower': eps_delta_u_lower,
            'eps_y_upper': eps_y_upper, 'eps_y_lower': eps_y_lower
        }

        # 2. Параметри, що оновлюються на кожному кроці
        x_hist_param = cp.Parameter(self.n_inputs, name="x_hist_flat")
        u_prev_param = cp.Parameter(name="u_prev")
        d_hat_param = cp.Parameter(self.n_targets, name="d_hat")
        d_seq_param = cp.Parameter((self.Np, 2), name="d_seq")
        self._pars = {
            'x_hist': x_hist_param, 'u_prev': u_prev_param,
            'd_hat': d_hat_param, 'd_seq': d_seq_param
        }

        # 3. Константи моделі та масштабування (W та b тепер константи!)
        W_c = cp.Constant(self.model.W)
        b_c = cp.Constant(self.model.b)
        mean_c = cp.Constant(self.x_scaler.mean_)
        scale_c = cp.Constant(self.x_scaler.scale_)

        # 4. Побудова прогнозу
        preds = []
        xk_unscaled_list = [x_hist_param[i*3:(i+1)*3] for i in range(self.L + 1)]

        for k in range(self.Np):
            uk = u_var[k] if k < self.Nc else u_var[self.Nc - 1]
            Xk_unscaled = cp.hstack(xk_unscaled_list)
            Xk_scaled = (Xk_unscaled - mean_c) / scale_c
            yk = Xk_scaled @ W_c + b_c + d_hat_param
            preds.append(yk)

            feed_fe, ore_flow = d_seq_param[k, 0], d_seq_param[k, 1]
            xk_unscaled_list.pop(0)
            xk_unscaled_list.append(cp.hstack([feed_fe, ore_flow, uk]))

        y_preds_stacked = cp.vstack(preds)
        conc_fe_preds = y_preds_stacked[:, 0]
        conc_mass_preds = y_preds_stacked[:, 1]

        # 5. Формування обмежень
        cons = [u_var >= self.u_min, u_var <= self.u_max]
        du0 = u_var[0] - u_prev_param
        du_rest = u_var[1:] - u_var[:-1] if self.Nc > 1 else []
        Du_ext = cp.hstack([du0] + ([du_rest] if self.Nc > 1 else []))

        cons.extend([
            Du_ext <= self.delta_u_max + eps_delta_u_upper,
            Du_ext >= -self.delta_u_max - eps_delta_u_lower
        ])

        if eps_y_upper is not None:
            cons.append(y_preds_stacked <= self.y_max + eps_y_upper)
        if eps_y_lower is not None:
            cons.append(y_preds_stacked >= self.y_min - eps_y_lower)

        # 6. Формування цільової функції
        base_cost = self.objective.cost_full(
            conc_fe_preds=conc_fe_preds, conc_mass_preds=conc_mass_preds,
            u_seq=u_var, u_prev=u_prev_param
        )
        penalty_cost = self.rho_delta_u * (cp.sum_squares(eps_delta_u_upper) + cp.sum_squares(eps_delta_u_lower))
        if eps_y_upper is not None: penalty_cost += self.rho_y * cp.sum_squares(eps_y_upper)
        if eps_y_lower is not None: penalty_cost += self.rho_y * cp.sum_squares(eps_y_lower)

        total_cost = base_cost + penalty_cost
        self.problem = cp.Problem(cp.Minimize(total_cost), cons)


    def optimize(self, d_seq: np.ndarray, u_prev: float) -> np.ndarray:
        if self.problem is None:
            raise RuntimeError("Метод fit() має бути викликаний перед optimize()")

        self._pars['x_hist'].value = self.x_hist.flatten()
        self._pars['u_prev'].value = u_prev
        self._pars['d_seq'].value = d_seq
        self._pars['d_hat'].value = self.d_hat if self.use_disturbance_estimator and self.d_hat is not None else np.zeros(self.n_targets)

        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True)
        except cp.error.SolverError:
            print("ПОПЕРЕДЖЕННЯ: Помилка солвера. Використовується попереднє керування.")
            return np.array([u_prev] * self.Nc)

        if self.problem.status in ["infeasible", "unbounded"]:
            print(f"ПОПЕРЕДЖЕННЯ: Задача оптимізації не має розв'язку (статус: {self.problem.status}). Використовується попереднє керування.")
            return np.array([u_prev] * self.Nc)

        u_optimal = self._vars['u'].value
        # (Опціонально) Діагностика порушень, як у KMPC
        return u_optimal