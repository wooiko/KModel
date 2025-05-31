# objectives.py

from abc import ABC, abstractmethod
import cvxpy as cp

class ControlObjective(ABC):
    """
    Базовий інтерфейс для цільових функцій MPC.
    Реалізації повинні повертати CVXPY-вираз вартості для одного кроку.
    """

    @abstractmethod
    def cost_term(self,
                  y_pred: list,
                  u_k: cp.Expression,
                  u_prev: cp.Expression) -> cp.Expression:
        """
        Формує терм цільової функції для одного кроку прогнозу.

        Args:
            y_pred: список [conc_fe, tail_fe, conc_mass, tail_mass] — CVXPY-вирази
            u_k:     змінна керування в кроці k (CVXPY Expression)
            u_prev:  змінна керування в кроці k-1 або скаляр (CVXPY Expression)

        Повертає:
            CVXPY Expression – вклад у сумарну вартість
        """
        raise NotImplementedError("Метод cost_term() треба реалізувати в підкласі.")


class MaxIronMassObjective(ControlObjective):
    """
    Сурогатна ціль: максимізувати масу заліза в концентраті
    через мінімізацію −(w_fe·conc_fe + w_mass·conc_mass) + λ·(u_k − u_{k−1})².
    """

    def __init__(self,
                 λ: float = 0.1,
                 w_fe: float = 1.0,
                 w_mass: float = 1.0):
        """
        Args:
            λ:      коефіцієнт штрафу за різкі зміни керування
            w_fe:   вага концентрації Fe
            w_mass: вага маси потоку
        """
        self.λ      = λ
        self.w_fe   = w_fe
        self.w_mass = w_mass

    def cost_term(self,
                  y_pred: list,
                  u_k: cp.Expression,
                  u_prev: cp.Expression) -> cp.Expression:
        # розпаковуємо прогнозовані виходи
        conc_fe, _, conc_mass, _ = y_pred

        # лінійний термін (мінімізуємо мінус від суми)
        linear_term    = - (self.w_fe * conc_fe + self.w_mass * conc_mass)
        # квадратичний штраф за зміну керування
        smoothing_term = self.λ * cp.square(u_k - u_prev)

        return linear_term + smoothing_term