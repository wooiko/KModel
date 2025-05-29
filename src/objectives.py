# objectives.py

from abc import ABC, abstractmethod
import cvxpy as cp

class ControlObjective(ABC):
    """
    Базовий інтерфейс для цільових функцій MPC.
    Реалізації повинні повертати вираз вартості (CVXPY-вираз) для одного кроку.
    """

    @abstractmethod
    def cost_term(self,
                  y_pred: list,
                  u_k: cp.Expression,
                  u_prev: cp.Expression) -> cp.Expression:
        """
        Формує терм цільової функції для одного кроку прогнозу.

        Args:
            y_pred: список [conc_fe, tail_fe, conc_mass, tail_mass]
            u_k:     змінна керування в кроці k (CVXPY Expression)
            u_prev:  змінна керування в кроці k-1 або скаляр (CVXPY Expression)

        Повертає:
            CVXPY Expression – вклад у сумарну вартість
        """
        pass


class MaxIronMassObjective(ControlObjective):
    """
    Ціль: максимізувати масу заліза в концентраті
          мінімізуючи −(conc_fe⋅conc_mass)
    із штрафом за різкі зміни керування λ·(u_k − u_{k−1})².
    """

    def __init__(self, λ: float = 0.1):
        """
        Args:
            λ: коефіцієнт штрафу за зміну керування
        """
        self.λ = λ

    def cost_term(self,
                  y_pred: list,
                  u_k: cp.Expression,
                  u_prev: cp.Expression) -> cp.Expression:
        conc_fe, _, conc_mass, _ = y_pred
        # −(концентрація Fe * маса) + λ·(Δu)²
        iron_mass_term = - conc_fe * conc_mass
        smoothing_term  = self.λ * cp.square(u_k - u_prev)
        return iron_mass_term + smoothing_term