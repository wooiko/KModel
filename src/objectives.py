# objectives.py

import numpy as np

from abc import ABC, abstractmethod
import cvxpy as cp

class ControlObjective(ABC):
    """
    Базовий клас для цільових функцій MPC.
    Старий метод cost_term лишається для зворотної сумісності, 
    але тепер він просто викликає cost_full або кидає помилку.
    """

    @abstractmethod
    def cost_full(self,
                  conc_fe_preds: cp.Expression,
                  conc_mass_preds: cp.Expression,
                  u_seq: cp.Expression,
                  u_prev: cp.Expression) -> cp.Expression:
        """
        Обов’язково треба реалізувати:
        векторизована ціль на весь горизонт.
        """

    def cost_term(self,
                  y_pred: list,
                  u_k: cp.Expression,
                  u_prev: cp.Expression) -> cp.Expression:
        """
        Депрекейтед: для зворотньої сумісності, 
        але якщо викликають – кидаємо помилку або транслюємо в cost_full.
        """
        raise NotImplementedError(
            "Метод cost_term застарів, використовуйте cost_full(conc_fe_preds, conc_mass_preds, u_seq, u_prev)"
        )


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
    
class MaxIronMassTrackingObjective(ControlObjective):
    """
    Ціль з квадратичним трекінгом, гладкістю Δu та інтегральним штрафом.
    """

    def __init__(self,
                 ref_fe: float,
                 ref_mass: float,
                 w_fe: float = 1.0,
                 w_mass: float = 1.0,
                 λ: float = 0.1,
                 K_I: float = 0.01):
        super().__init__()
        self.ref_fe   = ref_fe
        self.ref_mass = ref_mass
        self.w_fe     = w_fe
        self.w_mass   = w_mass
        self.λ        = λ
        self.K_I      = K_I

    def cost_full(self,
                  conc_fe_preds: cp.Expression,
                  conc_mass_preds: cp.Expression,
                  u_seq: cp.Expression,
                  u_prev: cp.Expression) -> cp.Expression:
        Np = conc_fe_preds.shape[0]
        Nc = u_seq.shape[0]

        # 1) помилки
        err_fe   = conc_fe_preds   - self.ref_fe    # (Np,)
        err_mass = conc_mass_preds - self.ref_mass  # (Np,)

        # 2) tracking у вигляді квадратичної форми з Q_full
        # — збираємо в один вектор довжини 2*Np
        err_stack = cp.hstack([err_fe, err_mass])   # (2*Np,)

        # — вагова матриця для одного кроку
        Q_stage = np.diag([self.w_fe, self.w_mass])        # (2×2)
        # — Q_full = kron(I_Np, Q_stage)
        Q_full  = np.kron(np.eye(Np), Q_stage)             # (2Np×2Np)

        cost_track = cp.quad_form(err_stack, Q_full)

        # 3) smoothness: Δu
        du0     = u_seq[0] - u_prev                        # scalar
        du_rest = u_seq[1:] - u_seq[:-1]                   # (Nc-1,)
        Du_ext  = cp.hstack([du0, du_rest])                # (Nc,)

        R_full      = self.λ * np.eye(Nc)                  # (Nc×Nc)
        cost_smooth = cp.quad_form(Du_ext, R_full)

        # 4) інтегральний штраф
        L     = np.tril(np.ones((Np, Np)))
        Q_int = L.T @ L                                    # (Np×Np)
        cost_int = self.K_I * (
            cp.quad_form(err_fe,   Q_int) +
            cp.quad_form(err_mass, Q_int)
        )

        return cost_track + cost_smooth + cost_int
