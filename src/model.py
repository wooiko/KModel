# model.py

import inspect
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

import numpy as np
from scipy.stats import loguniform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Product,
    RBF,
    Sum,
    WhiteKernel,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from data_gen import StatefulDataGenerator


# ======================================================================
#                        БАЗОВА СТРАТЕГІЯ
# ======================================================================
class _BaseKernelModel(ABC):
    def __init__(self):
        self._kernel: str | None = None

    # obов’язкові API-методи
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def linearize(self, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

    # kernel з можливістю set
    @property
    def kernel(self) -> str | None:  # noqa: D401
        return self._kernel

    @kernel.setter
    def kernel(self, value: str) -> None:
        self._kernel = value.lower() if value else None


# ======================================================================
#                    KRR (Kernel Ridge Regression)
# ======================================================================
class _KRRModel(_BaseKernelModel):
    def __init__(
        self,
        kernel: str = "linear",
        alpha: float = 1.0,
        gamma: float | None = None,
        find_optimal_params: bool = False,
        n_iter_random_search: int = 20,
    ):
        super().__init__()
        self.kernel = kernel
        if self.kernel not in ("linear", "rbf"):
            raise ValueError(
                f"Невідоме ядро '{self.kernel}' для KRR. Підтримуються 'linear', 'rbf'."
            )

        self.alpha = alpha
        self.gamma = gamma
        self.find_optimal_params = find_optimal_params
        self.n_iter_random_search = n_iter_random_search

        self.model: KernelRidge | None = None
        self.X_train_: np.ndarray | None = None
        self.dual_coef_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray, config_params: dict = None) -> None:
        """
        Навчання KRR моделі.
        
        Args:
            X: Тренувальні ознаки
            Y: Тренувальні цілі
            config_params: Конфігураційні параметри (не використовуються KRR, але потрібні для сумісності)
        """
        # Вся існуюча логіка методу залишається без змін
        if self.find_optimal_params:
            self.model = self._run_random_search(X, Y)
        else:
            gamma_eff = (
                self.gamma
                if self.gamma is not None
                else (
                    self._median_gamma(X)
                    if self.kernel == "rbf"
                    else None
                )
            )
            self.model = KernelRidge(alpha=self.alpha, kernel=self.kernel, gamma=gamma_eff)
            self.model.fit(X, Y)
    
        self.X_train_ = X.copy()
        self.dual_coef_ = self.model.dual_coef_
        if self.kernel == "linear":
            self.coef_ = X.T @ self.dual_coef_
            self.intercept_ = np.zeros(Y.shape[1])
        else:
            self.coef_ = None
            self.intercept_ = np.zeros(Y.shape[1])
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Модель KRR не навчена.")
        return self.model.predict(X)

    def linearize(self, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X0.ndim == 1:
            X0 = X0.reshape(1, -1)

        if self.kernel == "linear":
            return self.coef_, self.intercept_

        gamma_eff = getattr(self.model, "gamma", None) or self._median_gamma(self.X_train_)
        diffs = X0[:, None, :] - self.X_train_[None, :, :]  # shape (1,n,d)
        sq = np.sum(diffs**2, axis=-1)
        K_row = np.exp(-gamma_eff * sq)
        dK_dX = -2 * gamma_eff * diffs * K_row[..., None]  # (1,n,d)

        W = np.einsum("ijk,ji->ki", dK_dX, self.dual_coef_)  # (d,m)
        b = (self.predict(X0) - X0 @ W).flatten()
        return np.clip(W, -1e3, 1e3), np.clip(b, -1e3, 1e3)

    # ------------------------------------------------------------------
    def _run_random_search(self, X, Y) -> KernelRidge:
        base = KernelRidge(kernel=self.kernel)
        if self.kernel == "linear":
            param_dist = {"alpha": loguniform(1e-3, 1e2)}
        else:
            param_dist = {
                "alpha": loguniform(1e-2, 1e2),
                "gamma": loguniform(1e-3, 1e1),
            }

        rs = RandomizedSearchCV(
            base,
            param_dist,
            n_iter=self.n_iter_random_search,
            cv=3,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        rs.fit(X, Y)
        return rs.best_estimator_

    @staticmethod
    def _median_gamma(X: np.ndarray) -> float:
        subset = X if X.shape[0] <= 1000 else X[np.random.choice(X.shape[0], 1000, False)]
        d2 = np.sum((subset[:, None] - subset[None]) ** 2, axis=-1)
        med = np.median(d2[np.triu_indices_from(d2, 1)]) or 1.0
        return 1.0 / med


# ======================================================================
#                Gaussian Process Regression
# ======================================================================
class _GPRModel(_BaseKernelModel):
    def __init__(self):
        super().__init__()
        self.models: list[GaussianProcessRegressor] = []

    def fit(self, X: np.ndarray, Y: np.ndarray, config_params: dict = None) -> None:
        """
        Навчання GPR моделі.
        
        Args:
            X: Тренувальні ознаки
            Y: Тренувальні цілі
            config_params: Конфігураційні параметри (не використовуються GPR, але потрібні для сумісності)
        """
        # Вся існуюча логіка методу залишається без змін
        k_base = (
            C(1.0, (1e-3, 1e3))
            * RBF(1.0, (0.1, 20.0))
            + WhiteKernel(0.1, (1e-5, 1e1))
        )
        self.models = []
        for i in range(Y.shape[1]):
            gpr = GaussianProcessRegressor(
                kernel=k_base,
                alpha=0,
                normalize_y=True,
                n_restarts_optimizer=2,
            )
            gpr.fit(X, Y[:, i])
            self.models.append(gpr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([m.predict(X) for m in self.models]).T

    def linearize(self, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X0.ndim == 1:
            X0 = X0[None, :]
        W_cols, b_cols = [], []
        for gpr in self.models:
            rbf = self._find_rbf(gpr.kernel_)
            if rbf is None:
                raise RuntimeError("RBF-компонент ядра не знайдено.")
            gamma = 1.0 / (2 * rbf.length_scale**2)
            Xtr, alpha = gpr.X_train_, gpr.alpha_

            diffs = X0[:, None] - Xtr[None]
            sq = np.sum(diffs**2, axis=-1)
            K_row = np.exp(-gamma * sq)
            dK = -2 * gamma * diffs * K_row[..., None]

            W = (dK.squeeze(0).T @ alpha).reshape(-1, 1)
            y0 = gpr.predict(X0)
            b = (y0 - X0 @ W).flatten()
            W_cols.append(W)
            b_cols.append(b)
        return np.hstack(W_cols), np.hstack(b_cols)

    # ------------------------------------------------------------------
    @staticmethod
    def _find_rbf(kernel):
        if isinstance(kernel, RBF):
            return kernel
        if isinstance(kernel, (Product, Sum)):
            return _GPRModel._find_rbf(kernel.k1) or _GPRModel._find_rbf(kernel.k2)
        return None


# ======================================================================
#                   НОВА МОДЕЛЬ – SVR (Support-Vector Regression)
# ======================================================================
class _SVRModel(_BaseKernelModel):
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 100.0,        # ✅ Збільшено для кращої точності
        epsilon: float = 0.01,   # ✅ Зменшено для меншої толерантності
        gamma: float | None = None,
        degree: int = 3,
        find_optimal_params: bool = False,
        n_iter_random_search: int = 30,
    ):
        super().__init__()
        self.kernel = kernel.lower()
        if self.kernel not in ("linear", "rbf", "poly"):
            raise ValueError("SVR підтримує лише 'linear', 'rbf', 'poly'.")

        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.find_optimal_params = find_optimal_params
        self.n_iter_random_search = n_iter_random_search
        self.models: list[SVR] = []

    def fit(self, X: np.ndarray, Y: np.ndarray, config_params: dict = None) -> None:
        """
        Навчання SVR моделі.
        
        Args:
            X: Тренувальні ознаки
            Y: Тренувальні цілі
            config_params: Конфігураційні параметри (не використовуються SVR, але потрібні для сумісності)
        """
        # Вся існуюча логіка методу залишається без змін
        n_targets = Y.shape[1]
        self.models.clear()
        self.X_train_ = X.copy()
    
        for k in range(n_targets):
            y = Y[:, k]
    
            if self.find_optimal_params:
                mdl = self._run_random_search(X, y)
            else:
                # Всі існуючі обчислення gamma_eff та створення моделі
                if self.kernel == "rbf":
                    if self.gamma is not None:
                        gamma_eff = self.gamma
                    else:
                        gamma_eff = 1.0 / (X.shape[1] * X.var())
                else:
                    gamma_eff = self.gamma
    
                mdl = SVR(
                    kernel=self.kernel,
                    C=self.C,
                    epsilon=self.epsilon,
                    gamma=gamma_eff,
                    degree=self.degree,
                )
                mdl.fit(X, y)
                mdl._actual_gamma = gamma_eff if gamma_eff is not None else 'scale'
    
            self.models.append(mdl)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("SVRModel не навчена.")
        preds = [m.predict(X) for m in self.models]
        return np.vstack(preds).T

    def linearize(self, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X0.ndim == 1:
            X0 = X0[None, :]
    
        W_cols, b_cols = [], []
        for mdl in self.models:
            if self.kernel == "linear":
                W = mdl.coef_.reshape(-1, 1)
                b = mdl.intercept_.copy()
                
            elif self.kernel == "rbf":
                # ✅ ВИПРАВЛЕНА логіка gamma в linearize()
                if hasattr(mdl, '_actual_gamma') and isinstance(mdl._actual_gamma, float):
                    gamma_eff = mdl._actual_gamma
                else:
                    # ✅ ПРАВИЛЬНА sklearn формула
                    gamma_eff = 1.0 / (self.X_train_.shape[1] * self.X_train_.var())
                
                sv = mdl.support_vectors_
                coef = mdl.dual_coef_.ravel()
    
                # ✅ ВИПРАВЛЕНЕ обчислення градієнтів:
                diffs = X0[:, None, :] - sv[None, :, :]  # (1, n_sv, n_features)
                sq = np.sum(diffs**2, axis=-1)           # (1, n_sv)
                K_row = np.exp(-gamma_eff * sq)          # (1, n_sv)
                dK = -2 * gamma_eff * diffs * K_row[..., None]  # (1, n_sv, n_features)
    
                # ✅ БЕЗПЕЧНЕ витягування градієнтів:
                if dK.shape[0] == 1:
                    dK_2d = dK[0]  # (n_sv, n_features) 
                else:
                    dK_2d = dK.squeeze(0)  # Backup
                    
                W = (dK_2d.T @ coef).reshape(-1, 1)  # (n_features, 1)
                y0 = mdl.predict(X0)
                b = (y0 - X0 @ W).flatten()
                
            else:
                raise NotImplementedError(
                    "Лінеаризація SVR підтримується лише для 'linear' та 'rbf'."
                )
    
            W_cols.append(np.clip(W, -1e3, 1e3))
            b_cols.append(np.clip(b, -1e3, 1e3))
    
        W_local = np.hstack(W_cols)
        b_local = np.hstack(b_cols)
        return W_local, b_local

    def _run_random_search(self, X, y) -> SVR:
        """Оптимізована версія з швидшою обробкою linear kernel"""
        
        # 🚀 ШВИДКА ЛОГІКА ДЛЯ LINEAR KERNEL
        if self.kernel == "linear":
            # Linear kernel не потребує складної оптимізації
            param_dist = {
                "C": [1.0, 10.0, 100.0],           # Дискретні значення
                "epsilon": [0.001, 0.01, 0.1]      # Типові значення
            }
            n_iter = 9  # 3x3 = всі комбінації
            cv_folds = 2  # Менше фолдів
            
            print(f"🚀 SVR Linear: швидка оптимізація {n_iter} комбінацій...")
            
        elif self.kernel == "rbf":
            # Повна оптимізація для RBF
            param_dist = {
                "C": loguniform(10, 1000),      
                "epsilon": loguniform(1e-3, 0.1),
                "gamma": loguniform(1e-4, 1e-1)
            }
            n_iter = self.n_iter_random_search
            cv_folds = min(3, len(y) // 50)
            
            print(f"🔧 SVR RBF: повна оптимізація {n_iter} ітерацій...")
            
        elif self.kernel == "poly":
            # Обмежена оптимізація для poly
            param_dist = {
                "C": loguniform(10, 500),        # Менший діапазон
                "epsilon": loguniform(1e-3, 0.05),
                "gamma": loguniform(1e-4, 1e-2), # Менший діапазон
                "degree": [2, 3]                 # Тільки 2-3 степені
            }
            n_iter = min(20, self.n_iter_random_search)
            cv_folds = min(3, len(y) // 50)
            
            print(f"⚙️ SVR Poly: обмежена оптимізація {n_iter} ітерацій...")
        
        else:
            raise ValueError(f"Непідтримуваний kernel: {self.kernel}")
    
        base = SVR(kernel=self.kernel, degree=self.degree)
        
        # 🚀 ВИКОРИСТОВУЄМО GridSearchCV для linear (швидше)
        if self.kernel == "linear":
            from sklearn.model_selection import GridSearchCV
            rs = GridSearchCV(
                base,
                param_dist,  # Тут це буде dict з lists
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=1  # Показуємо прогрес
            )
        else:
            # RandomizedSearchCV для інших kernels
            rs = RandomizedSearchCV(
                base,
                param_dist,
                n_iter=n_iter,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                random_state=42,
                n_jobs=-1,
                verbose=1  # Показуємо прогрес
            )
        
        rs.fit(X, y)
        
        # ✅ Зберігаємо оптимальне gamma
        best_model = rs.best_estimator_
        if hasattr(best_model, 'gamma'):
            best_model._actual_gamma = best_model.gamma
        
        print(f"✅ Оптимальні параметри: {rs.best_params_}")
        
        return best_model

# ======================================================================
#                   LINEAR MODEL (для L-MPC)
# ======================================================================

class _LinearModel(_BaseKernelModel):
    """
    Лінійна модель для L-MPC з підтримкою різних регуляризацій та поліноміальних ознак
    """
    
    def __init__(
        self,
        linear_type: str = "ols",         # "ols", "ridge", "lasso" 
        alpha: float = 1.0,               # Коефіцієнт регуляризації для Ridge/Lasso
        poly_degree: int = 1,             # Степінь поліноміальних ознак (1=лінійна)
        include_bias: bool = True,        # Включати bias термин
        find_optimal_params: bool = False, # Пошук оптимальних параметрів
        n_iter_random_search: int = 20,
    ):
        super().__init__()
        self.linear_type = linear_type.lower()
        self.alpha = alpha
        self.poly_degree = poly_degree
        self.include_bias = include_bias
        self.find_optimal_params = find_optimal_params
        self.n_iter_random_search = n_iter_random_search
        
        # Валідація
        if self.linear_type not in ("ols", "ridge", "lasso"):
            raise ValueError("linear_type повинен бути 'ols', 'ridge' або 'lasso'")
        
        if self.poly_degree < 1 or self.poly_degree > 3:
            raise ValueError("poly_degree повинен бути від 1 до 3")
        
        # Внутрішні атрибути
        self.model: LinearRegression | Ridge | Lasso | None = None
        self.poly_features: PolynomialFeatures | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        
        # Для compatibility з kernel моделями
        self._kernel = "linear"  # Позначаємо як лінійну

    def fit(self, X: np.ndarray, Y: np.ndarray, config_params: dict = None) -> None:
        """
        Навчання лінійної моделі.
        
        Args:
            X: Тренувальні ознаки
            Y: Тренувальні цілі
            config_params: Конфігураційні параметри (не використовуються Linear, але потрібні для сумісності)
        """
        # Вся існуюча логіка методу залишається без змін
        print(f"🔧 Навчання Linear Model: {self.linear_type}, poly_degree={self.poly_degree}")
        
        # Створення поліноміальних ознак якщо потрібно
        if self.poly_degree > 1:
            self.poly_features = PolynomialFeatures(
                degree=self.poly_degree,
                include_bias=False
            )
            X_features = self.poly_features.fit_transform(X)
        else:
            self.poly_features = None
            X_features = X
            
        # Вибір та налаштування моделі
        if self.find_optimal_params:
            self.model = self._run_random_search(X_features, Y)
        else:
            if self.linear_type == "ols":
                self.model = LinearRegression(fit_intercept=self.include_bias)
            elif self.linear_type == "ridge":
                self.model = Ridge(alpha=self.alpha, fit_intercept=self.include_bias)
            elif self.linear_type == "lasso":
                self.model = Lasso(alpha=self.alpha, fit_intercept=self.include_bias, max_iter=2000)
        
        # Навчання
        self.model.fit(X_features, Y)
        
        # Збереження коефіцієнтів для швидкого доступу
        self.coef_ = self.model.coef_.T if Y.ndim > 1 else self.model.coef_.reshape(-1, 1)
        self.intercept_ = (
            self.model.intercept_ if hasattr(self.model, 'intercept_') 
            else np.zeros(Y.shape[1] if Y.ndim > 1 else 1)
        )
        
        print(f"✅ Linear Model навчена: {self.linear_type}, poly_degree={self.poly_degree}")
        print(f"   Коефіцієнти shape: {self.coef_.shape}, Intercept: {self.intercept_.shape}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Передбачення лінійної моделі"""
        if self.model is None:
            raise RuntimeError("Linear Model не навчена!")
            
        # Застосування поліноміальних ознак якщо потрібно
        if self.poly_features is not None:
            X_features = self.poly_features.transform(X)
        else:
            X_features = X
            
        return self.model.predict(X_features)

    def linearize(self, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Лінеаризація для лінійної моделі
        
        Returns:
            W: (n_features, n_outputs) - градієнт матриця (як у K-MPC)
            b: (n_outputs,) - зміщення вектор
        """
        if X0.ndim == 1:
            X0 = X0.reshape(1, -1)
            
        if self.coef_ is None:
            raise RuntimeError("Модель не навчена!")
            
        # 🎯 ЛІНІЙНИЙ ВИПАДОК (poly_degree = 1)
        if self.poly_degree == 1:
            # sklearn LinearRegression: coef_ = (n_features, n_outputs)
            W = self.coef_  # ✅ Залишаємо (n_features, n_outputs) як у K-MPC
            b = self.intercept_  # (n_outputs,)
            return W, b
            
        # 🎯 ПОЛІНОМІАЛЬНИЙ ВИПАДОК (poly_degree > 1)
        else:
            # ✅ ВИПРАВЛЕНО: використовуємо include_bias=False
            grad_poly = self._compute_polynomial_gradient(X0)  # (n_samples, n_features, n_poly_features_no_bias)
            
            # Перевіряємо розмірності
            n_samples, n_features, n_poly_features = grad_poly.shape
            n_coef_features, n_outputs = self.coef_.shape
            
            if n_poly_features != n_coef_features:
                raise ValueError(f"Розмірність градієнта {n_poly_features} != розмірність коефіцієнтів {n_coef_features}")
            
            # W = градієнт * коефіцієнти
            W = np.einsum('ijk,kl->ijl', grad_poly, self.coef_)  # (n_samples, n_features, n_outputs)
            
            # Беремо перший семпл: (n_features, n_outputs) - як у K-MPC
            W_local = W[0]  # ✅ (n_features, n_outputs)
            
            # ✅ ПРАВИЛЬНЕ МНОЖЕННЯ: X @ W (як у K-MPC)
            y0 = self.predict(X0)
            b_local = y0[0] - X0[0] @ W_local
            
            return W_local, b_local

    def _compute_polynomial_gradient(self, X0: np.ndarray) -> np.ndarray:
        """
        Обчислює градієнт поліноміальних ознак в точці X0
        
        Returns:
            grad_poly: (n_samples, n_features, n_poly_features) - градієнт матриця
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        n_samples, n_features = X0.shape
        
        # ✅ ВИПРАВЛЕНО: include_bias=False, щоб відповідати sklearn coef_
        poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
        
        # Фітуємо на dummy даних щоб отримати powers_
        dummy_X = np.ones((1, n_features))
        poly.fit(dummy_X)
        
        # Отримуємо кількість ознак БЕЗ bias
        n_poly_features = len(poly.powers_)
        
        # Ініціалізуємо градієнт
        grad_poly = np.zeros((n_samples, n_features, n_poly_features))
        
        # Обчислюємо градієнт для кожної поліноміальної ознаки
        for i, powers in enumerate(poly.powers_):
            # powers - масив степенів для кожної оригінальної ознаки
            for j in range(n_features):
                if powers[j] > 0:
                    # ∂(x₁^p₁ * x₂^p₂ * ... * xⱼ^pⱼ * ...)/∂xⱼ = pⱼ * x₁^p₁ * ... * xⱼ^(pⱼ-1) * ...
                    grad_powers = powers.copy()
                    grad_powers[j] -= 1
                    
                    # Обчислюємо значення градієнта
                    grad_value = powers[j]  # Коефіцієнт від диференціювання
                    
                    for k in range(n_features):
                        if grad_powers[k] > 0:
                            grad_value *= (X0[:, k] ** grad_powers[k])
                        # Якщо grad_powers[k] == 0, то x^0 = 1 (не множимо)
                    
                    grad_poly[:, j, i] = grad_value
        
        return grad_poly

    def _run_random_search(self, X, Y):
        """Пошук оптимальних гіперпараметрів"""
        
        if self.linear_type == "ols":
            # OLS не має гіперпараметрів для оптимізації
            return LinearRegression(fit_intercept=self.include_bias)
            
        elif self.linear_type == "ridge":
            param_dist = {"alpha": loguniform(1e-3, 1e3)}
            base = Ridge(fit_intercept=self.include_bias)
            
        elif self.linear_type == "lasso":
            param_dist = {"alpha": loguniform(1e-4, 1e1)}
            base = Lasso(fit_intercept=self.include_bias, max_iter=2000)
            
        rs = RandomizedSearchCV(
            base,
            param_dist,
            n_iter=self.n_iter_random_search,
            cv=3,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        rs.fit(X, Y)
        print(f"✅ Оптимальні параметри Linear {self.linear_type}: {rs.best_params_}")
        
        return rs.best_estimator_
    
class _NeuralNetworkModel(_BaseKernelModel):
    """
    Нейронна мережа (MLP) для системи MPC з підтримкою лінеаризації.
    Реалізує той самий інтерфейс, що й інші моделі у системі.
    """
    
    def __init__(
        self,
        hidden_layer_sizes: tuple = (50, 25),  # Архітектура мережі
        activation: str = 'relu',              # Функція активації
        solver: str = 'adam',                  # Оптимізатор
        alpha: float = 0.001,                  # L2 регуляризація
        learning_rate_init: float = 0.001,     # Початкова швидкість навчання
        max_iter: int = 2000,                  # Максимальна кількість епох
        early_stopping: bool = True,           # Рання зупинка
        validation_fraction: float = 0.1,      # Частка валідації для ранньої зупинки
        n_iter_no_change: int = 20,           # Терпіння для ранньої зупинки
        find_optimal_params: bool = False,     # Автопошук гіперпараметрів
        n_iter_random_search: int = 30,        # Кількість ітерацій випадкового пошуку
        random_state: int = 42                 # Фіксація випадковості
    ):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.find_optimal_params = find_optimal_params
        self.n_iter_random_search = n_iter_random_search
        self.random_state = random_state
        
        # Внутрішні атрибути
        self.model: MLPRegressor | None = None
        self.n_features_: int | None = None
        self.n_outputs_: int | None = None
        
        # Для сумісності з kernel interface
        self._kernel = "neural_network"

    def fit(self, X: np.ndarray, Y: np.ndarray, config_params: dict = None) -> None:
        """
        Навчання нейронної мережі на тренувальних даних з підтримкою конфігураційних параметрів.
        
        Args:
            X: Тренувальні ознаки
            Y: Тренувальні цілі
            config_params: Додаткові параметри з конфігурації (включаючи param_search_space)
        """
        print(f"🧠 Навчання Neural Network...")
        print(f"   Архітектура: {self.hidden_layer_sizes}")
        print(f"   Активація: {self.activation}, Solver: {self.solver}")
        
        self.n_features_ = X.shape[1]
        self.n_outputs_ = Y.shape[1] if Y.ndim > 1 else 1
        
        if self.find_optimal_params:
            print(f"🔍 Автоматичний пошук оптимальних гіперпараметрів...")
            # Передаємо конфігураційні параметри до методу пошуку
            self.model = self._run_random_search(X, Y, config_params)
        else:
            # Створення моделі з заданими параметрами (без змін)
            self.model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                early_stopping=self.early_stopping,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                random_state=self.random_state
            )
            
            print(f"📚 Навчання на {X.shape[0]} зразках...")
            self.model.fit(X, Y)
        
        # Перевірка конвергенції (без змін)
        if hasattr(self.model, 'n_iter_'):
            print(f"✅ Навчання завершено за {self.model.n_iter_} епох")
            if self.model.n_iter_ >= self.max_iter:
                print(f"⚠️  Досягнуто максимальну кількість епох. Можливо потрібно збільшити max_iter")
        
        print(f"   Фінальна функція втрат: {getattr(self.model, 'loss_', 'невідома'):.6f}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Передбачення за допомогою навченої нейронної мережі."""
        if self.model is None:
            raise RuntimeError("Нейронна мережа не навчена. Викличте fit() спочатку.")
        
        predictions = self.model.predict(X)
        
        # Забезпечуємо правильну форму виходу
        if predictions.ndim == 1 and self.n_outputs_ > 1:
            predictions = predictions.reshape(-1, self.n_outputs_)
        elif predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
            
        return predictions

    def linearize(self, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Лінеаризація нейронної мережі в точці X0 за допомогою числового диференціювання.
        
        Повертає:
            W: (n_features, n_outputs) - матриця градієнтів (якобіан)
            b: (n_outputs,) - вектор зміщення для лінійної апроксимації
        """
        if self.model is None:
            raise RuntimeError("Нейронна мережа не навчена.")
        
        if X0.ndim == 1:
            X0 = X0.reshape(1, -1)
        
        # Параметри для числового диференціювання
        epsilon = 1e-7  # Крок для обчислення градієнту
        n_features = X0.shape[1]
        
        # Передбачення в точці X0
        y0 = self.predict(X0)[0]  # (n_outputs,)
        n_outputs = len(y0)
        
        # Ініціалізація матриці якобіану
        W = np.zeros((n_features, n_outputs))
        
        # Обчислення часткових похідних для кожної змінної
        for i in range(n_features):
            # Створюємо збурену точку
            X_plus = X0.copy()
            X_minus = X0.copy()
            
            X_plus[0, i] += epsilon
            X_minus[0, i] -= epsilon
            
            # Обчислюємо градієнт методом центральних різниць
            y_plus = self.predict(X_plus)[0]
            y_minus = self.predict(X_minus)[0]
            
            # Часткова похідна для i-ї змінної
            grad_i = (y_plus - y_minus) / (2 * epsilon)
            W[i, :] = grad_i
        
        # Обчислюємо зміщення для лінійної апроксимації
        # y ≈ y0 + W^T * (x - x0) = (y0 - W^T * x0) + W^T * x
        # Тому b = y0 - W^T * x0
        b = y0 - X0[0] @ W
        
        # Обмеження градієнтів для стабільності
        W = np.clip(W, -1e3, 1e3)
        b = np.clip(b, -1e3, 1e3)
        
        return W, b

    def _run_random_search(self, X: np.ndarray, Y: np.ndarray, config_params: dict = None) -> MLPRegressor:
        """
        Випадковий пошук оптимальних гіперпараметрів для нейронної мережі.
        Спрощена версія без непотрібних функцій.
        """
        
        print(f"🎯 Випадковий пошук серед {self.n_iter_random_search} конфігурацій...")
        
        # Визначаємо простір пошуку
        if config_params and 'param_search_space' in config_params:
            print(f"📋 Використовуємо обмежений простір пошуку з конфігурації")
            custom_space = config_params['param_search_space']
            
            param_dist = {}
            
            # Архітектура
            if 'hidden_layer_sizes' in custom_space:
                param_dist['hidden_layer_sizes'] = custom_space['hidden_layer_sizes']
            else:
                param_dist['hidden_layer_sizes'] = [(50,), (100,), (50, 25), (100, 50)]
                
            # Функції активації  
            if 'activation' in custom_space:
                param_dist['activation'] = custom_space['activation']
            else:
                param_dist['activation'] = ['relu', 'tanh']
                
            # Оптимізатори - тільки надійні
            if 'solver' in custom_space:
                param_dist['solver'] = custom_space['solver']
            else:
                param_dist['solver'] = ['adam']  # Тільки Adam для стабільності
                
            # Регуляризація
            if 'alpha' in custom_space:
                if isinstance(custom_space['alpha'], list):
                    param_dist['alpha'] = custom_space['alpha']
                else:
                    param_dist['alpha'] = loguniform(1e-5, 1e-1)
            else:
                param_dist['alpha'] = loguniform(1e-5, 1e-1)
                
            # Швидкість навчання
            if 'learning_rate_init' in custom_space:
                if isinstance(custom_space['learning_rate_init'], list):
                    param_dist['learning_rate_init'] = custom_space['learning_rate_init']
                else:
                    param_dist['learning_rate_init'] = loguniform(1e-4, 1e-1)
            else:
                param_dist['learning_rate_init'] = loguniform(1e-4, 1e-1)
                
        else:
            print(f"📋 Використовуємо стандартний простір пошуку")
            # Спрощений стандартний простір пошуку
            param_dist = {
                'hidden_layer_sizes': [
                    (50,), (100,), (50, 25), (100, 50), (100, 50, 25), (150, 75)
                ],
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],  # Тільки надійний Adam
                'alpha': loguniform(1e-5, 1e-1),
                'learning_rate_init': loguniform(1e-4, 1e-2)
            }
        
        # Базова модель з оптимальними параметрами для Adam
        base_model = MLPRegressor(
            max_iter=2000,                    # Достатньо для збіжності Adam
            early_stopping=True,              # Adam добре працює з early stopping
            validation_fraction=0.15,         # Трохи більше даних для валідації
            n_iter_no_change=25,             # Більше терпіння
            random_state=self.random_state
        )
        
        # Пригнічуємо попередження про збіжність під час пошуку
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            
            random_search = RandomizedSearchCV(
                base_model,
                param_dist,
                n_iter=self.n_iter_random_search,
                cv=3,
                scoring='neg_mean_squared_error',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
            
            print(f"🔍 Запуск пошуку...")
            random_search.fit(X, Y)
        
        print(f"✅ Знайдено оптимальні параметри:")
        for param, value in random_search.best_params_.items():
            print(f"   • {param}: {value}")
        print(f"   • Найкращий результат CV: {-random_search.best_score_:.6f}")
        
        # Повертаємо найкращу модель
        best_model = random_search.best_estimator_
        
        # Якщо модель все ще має проблеми зі збіжністю, даємо їй ще один шанс
        if hasattr(best_model, 'n_iter_') and best_model.n_iter_ >= best_model.max_iter - 10:
            print(f"🔄 Модель близька до ліміту ітерацій. Перенавчаємо з більшим max_iter...")
            
            # Створюємо копію з збільшеними параметрами
            final_model = MLPRegressor(
                hidden_layer_sizes=best_model.hidden_layer_sizes,
                activation=best_model.activation,
                solver=best_model.solver,
                alpha=best_model.alpha,
                learning_rate_init=getattr(best_model, 'learning_rate_init', 0.001),
                max_iter=4000,                # Подвоюємо кількість ітерацій
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,         # Ще більше терпіння
                random_state=self.random_state
            )
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                final_model.fit(X, Y)
                
            print(f"✅ Перенавчання завершено за {getattr(final_model, 'n_iter_', 'невідомо')} ітерацій")
            return final_model
        
        return best_model    
    
    def get_model_info(self) -> dict:
        """Повертає інформацію про навчену модель."""
        if self.model is None:
            return {"status": "не навчена"}
        
        info = {
            "status": "навчена",
            "architecture": self.hidden_layer_sizes,
            "activation": self.activation,
            "solver": self.solver,
            "n_features": self.n_features_,
            "n_outputs": self.n_outputs_,
            "n_epochs": getattr(self.model, 'n_iter_', 'невідомо'),
            "final_loss": getattr(self.model, 'loss_', 'невідома')
        }
        
        # Додаткова інформація якщо доступна
        if hasattr(self.model, 'coefs_'):
            total_params = sum(w.size for w in self.model.coefs_) + sum(b.size for b in self.model.intercepts_)
            info["total_parameters"] = total_params
            
        return info
    
# ======================================================================
#                              FACADE
# ======================================================================
class KernelModel:
    """
    Являє собою «тонкий» фасад.  
    За `model_type` вибирається конкретна стратегія, а зайві kwargs
    автоматично відкидаються (щоб не виникало TypeError).
    """

    _REGISTRY: Dict[str, Type[_BaseKernelModel]] = {
        "krr": _KRRModel,
        "gpr": _GPRModel,
        "svr": _SVRModel,
        "linear": _LinearModel,
        "nn": _NeuralNetworkModel,      # 🆕 ДОДАНО Neural Network!
        "neural": _NeuralNetworkModel,  # 🆕 Альтернативна назва
    }

    def __init__(self, model_type: str = "krr", **kwargs):
        mtype = model_type.lower()
        if mtype not in self._REGISTRY:
            raise ValueError(f"Невідома модель '{model_type}'. Доступні: {list(self._REGISTRY)}")

        impl_cls = self._REGISTRY[mtype]

        # --- фільтрація аргументів під конкретний __init__ ---
        sig = inspect.signature(impl_cls.__init__)
        impl_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        self._impl = impl_cls(**impl_kwargs)

    def fit(self, X: np.ndarray, Y: np.ndarray, config_params: dict = None) -> None:
        """
        Навчання моделі з передачею конфігураційних параметрів.
        
        Args:
            X: Тренувальні ознаки
            Y: Тренувальні цілі  
            config_params: Параметри конфігурації для передачі до внутрішньої моделі
        """
        return self._impl.fit(X, Y, config_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._impl.predict(X)

    def linearize(self, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._impl.linearize(X0)

    def __getattr__(self, item):
        return getattr(self._impl, item)

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tqdm import tqdm  # прогрес-бар


import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_rel


def analyze_experiment_results(
    results_dir: str = "exp_results",
    kernel_focus: Optional[List[str]] = None,   # напр., ["rbf", "linear"]
    metrics_gain: List[str] = ("RMSE_Fe_gain_%", "RMSE_Mass_gain_%", "MSE_gain_%"),
    do_ttests: bool = True,
    save_figs: bool = True,
    show_figs: bool = False,
    export_tables: bool = True,
    style: str = "whitegrid",
    cmap: str = "RdYlGn",
) -> Dict[str, Any]:
    """
    Єдина функція аналізу результатів батч-експериментів.

    Що робить:
      - Завантажує summary.csv і всі details_*.csv з директорії results_dir
      - Будує теплові карти гейнів (по lag×N), профілі гейнів по lag для кожного N
      - Будує графіки часу тренування
      - (опційно) Виконує парні t-тести по сідах для RMSE (ARX vs KRR)
      - Зберігає графіки та таблиці у підпапку 'analysis' і повертає все у словнику

    Повертає:
      {
        "df_summary": pd.DataFrame,
        "df_details": pd.DataFrame | None,
        "ttests": pd.DataFrame | None,
        "fig_paths": List[str],
        "table_paths": Dict[str, str],
        "meta": Dict[str, Any],
      }

    Примітки:
      - Очікується, що summary.csv має колонки:
          ["kernel","N","lag","seeds", "RMSE_Fe_gain_%","RMSE_Mass_gain_%","MSE_gain_%","ARX_time_s","KRR_time_s", ...]
      - Очікується, що details_*.csv містять принаймні:
          ["seed","N","lag","kernel","MSE_ARX","RMSE_Fe_ARX","RMSE_Mass_ARX","MSE_KRR","RMSE_Fe_KRR","RMSE_Mass_KRR","Train_s_ARX","Train_s_KRR"]
        (N, lag, kernel додаються при завантаженні з імені файлу)
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Директорія не знайдена: {results_dir}")

    out_dir = results_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Headless-сейф режим для фігур
    if not show_figs:
        matplotlib.use("Agg")

    sns.set(style=style)

    # 1) Завантаження summary
    summary_fp = results_path / "summary.csv"
    if not summary_fp.exists():
        raise FileNotFoundError(f"Не знайдено summary.csv у {results_dir}")

    df_summary = pd.read_csv(summary_fp)
    # Мінімальна валідація
    required_summary_cols = {"kernel", "N", "lag"}
    if not required_summary_cols.issubset(df_summary.columns):
        raise ValueError(f"summary.csv не містить потрібних колонок: {sorted(required_summary_cols - set(df_summary.columns))}")

    # 2) Завантаження details (опційно)
    details_files = sorted(results_path.glob("details_*.csv"))
    df_details = None
    if details_files:
        rows = []
        for fp in details_files:
            m = re.match(r"details_N(?P<N>\d+)_L(?P<L>\d+)_K(?P<K>\w+)\.csv", fp.name)
            if not m:
                # Пропустити файли, що не підпадають під патерн
                continue
            N = int(m.group("N")); L = int(m.group("L")); K = m.group("K")
            d = pd.read_csv(fp)
            d["N"] = N
            d["lag"] = L
            d["kernel"] = K
            rows.append(d)
        if rows:
            df_details = pd.concat(rows, ignore_index=True)

    # Ядра для аналізу
    if kernel_focus is None:
        kernel_focus = sorted(df_summary["kernel"].unique().tolist())

    fig_paths: List[str] = []
    table_paths: Dict[str, str] = {}

    # 3) Експорт базових таблиць
    if export_tables:
        tbl_rbf = (df_summary
                   .query("kernel=='rbf'") if "rbf" in df_summary["kernel"].unique() else df_summary.copy())
        tbl_rbf = tbl_rbf.loc[:, [c for c in ["kernel","N","lag","seeds","RMSE_Fe_gain_%","RMSE_Mass_gain_%","MSE_gain_%","KRR_time_s","ARX_time_s"] if c in df_summary.columns]]
        table_paths["table_rbf_gains.csv"] = str((out_dir / "table_rbf_gains.csv").resolve())
        tbl_rbf.to_csv(table_paths["table_rbf_gains.csv"], index=False)

        # Зведені півод-таблиці по гейнах (для кожного kernel окремо)
        for k in kernel_focus:
            for metric in metrics_gain:
                if metric not in df_summary.columns:
                    continue
                pivot = (df_summary[df_summary["kernel"] == k]
                         .pivot_table(index="N", columns="lag", values=metric, aggfunc="mean"))
                fp = out_dir / f"pivot_{metric}_kernel-{k}.csv"
                pivot.to_csv(fp)
                table_paths[fp.name] = str(fp.resolve())

    # 4) Графіки — теплові карти гейнів
    def plot_heatmap_gain(df, metric: str, kernel: str, title: Optional[str] = None):
        d = df[(df["kernel"] == kernel)].pivot_table(index="N", columns="lag", values=metric, aggfunc="mean")
        if d.empty:
            return None
        plt.figure(figsize=(6, 4))
        ax = sns.heatmap(d, annot=True, fmt=".1f", cmap=cmap, center=0, cbar_kws={"label": metric})
        plt.title(title or f"{metric} — kernel={kernel}")
        plt.ylabel("N"); plt.xlabel("lag")
        plt.tight_layout()
        out_fp = out_dir / f"heatmap_{metric}_kernel-{kernel}.png"
        plt.savefig(out_fp, dpi=200)
        if show_figs:
            plt.show()
        plt.close()
        return str(out_fp.resolve())

    for k in kernel_focus:
        for metric in metrics_gain:
            if metric in df_summary.columns:
                fp = plot_heatmap_gain(df_summary, metric=metric, kernel=k, title=None)
                if fp:
                    fig_paths.append(fp)

    # 5) Профілі гейнів по lag для кожного N
    def plot_line_profile(df, metric: str, kernel: str):
        vals = df[(df["kernel"] == kernel)]
        if vals.empty or metric not in vals.columns:
            return None
        Ns_sorted = sorted(vals["N"].unique().tolist())
        plt.figure(figsize=(7, 4))
        for N in Ns_sorted:
            d = vals[vals["N"] == N].sort_values("lag")
            plt.plot(d["lag"], d[metric], marker="o", label=f"N={N}")
        plt.axhline(0, color="k", lw=1)
        plt.title(f"{metric} vs lag — {kernel}")
        plt.xlabel("lag"); plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        out_fp = out_dir / f"line_{metric}_kernel-{kernel}.png"
        plt.savefig(out_fp, dpi=200)
        if show_figs:
            plt.show()
        plt.close()
        return str(out_fp.resolve())

    for k in kernel_focus:
        for metric in metrics_gain:
            if metric in df_summary.columns:
                fp = plot_line_profile(df_summary, metric=metric, kernel=k)
                if fp:
                    fig_paths.append(fp)

    # 6) Час тренування (KRR_time_s по lag для кожного N і ядра)
    if "KRR_time_s" in df_summary.columns:
        for k in kernel_focus:
            d_k = df_summary[df_summary["kernel"] == k]
            if d_k.empty:
                continue
            plt.figure(figsize=(7, 4))
            for N in sorted(d_k["N"].unique().tolist()):
                d = d_k[d_k["N"] == N].sort_values("lag")
                if "KRR_time_s" in d.columns:
                    plt.plot(d["lag"], d["KRR_time_s"], marker="s", label=f"N={N}")
            plt.title(f"KRR час тренування vs lag — kernel={k}")
            plt.xlabel("lag"); plt.ylabel("секунди")
            plt.legend()
            plt.tight_layout()
            out_fp = out_dir / f"time_KRR_kernel-{k}.png"
            plt.savefig(out_fp, dpi=200)
            if show_figs:
                plt.show()
            plt.close()
            fig_paths.append(str(out_fp.resolve()))

    # 7) Парні t-тести (по сідах) — якщо доступні details
    df_sig = None
    if do_ttests and df_details is not None:
        required_detail_cols = {"RMSE_Fe_ARX", "RMSE_Fe_KRR", "RMSE_Mass_ARX", "RMSE_Mass_KRR", "N", "lag", "kernel"}
        if required_detail_cols.issubset(df_details.columns):
            rows = []
            for (N, L, K) in sorted(df_details.groupby(["N", "lag", "kernel"]).groups.keys()):
                g = df_details[(df_details["N"] == N) & (df_details["lag"] == L) & (df_details["kernel"] == K)]
                if len(g) < 2:
                    # Двох і більше сідів потрібно для t-тесту
                    continue
                # t-тести RMSE Fe і Mass
                t_fe, p_fe = ttest_rel(g["RMSE_Fe_ARX"], g["RMSE_Fe_KRR"])
                t_ms, p_ms = ttest_rel(g["RMSE_Mass_ARX"], g["RMSE_Mass_KRR"])
                rows.append({
                    "N": N, "lag": L, "kernel": K, "n_seeds": len(g),
                    "p_value_fe": p_fe,
                    "p_value_mass": p_ms,
                    "mean_gain_fe_%": 100.0 * (g["RMSE_Fe_ARX"].mean() - g["RMSE_Fe_KRR"].mean()) / (g["RMSE_Fe_ARX"].mean() + 1e-12),
                    "mean_gain_mass_%": 100.0 * (g["RMSE_Mass_ARX"].mean() - g["RMSE_Mass_KRR"].mean()) / (g["RMSE_Mass_ARX"].mean() + 1e-12),
                })
            if rows:
                df_sig = pd.DataFrame(rows).sort_values(["kernel", "N", "lag"]).reset_index(drop=True)
                if export_tables:
                    fp = out_dir / "ttests_by_combo.csv"
                    df_sig.to_csv(fp, index=False)
                    table_paths[fp.name] = str(fp.resolve())

    # 8) Зберегти метадані аналізу
    meta = {
        "results_dir": str(results_path.resolve()),
        "analysis_dir": str(out_dir.resolve()),
        "kernels_analyzed": kernel_focus,
        "metrics_gain": list(metrics_gain),
        "has_details": df_details is not None,
        "fig_paths": fig_paths,
        "table_paths": table_paths,
    }
    with open(out_dir / "analysis_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "df_summary": df_summary,
        "df_details": df_details,
        "ttests": df_sig,
        "fig_paths": fig_paths,
        "table_paths": table_paths,
        "meta": meta,
    }


def run_experiment_suite(
    reference_df: pd.DataFrame,
    lags=(1,2,3,4),
    Ns=(5000, 7000),
    krr_kernels=("linear", "rbf"),
    anomaly_severity="mild",
    use_anomalies=True,
    seeds=(123, 42),
    noise_level="none",
    results_dir="exp_results",
    n_iter_search_rbf=15,
    verbose=False,
):
    """
    Батч-експерименти за сіткою (N, lag, kernel∈{linear, rbf}) + мультисіди з прогрес-баром (tqdm).
    Зберігає детальні CSV для кожної комбінації і summary.csv з агрегатами.
    """

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    rows_all, detail_records = [], []

    # Підрахунок загальної кількості ітерацій (цикл = одна комбінація для одного seed)
    total_cycles = len(Ns) * len(lags) * len(krr_kernels) * len(seeds)

    # Зовнішній прогрес-бар по всім ітераціям
    pbar = tqdm(total=total_cycles, desc="Experiment grid", unit="cycle")

    for N_data in Ns:
        for lag in lags:
            for kernel in krr_kernels:

                metrics = []
                for sd in seeds:
                    t_cycle_start = time.time()

                    # 1) Параметри генерації
                    simulation_params = {
                        "N_data": int(N_data),
                        "control_pts": max(12, int(0.1*N_data)),
                        "lag": int(lag),
                        "train_size": 0.8,
                        "val_size": 0.1,
                        "test_size": 0.1,
                        "time_step_s": 5,
                        "time_constants_s": {
                            "concentrate_fe_percent": 8.0,
                            "tailings_fe_percent": 10.0,
                            "concentrate_mass_flow": 5.0,
                            "tailings_mass_flow": 7.0
                        },
                        "dead_times_s": {
                            "concentrate_fe_percent": 20.0,
                            "tailings_fe_percent": 25.0,
                            "concentrate_mass_flow": 20.0,
                            "tailings_mass_flow": 25.0
                        },
                        "plant_model_type": "rf",
                        "seed": int(sd),
                        "n_neighbors": 5,
                        "noise_level": noise_level,
                        # Нелінійність (фіксована)
                        "enable_nonlinear": True,
                        "nonlinear_config": {
                            "concentrate_fe_percent": ("pow", 2.0),
                            "concentrate_mass_flow": ("pow", 1.6),
                        },
                        # Аномалії
                        "use_anomalies": bool(use_anomalies),
                        "anomaly_severity": anomaly_severity,
                        "anomaly_in_train": False,
                    }

                    # 2) Дані
                    _, df_sim = create_simulation_data(reference_df, simulation_params)

                    # 3) Лаги + спліт
                    X, Y = _create_lagged_matrices_corrected(df_sim, lag)
                    n = X.shape[0]
                    n_train = int(simulation_params["train_size"] * n)
                    n_val = int(simulation_params["val_size"] * n)
                    Xtr, Ytr = X[:n_train], Y[:n_train]
                    Xva, Yva = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
                    Xte, Yte = X[n_train+n_val:], Y[n_train+n_val:]

                    # 4) Скейлінг
                    xs, ys = StandardScaler(), StandardScaler()
                    Xtr_s, Ytr_s = xs.fit_transform(Xtr), ys.fit_transform(Ytr)
                    Xva_s, Yva_s = xs.transform(Xva), ys.transform(Yva)
                    Xte_s = xs.transform(Xte)

                    # 5) ARX (лінійний бенчмарк)
                    arx = KernelModel(model_type="linear", linear_type="ols", poly_degree=1, include_bias=True)
                    t0 = time.time()
                    try:
                        arx.fit(Xtr_s, Ytr_s, X_val=Xva_s, Y_val=Yva_s)
                    except TypeError:
                        arx.fit(Xtr_s, Ytr_s)
                    arx_t = time.time() - t0
                    Yhat_arx = ys.inverse_transform(arx.predict(Xte_s))
                    mse_arx = mean_squared_error(Yte, Yhat_arx)
                    rmse_fe_arx = np.sqrt(mean_squared_error(Yte[:,0], Yhat_arx[:,0]))
                    rmse_mass_arx = np.sqrt(mean_squared_error(Yte[:,1], Yhat_arx[:,1]))

                    # 6) KRR (linear або rbf)
                    if kernel not in ("linear", "rbf"):
                        raise ValueError("Підтримуються тільки 'linear' та 'rbf' для KRR")
                    krr_kwargs = {"model_type": "krr", "kernel": kernel}
                    if kernel == "rbf":
                        krr_kwargs.update({"find_optimal_params": True, "n_iter_random_search": int(n_iter_search_rbf)})
                    else:
                        krr_kwargs.update({"find_optimal_params": False})

                    krr = KernelModel(**krr_kwargs)
                    t0 = time.time()
                    try:
                        krr.fit(Xtr_s, Ytr_s, X_val=Xva_s, Y_val=Yva_s)
                    except TypeError:
                        krr.fit(Xtr_s, Ytr_s)
                    krr_t = time.time() - t0
                    Yhat_krr = ys.inverse_transform(krr.predict(Xte_s))
                    mse_krr = mean_squared_error(Yte, Yhat_krr)
                    rmse_fe_krr = np.sqrt(mean_squared_error(Yte[:,0], Yhat_krr[:,0]))
                    rmse_mass_krr = np.sqrt(mean_squared_error(Yte[:,1], Yhat_krr[:,1]))

                    metrics.append({
                        "seed": sd, "N": N_data, "lag": lag, "kernel": kernel,
                        "MSE_ARX": mse_arx, "RMSE_Fe_ARX": rmse_fe_arx, "RMSE_Mass_ARX": rmse_mass_arx, "Train_s_ARX": arx_t,
                        "MSE_KRR": mse_krr, "RMSE_Fe_KRR": rmse_fe_krr, "RMSE_Mass_KRR": rmse_mass_krr, "Train_s_KRR": krr_t,
                    })

                    # Оновлюємо прогрес-бар + ETA
                    t_cycle = time.time() - t_cycle_start
                    pbar.set_postfix({
                        "N": N_data, "L": lag, "K": kernel, "seed": sd,
                        "cycle_s": f"{t_cycle:.1f}",
                        "arx_s": f"{arx_t:.1f}",
                        "krr_s": f"{krr_t:.1f}",
                    })
                    pbar.update(1)

                # 7) Агрегація по сідах для комбінації (N, lag, kernel)
                dfm = pd.DataFrame(metrics)
                agg = dfm.agg({
                    "MSE_ARX": ["mean","std"], "RMSE_Fe_ARX": ["mean","std"], "RMSE_Mass_ARX": ["mean","std"], "Train_s_ARX": ["mean","std"],
                    "MSE_KRR": ["mean","std"], "RMSE_Fe_KRR": ["mean","std"], "RMSE_Mass_KRR": ["mean","std"], "Train_s_KRR": ["mean","std"],
                }).T.reset_index()
                agg.columns = ["metric", "mean", "std"]

                summary = {
                    "N": N_data, "lag": lag, "kernel": kernel, "seeds": len(seeds),
                    "MSE_gain_%": (agg.loc[agg.metric=="MSE_ARX","mean"].values[0] - agg.loc[agg.metric=="MSE_KRR","mean"].values[0])
                                   / (agg.loc[agg.metric=="MSE_ARX","mean"].values[0] + 1e-12) * 100,
                    "RMSE_Fe_gain_%": (agg.loc[agg.metric=="RMSE_Fe_ARX","mean"].values[0] - agg.loc[agg.metric=="RMSE_Fe_KRR","mean"].values[0])
                                   / (agg.loc[agg.metric=="RMSE_Fe_ARX","mean"].values[0] + 1e-12) * 100,
                    "RMSE_Mass_gain_%": (agg.loc[agg.metric=="RMSE_Mass_ARX","mean"].values[0] - agg.loc[agg.metric=="RMSE_Mass_KRR","mean"].values[0])
                                   / (agg.loc[agg.metric=="RMSE_Mass_ARX","mean"].values[0] + 1e-12) * 100,
                    "ARX_time_s": agg.loc[agg.metric=="Train_s_ARX","mean"].values[0],
                    "KRR_time_s": agg.loc[agg.metric=="Train_s_KRR","mean"].values[0],
                }
                rows_all.append(summary)

                # Збереження детальних метрик по сідах для цієї комбінації
                detail_path = Path(results_dir) / f"details_N{N_data}_L{lag}_K{kernel}.csv"
                dfm.to_csv(detail_path, index=False)
                detail_records.append({"combo": (N_data, lag, kernel), "path": str(detail_path)})

    pbar.close()

    df_summary = pd.DataFrame(rows_all).sort_values(["kernel","N","lag"]).reset_index(drop=True)
    summary_path = Path(results_dir) / "summary.csv"
    df_summary.to_csv(summary_path, index=False)

    meta = {
        "created_at": pd.Timestamp.now().isoformat(),
        "lags": list(lags), "Ns": list(Ns), "krr_kernels": list(krr_kernels),
        "anomaly_severity": anomaly_severity, "use_anomalies": use_anomalies,
        "seeds": list(seeds), "noise_level": noise_level,
        "detail_files": detail_records, "summary_csv": str(summary_path),
    }
    with open(Path(results_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"Summary saved to: {summary_path}")
        print(pd.DataFrame(rows_all))

    return df_summary
def compare_linear_vs_kernel_models(reference_df=None, **kwargs):
    """
    Пряме порівняння лінійних (ARX) та ядерних моделей для дисертації.
    Тепер: підтримка конфігурованих аномалій у val/test і спільний anomaly_cfg.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    import numpy as np
    from data_gen import StatefulDataGenerator

    print("🎓 ПОРІВНЯННЯ МОДЕЛЕЙ ДЛЯ ДИСЕРТАЦІЇ")
    print("=" * 60)
    print("Розділ 2.1.1: Логічний перехід до ядерних моделей")
    print("=" * 60)

    # ----1. Підготовка референтних даних
    if reference_df is None:
        print("📊 Завантаження референтних даних...")
        try:
            reference_df = pd.read_parquet('processed.parquet')
            print(f"✅ Завантажено {len(reference_df)} записів")
        except FileNotFoundError:
            print("❌ Файл 'processed.parquet' не знайдено")

    # ---- Параметри симуляції
    train_size = kwargs.get('train_size', 0.8)
    val_size   = kwargs.get('val_size', 0.1)
    test_size  = kwargs.get('test_size', 0.1)

    simulation_params = {
        'N_data': kwargs.get('N_data', 7000),
        'control_pts': 700,
        'lag': kwargs.get('lag', 2),
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'time_step_s': 5,
        'time_constants_s': {
            'concentrate_fe_percent': 8.0,
            'tailings_fe_percent': 10.0,
            'concentrate_mass_flow': 5.0,
            'tailings_mass_flow': 7.0
        },
        'dead_times_s': {
            'concentrate_fe_percent': 20.0,
            'tailings_fe_percent': 25.0,
            'concentrate_mass_flow': 20.0,
            'tailings_mass_flow': 25.0
        },
        'plant_model_type': 'rf',
        'seed': kwargs.get('seed', 42),
        'n_neighbors': 5,
        'noise_level': kwargs.get('noise_level', 'none'),

        # Нелінійність
        'enable_nonlinear': True,
        'nonlinear_config': {
            'concentrate_fe_percent': ('pow', 2.0),
            'concentrate_mass_flow': ('pow', 1.5)
        },

        # Аномалії
        'use_anomalies': kwargs.get('use_anomalies', True),
        'anomaly_severity': kwargs.get('anomaly_severity', 'mild'),
        'anomaly_in_train': kwargs.get('anomaly_in_train', False),
    }

    print(f"📈 Створення симуляційних даних (N={simulation_params['N_data']}, L={simulation_params['lag']})...")
    true_gen, df_sim = create_simulation_data(reference_df, simulation_params)

    # ---- Лаговані матриці
    X, Y = _create_lagged_matrices_corrected(df_sim, simulation_params['lag'])
    print(f"   Розмірність X: {X.shape}, Y: {Y.shape}")

    # ----2. Спліт на train/val/test
    n = X.shape[0]
    n_train = int(train_size * n)
    n_val   = int(val_size * n)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_test,  Y_test  = X[n_train + n_val:], Y[n_train + n_val:]

    # ---- Нормалізація (fit тільки на train)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    Y_train_scaled = y_scaler.fit_transform(Y_train)
    X_val_scaled   = x_scaler.transform(X_val)
    Y_val_scaled   = y_scaler.transform(Y_val)
    X_test_scaled  = x_scaler.transform(X_test)
    Y_test_scaled  = y_scaler.transform(Y_test)

    print(f"   Тренувальний набір: {X_train_scaled.shape[0]} зразків")
    print(f"   Валідаційний набір: {X_val_scaled.shape[0]} зразків")
    print(f"   Тестовий набір: {X_test_scaled.shape[0]} зразків")

    # ----3. Лінійна модель (ARX)
    print("\n🔴 НАВЧАННЯ ЛІНІЙНОЇ МОДЕЛІ (ARX)")
    print("-" * 40)
    linear_model = KernelModel(model_type='linear', linear_type='ols', poly_degree=1, include_bias=True)

    import time
    start_time = time.time()
    try:
        # Якщо реалізація підтримує val-дані для тюнінгу
        linear_model.fit(X_train_scaled, Y_train_scaled, X_val=X_val_scaled, Y_val=Y_val_scaled)
    except TypeError:
        linear_model.fit(X_train_scaled, Y_train_scaled)
    linear_train_time = time.time() - start_time

    Y_pred_linear_scaled = linear_model.predict(X_test_scaled)
    Y_pred_linear = y_scaler.inverse_transform(Y_pred_linear_scaled)

    linear_mse = mean_squared_error(Y_test, Y_pred_linear)
    linear_rmse_fe = np.sqrt(mean_squared_error(Y_test[:, 0], Y_pred_linear[:, 0]))
    linear_rmse_mass = np.sqrt(mean_squared_error(Y_test[:, 1], Y_pred_linear[:, 1]))

    print(f"   ⏱️ Час навчання: {linear_train_time:.3f} сек")
    print(f"   📊 MSE: {linear_mse:.6f}")
    print(f"   📊 RMSE Fe: {linear_rmse_fe:.3f}")
    print(f"   📊 RMSE Mass: {linear_rmse_mass:.3f}")

    # ----4. Ядерна модель (KRR, RBF)
    print("\n🟢 НАВЧАННЯ ЯДЕРНОЇ МОДЕЛІ (KRR)")
    print("-" * 40)
    kernel_model = KernelModel(
        model_type='krr',
        kernel='rbf',
        find_optimal_params=kwargs.get('find_optimal_params', True),
        n_iter_random_search=kwargs.get('n_iter_search', 20)
    )

    start_time = time.time()
    try:
        kernel_model.fit(X_train_scaled, Y_train_scaled, X_val=X_val_scaled, Y_val=Y_val_scaled)
    except TypeError:
        kernel_model.fit(X_train_scaled, Y_train_scaled)
    kernel_train_time = time.time() - start_time

    Y_pred_kernel_scaled = kernel_model.predict(X_test_scaled)
    Y_pred_kernel = y_scaler.inverse_transform(Y_pred_kernel_scaled)

    kernel_mse = mean_squared_error(Y_test, Y_pred_kernel)
    kernel_rmse_fe = np.sqrt(mean_squared_error(Y_test[:, 0], Y_pred_kernel[:, 0]))
    kernel_rmse_mass = np.sqrt(mean_squared_error(Y_test[:, 1], Y_pred_kernel[:, 1]))

    print(f"   ⏱️ Час навчання: {kernel_train_time:.3f} сек")
    print(f"   📊 MSE: {kernel_mse:.6f}")
    print(f"   📊 RMSE Fe: {kernel_rmse_fe:.3f}")
    print(f"   📊 RMSE Mass: {kernel_rmse_mass:.3f}")

    # ----5. Порівняння і аналіз нелінійності
    improvement_mse = ((linear_mse - kernel_mse) / (linear_mse + 1e-12)) * 100
    improvement_fe = ((linear_rmse_fe - kernel_rmse_fe) / (linear_rmse_fe + 1e-12)) * 100
    improvement_mass = ((linear_rmse_mass - kernel_rmse_mass) / (linear_rmse_mass + 1e-12)) * 100

    print("\n📊 ПОРІВНЯННЯ ТА АНАЛІЗ НЕЛІНІЙНОСТІ")
    print("-" * 50)
    print("🎯 КЛЮЧОВІ РЕЗУЛЬТАТИ ДЛЯ ДИСЕРТАЦІЇ:")
    print(f"   💡 Покращення MSE: {improvement_mse:.1f}%")
    print(f"   💡 Покращення RMSE Fe: {improvement_fe:.1f}%")
    print(f"   💡 Покращення RMSE Mass: {improvement_mass:.1f}%")

    target_achieved = improvement_mse >= 15
    print(f"   {'✅' if target_achieved else '❌'} Цільовий діапазон "
          f"{'ДОСЯГНУТО' if target_achieved else 'НЕ досягнуто'}")

    nonlinearity_metrics = _analyze_simulation_nonlinearity(df_sim, true_gen)

    print("\n🔍 АНАЛІЗ НЕЛІНІЙНОСТІ ПРОЦЕСУ")
    print("-" * 40)
    for metric_name, value in nonlinearity_metrics.items():
        print(f"   📈 {metric_name}: {value:.3f}")

    # ----6. Візуалізація та збереження
    print("\n📊 ГЕНЕРАЦІЯ ВІЗУАЛІЗАЦІЙ...")
    figures = _create_comparison_visualizations(
        Y_test, Y_pred_linear, Y_pred_kernel,
        linear_mse, kernel_mse, improvement_mse,
        nonlinearity_metrics, df_sim
    )

    # Результати
    import json
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'simulation_params': simulation_params,
        'data_info': {
            'samples_total': len(df_sim),
            'samples_train': X_train_scaled.shape[0],
            'samples_val': X_val_scaled.shape[0],
            'samples_test': X_test_scaled.shape[0],
            'lag_used': simulation_params['lag'],
            'features': X_train_scaled.shape[1]
        },
        'linear_model': {
            'type': 'Linear (ARX)',
            'mse': linear_mse,
            'rmse_fe': linear_rmse_fe,
            'rmse_mass': linear_rmse_mass,
            'train_time': linear_train_time
        },
        'kernel_model': {
            'type': 'Kernel Ridge Regression (RBF)',
            'mse': kernel_mse,
            'rmse_fe': kernel_rmse_fe,
            'rmse_mass': kernel_rmse_mass,
            'train_time': kernel_train_time
        },
        'performance_comparison': {
            'mse_improvement_percent': improvement_mse,
            'rmse_fe_improvement_percent': improvement_fe,
            'rmse_mass_improvement_percent': improvement_mass,
            'target_achieved': target_achieved,
            'target_range': (15, 20)
        },
        'nonlinearity_analysis': nonlinearity_metrics
    }

    fname = f'dissertation_comparison_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(fname, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n✅ АНАЛІЗ ЗАВЕРШЕНО")
    print(f"📁 Результати збережено у {fname}")
    print("🖼️ Графіки збережено у PNG файли")

    return results

def make_anomaly_config_for_comparison(
    N_data: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    severity: str = "mild",
    include_train: bool = False
) -> dict:
    """
    Генерує reproducible anomaly_config для DataGenerator.generate_anomalies().
    За замовчуванням: аномалії тільки у val/test.
    severity: 'mild' | 'medium' | 'strong'
    """
    total = train_frac + val_frac + test_frac
    train_frac, val_frac, test_frac = train_frac/total, val_frac/total, test_frac/total

    train_end = int(train_frac * N_data)
    val_end   = train_end + int(val_frac * N_data)
    segments = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, N_data)}

    base_durations = {"spike": 1, "drift": 25, "drop": 20, "freeze": 15}
    sev_map = {"mild": (0.08, 0.15), "medium": (0.12, 0.22), "strong": (0.18, 0.30)}
    mag_lo, mag_hi = sev_map.get(severity, sev_map["mild"])

    rng = np.random.default_rng(seed)

    def seg_bounds(name):
        s, e = segments[name]
        return s, max(s, e-1), max(0, e-s)

    def pick_start(seg_name, dur):
        s, e_minus1, length = seg_bounds(seg_name)
        if length <= 0:
            return None
        dur = min(dur, length)
        hi = max(s, e_minus1 - (dur-1))
        return int(rng.integers(low=s, high=hi+1)), dur

    params = [
        "ore_mass_flow", "feed_fe_percent", "solid_feed_percent",
        "concentrate_fe_percent", "tailings_fe_percent",
        "concentrate_mass_flow", "tailings_mass_flow"
    ]
    cfg = {p: [] for p in params}

    def add_anom(param, seg, typ, mag=None, force_positive=False):
        dur = base_durations[typ]
        sd = pick_start(seg, dur)
        if sd is None:
            return
        start, dur = sd
        if typ == "freeze":
            cfg[param].append({"start": start, "duration": dur, "type": typ})
        else:
            m = float(abs(mag) if mag is not None else rng.uniform(mag_lo, mag_hi))
            if typ != "drop" and not force_positive and rng.random() < 0.5:
                m = -m
            if typ == "drop":
                m = abs(m)
            cfg[param].append({"start": start, "duration": dur, "magnitude": m, "type": typ})

    apply_train_here = include_train

    # План аномалій (val/test; train опц.)
    if apply_train_here: add_anom("ore_mass_flow", "train", "drift")
    add_anom("ore_mass_flow", "val",  "drift")
    add_anom("ore_mass_flow", "test", "spike")

    if apply_train_here: add_anom("solid_feed_percent", "train", "freeze")
    add_anom("solid_feed_percent", "val", "freeze")

    add_anom("feed_fe_percent", "test", "spike")
    add_anom("concentrate_mass_flow", "val", "drop", force_positive=True)
    add_anom("tailings_mass_flow", "test", "drift")
    add_anom("concentrate_fe_percent", "val", "spike")
    add_anom("tailings_fe_percent", "test", "freeze")

    return cfg

def create_simulation_data(reference_df: pd.DataFrame, params: dict):
    """
    Створення симуляційних даних через StatefulDataGenerator.
    Адаптовано: один і той самий anomaly_cfg для базових і нелінійних даних.
    """
    from data_gen import StatefulDataGenerator

    true_gen = StatefulDataGenerator(
        reference_df,
        ore_flow_var_pct=3.0,
        time_step_s=params['time_step_s'],
        time_constants_s=params['time_constants_s'],
        dead_times_s=params['dead_times_s'],
        true_model_type=params['plant_model_type'],
        seed=params['seed']
    )

    # 1) Аномалії (за замовчуванням увімкнено; у train — вимкнено)
    anomaly_cfg = None
    if params.get('use_anomalies', True):
        anomaly_cfg = make_anomaly_config_for_comparison(
            N_data=params['N_data'],
            train_frac=params.get('train_size', 0.8),
            val_frac=params.get('val_size', 0.1),
            test_frac=params.get('test_size', 0.1),
            seed=params['seed'],
            severity=params.get('anomaly_severity', 'mild'),
            include_train=params.get('anomaly_in_train', False),
        )

    # 2) Базові дані
    df_true_orig = true_gen.generate(
        T=params['N_data'],
        control_pts=params['control_pts'],
        n_neighbors=params['n_neighbors'],
        noise_level=params.get('noise_level', 'none'),
        anomaly_config=anomaly_cfg
    )

    # 3) Нелінійний варіант (ті самі аномалії)
    if params.get('enable_nonlinear', False):
        df_true = true_gen.generate_nonlinear_variant(
            base_df=df_true_orig,
            non_linear_factors=params['nonlinear_config'],
            noise_level='none',            # Щоб не «накладати» додатковий шум
            anomaly_config=anomaly_cfg
        )
    else:
        df_true = df_true_orig

    return true_gen, df_true
def _create_lagged_matrices_corrected(df, lag=2):
    """Створення лагових матриць для порівняння моделей"""
    
    # Використовуємо стандартні назви колонок з StatefulDataGenerator
    input_vars = ['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']
    output_vars = ['concentrate_fe', 'concentrate_mass']  # Скорочені назви з генератора
    
    # Перевірка альтернативних назв
    if 'concentrate_fe' not in df.columns and 'concentrate_fe_percent' in df.columns:
        output_vars = ['concentrate_fe_percent', 'concentrate_mass_flow']
    
    # Перевірка наявності колонок
    missing_vars = [var for var in input_vars + output_vars if var not in df.columns]
    if missing_vars:
        print(f"⚠️ Відсутні колонки: {missing_vars}")
        print(f"📋 Доступні колонки: {list(df.columns)}")
        # Використовуємо StatefulDataGenerator метод
        return StatefulDataGenerator.create_lagged_dataset(df, lags=lag)
    
    n = len(df)
    X, Y = [], []
    
    for i in range(lag, n):
        # Лагова структура для динамічних моделей
        row = []
        for var in input_vars:
            for j in range(lag + 1):  # від t до t-L
                row.append(df[var].iloc[i - j])
        X.append(row)
        
        # Вихідні змінні в момент t
        Y.append([df[var].iloc[i] for var in output_vars])
    
    return np.array(X), np.array(Y)


def _analyze_simulation_nonlinearity(df_sim, true_gen):
    """Аналіз нелінійності симуляційних даних"""
    
    metrics = {}
    
    # 1. Оцінка S-подібності через варіації градієнтів
    if 'concentrate_fe' in df_sim.columns:
        fe_values = df_sim['concentrate_fe'].values
        fe_gradients = np.diff(fe_values)
        metrics['fe_gradient_variance'] = np.var(fe_gradients)
        metrics['fe_gradient_skewness'] = pd.Series(fe_gradients).skew()
    
    # 2. Нелінійні взаємодії через кореляційний аналіз
    numeric_cols = df_sim.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 3:
        pearson_corr = df_sim[numeric_cols].corr(method='pearson')
        spearman_corr = df_sim[numeric_cols].corr(method='spearman')
        nonlinearity_indicator = abs(spearman_corr - pearson_corr).mean().mean()
        metrics['correlation_nonlinearity'] = nonlinearity_indicator
    
    # 3. Ентропійна оцінка складності
    if 'solid_feed_percent' in df_sim.columns:
        control_changes = np.abs(np.diff(df_sim['solid_feed_percent']))
        control_entropy = -np.sum((control_changes + 1e-10) * np.log(control_changes + 1e-10))
        metrics['control_complexity'] = control_entropy
    
    # 4. Характеристики розподілу
    if 'concentrate_mass' in df_sim.columns:
        mass_values = df_sim['concentrate_mass'].values
        metrics['mass_distribution_kurtosis'] = pd.Series(mass_values).kurtosis()
        metrics['mass_distribution_skewness'] = pd.Series(mass_values).skew()
    
    return metrics


# Залишаємо оригінальні методи візуалізації та висновків без змін
def _create_comparison_visualizations(Y_test, Y_pred_linear, Y_pred_kernel, 
                                    linear_mse, kernel_mse, improvement, 
                                    nonlinearity_metrics, full_df):
    """Створення комплексних візуалізацій для порівняння моделей"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Налаштування стилю
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10
    
    figures = {}
    
    # === ОСНОВНА ФІГУРА: ПОРІВНЯННЯ МОДЕЛЕЙ ===
    fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig1.suptitle('Порівняння лінійної та ядерної моделей для дисертації', fontsize=16, fontweight='bold')
    
    # 1.1 Scatter plot для Fe концентрації
    ax = axes[0, 0]
    ax.scatter(Y_test[:, 0], Y_pred_linear[:, 0], alpha=0.6, s=20, color='red', label='Лінійна модель')
    ax.scatter(Y_test[:, 0], Y_pred_kernel[:, 0], alpha=0.6, s=20, color='green', label='Ядерна модель')
    
    # Ідеальна лінія
    min_val, max_val = Y_test[:, 0].min(), Y_test[:, 0].max()
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Ідеальна лінія')
    
    ax.set_xlabel('Реальна концентрація Fe (%)')
    ax.set_ylabel('Прогнозована концентрація Fe (%)')
    ax.set_title('Прогнозування концентрації Fe')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Додавання R² на графік
    r2_linear = 1 - np.sum((Y_test[:, 0] - Y_pred_linear[:, 0])**2) / np.sum((Y_test[:, 0] - np.mean(Y_test[:, 0]))**2)
    r2_kernel = 1 - np.sum((Y_test[:, 0] - Y_pred_kernel[:, 0])**2) / np.sum((Y_test[:, 0] - np.mean(Y_test[:, 0]))**2)
    ax.text(0.05, 0.95, f'R² лінійна: {r2_linear:.3f}\nR² ядерна: {r2_kernel:.3f}', 
            transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 1.2 Scatter plot для масового потоку
    ax = axes[0, 1]
    ax.scatter(Y_test[:, 1], Y_pred_linear[:, 1], alpha=0.6, s=20, color='red', label='Лінійна модель')
    ax.scatter(Y_test[:, 1], Y_pred_kernel[:, 1], alpha=0.6, s=20, color='green', label='Ядерна модель')
    
    min_val, max_val = Y_test[:, 1].min(), Y_test[:, 1].max()
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Ідеальна лінія')
    
    ax.set_xlabel('Реальний масовий потік (т/год)')
    ax.set_ylabel('Прогнозований масовий потік (т/год)')
    ax.set_title('Прогнозування масового потоку')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.3 Порівняння MSE
    ax = axes[0, 2]
    models = ['Лінійна\n(ARX)', 'Ядерна\n(KRR)']
    mse_values = [linear_mse, kernel_mse]
    colors = ['red', 'green']
    
    bars = ax.bar(models, mse_values, color=colors, alpha=0.7, width=0.6)
    ax.set_ylabel('MSE')
    ax.set_title(f'Порівняння MSE\n(покращення: {improvement:.1f}%)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Додавання значень на стовпці
    for bar, value in zip(bars, mse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.02,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Додавання стрілки покращення
    if improvement > 0:
        ax.annotate('', xy=(1, kernel_mse), xytext=(0, linear_mse),
                   arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
        ax.text(0.5, (linear_mse + kernel_mse)/2, f'-{improvement:.1f}%', 
               ha='center', va='center', color='blue', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 1.4 Часовий ряд помилок для Fe
    ax = axes[1, 0]
    time_steps = range(len(Y_test))
    error_linear_fe = Y_test[:, 0] - Y_pred_linear[:, 0]
    error_kernel_fe = Y_test[:, 0] - Y_pred_kernel[:, 0]
    
    ax.plot(time_steps, error_linear_fe, color='red', alpha=0.7, linewidth=1, label='Помилка лінійної')
    ax.plot(time_steps, error_kernel_fe, color='green', alpha=0.7, linewidth=1, label='Помилка ядерної')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Крок тестування')
    ax.set_ylabel('Помилка прогнозу Fe (%)')
    ax.set_title('Динаміка помилок прогнозування Fe')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.5 Розподіл помилок
    ax = axes[1, 1]
    ax.hist(error_linear_fe, bins=30, alpha=0.6, color='red', label='Лінійна модель', density=True)
    ax.hist(error_kernel_fe, bins=30, alpha=0.6, color='green', label='Ядерна модель', density=True)
    
    ax.set_xlabel('Помилка прогнозу Fe (%)')
    ax.set_ylabel('Щільність розподілу')
    ax.set_title('Розподіл помилок прогнозування')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Статистики помилок
    ax.text(0.02, 0.98, 
           f'Лінійна:\nСТД: {np.std(error_linear_fe):.3f}\nСер.: {np.mean(error_linear_fe):.3f}\n\n'
           f'Ядерна:\nСТД: {np.std(error_kernel_fe):.3f}\nСер.: {np.mean(error_kernel_fe):.3f}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 1.6 Метрики нелінійності
    ax = axes[1, 2]
    if nonlinearity_metrics:
        metric_names = list(nonlinearity_metrics.keys())
        metric_values = list(nonlinearity_metrics.values())
        
        # Скорочення назв для кращого відображення
        short_names = []
        for name in metric_names:
            if 'gradient' in name:
                short_names.append('Градієнт\nваріації')
            elif 'correlation' in name:
                short_names.append('Кореляційна\nнелінійність')
            elif 'complexity' in name:
                short_names.append('Складність\nкерування')
            elif 'kurtosis' in name:
                short_names.append('Куртозис\nрозподілу')
            elif 'skewness' in name:
                short_names.append('Асиметрія\nрозподілу')
            else:
                short_names.append(name[:10] + '...' if len(name) > 10 else name)
        
        bars = ax.bar(range(len(metric_values)), metric_values, color='orange', alpha=0.7)
        ax.set_xticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Значення метрики')
        ax.set_title('Характеристики нелінійності процесу')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Додавання значень на стовпці
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.02,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Метрики нелінійності\nне доступні', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Характеристики нелінійності процесу')
    
    plt.tight_layout()
    plt.savefig('dissertation_model_comparison.png', dpi=300, bbox_inches='tight')
    figures['main_comparison'] = fig1
    
    # === ДОДАТКОВА ФІГУРА: ДЕТАЛЬНИЙ АНАЛІЗ ===
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Детальний аналіз продуктивності моделей', fontsize=14, fontweight='bold')
    
    # 2.1 Box plot помилок
    ax = axes[0, 0]
    error_data = [error_linear_fe, error_kernel_fe]
    bp = ax.boxplot(error_data, labels=['Лінійна', 'Ядерна'], patch_artist=True)
    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][1].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_alpha(0.6)
    
    ax.set_ylabel('Помилка прогнозу Fe (%)')
    ax.set_title('Розподіл помилок (квартилі)')
    ax.grid(True, alpha=0.3)
    
    # 2.2 Кумулятивний розподіл помилок
    ax = axes[0, 1]
    sorted_linear = np.sort(np.abs(error_linear_fe))
    sorted_kernel = np.sort(np.abs(error_kernel_fe))
    
    y_linear = np.arange(1, len(sorted_linear) + 1) / len(sorted_linear)
    y_kernel = np.arange(1, len(sorted_kernel) + 1) / len(sorted_kernel)
    
    ax.plot(sorted_linear, y_linear, color='red', linewidth=2, label='Лінійна модель')
    ax.plot(sorted_kernel, y_kernel, color='green', linewidth=2, label='Ядерна модель')
    
    ax.set_xlabel('Абсолютна помилка Fe (%)')
    ax.set_ylabel('Кумулятивна ймовірність')
    ax.set_title('Кумулятивний розподіл помилок')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2.3 Кореляція залишків
    ax = axes[1, 0]
    ax.scatter(error_linear_fe, error_kernel_fe, alpha=0.6, s=20, color='purple')
    
    # Лінія кореляції
    correlation = np.corrcoef(error_linear_fe, error_kernel_fe)[0, 1]
    ax.plot([error_linear_fe.min(), error_linear_fe.max()], 
           [error_kernel_fe.min(), error_kernel_fe.max()], 'r--', alpha=0.8)
    
    ax.set_xlabel('Помилка лінійної моделі (%)')
    ax.set_ylabel('Помилка ядерної моделі (%)')
    ax.set_title(f'Кореляція помилок (r = {correlation:.3f})')
    ax.grid(True, alpha=0.3)
    
    # 2.4 Покращення за квартилями
    ax = axes[1, 1]
    quartiles = [25, 50, 75, 90, 95]
    linear_percentiles = np.percentile(np.abs(error_linear_fe), quartiles)
    kernel_percentiles = np.percentile(np.abs(error_kernel_fe), quartiles)
    improvements = ((linear_percentiles - kernel_percentiles) / linear_percentiles) * 100
    
    bars = ax.bar(range(len(quartiles)), improvements, color='blue', alpha=0.7)
    ax.set_xticks(range(len(quartiles)))
    ax.set_xticklabels([f'{q}%' for q in quartiles])
    ax.set_ylabel('Покращення (%)')
    ax.set_xlabel('Квартиль помилок')
    ax.set_title('Покращення за квартилями помилок')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Додавання значень на стовпці
    for bar, value in zip(bars, improvements):
        color = 'green' if value > 0 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(improvements)*0.02,
               f'{value:.1f}%', ha='center', va='bottom', color=color, fontweight='bold')
    
    plt.tight_layout()
    # plt.savefig('dissertation_detailed_analysis.png', dpi=300, bbox_inches='tight')
    figures['detailed_analysis'] = fig2
    plt.show()
    
    print("📊 Створено комплексні візуалізації:")
    print("   📈 dissertation_model_comparison.png - основне порівняння")
    print("   📊 dissertation_detailed_analysis.png - детальний аналіз")
    
    return figures

if __name__ == '__main__':

    reference_df = pd.read_parquet("processed.parquet")
    
    # df_summary = run_experiment_suite(
    #     reference_df= reference_df,
    #     lags=(1,2,3,4),
    #     Ns=(5000, 7000),
    #     krr_kernels=("linear", "rbf"),
    #     anomaly_severity="mild",
    #     use_anomalies=True,
    #     seeds=(123, 42),
    #     noise_level="none",
    #     results_dir="exp_results",
    #     n_iter_search_rbf=15,
    #     verbose=False,
    # )

    # df_summary = run_experiment_suite(
    #     reference_df= reference_df,
    #     lags=(2,3),
    #     Ns=(5000, 7000),
    #     krr_kernels=("linear", "rbf"),
    #     anomaly_severity="mild",
    #     use_anomalies=True,
    #     seeds=(123, 42),
    #     noise_level="none",
    #     results_dir="exp_results",
    #     n_iter_search_rbf=15,
    #     verbose=False,
    # )

    # print(df_summary)

    # res = analyze_experiment_results(
    #     results_dir="exp_results",
    #     kernel_focus=["rbf","linear"],  # можна залишити None, щоб взяти всі
    #     do_ttests=True,
    #     save_figs=True,
    #     show_figs=True,                # увімкни True, якщо хочеш показ у ноутбуці
    #     export_tables=True
    # )
    
    # print("Графіки:", *res["fig_paths"], sep="\n - ")
    # print("Таблиці:", res["table_paths"])
    # print("Meta:", res["meta"])
    
    # # Наприклад, подивитись ТОП по гейну Fe (rbf):
    # (df := res["df_summary"])
    # print(df[df.kernel=="rbf"].sort_values("RMSE_Fe_gain_%", ascending=False).head(10))    
    
    compare_linear_vs_kernel_models()