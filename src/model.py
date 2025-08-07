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
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
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

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
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

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        n_targets = Y.shape[1]
        self.models.clear()
        self.X_train_ = X.copy()
    
        for k in range(n_targets):
            y = Y[:, k]
    
            if self.find_optimal_params:
                mdl = self._run_random_search(X, y)
            else:
                # ✅ ВИПРАВЛЕНА логіка gamma
                if self.kernel == "rbf":
                    if self.gamma is not None:
                        gamma_eff = self.gamma
                    else:
                        # ✅ ПРАВИЛЬНА sklearn формула для "scale"
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
                
                # ✅ Зберігаємо точне gamma для linearize()
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

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Навчання лінійної моделі"""
        
        # 🔧 Створення поліноміальних ознак якщо потрібно
        if self.poly_degree > 1:
            self.poly_features = PolynomialFeatures(
                degree=self.poly_degree,
                include_bias=False  # bias додамо в модель
            )
            X_features = self.poly_features.fit_transform(X)
        else:
            self.poly_features = None
            X_features = X
            
        # 🎯 Вибір та налаштування моделі
        if self.find_optimal_params:
            self.model = self._run_random_search(X_features, Y)
        else:
            if self.linear_type == "ols":
                self.model = LinearRegression(fit_intercept=self.include_bias)
            elif self.linear_type == "ridge":
                self.model = Ridge(alpha=self.alpha, fit_intercept=self.include_bias)
            elif self.linear_type == "lasso":
                self.model = Lasso(alpha=self.alpha, fit_intercept=self.include_bias, max_iter=2000)
        
        # 🚀 Навчання
        self.model.fit(X_features, Y)
        
        # 📊 Зберігаємо коефіцієнти для швидкого доступу
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
        "linear": _LinearModel,  # 🆕 ДОДАНО L-MPC!

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

    # API – просто делегуємо
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        return self._impl.fit(X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._impl.predict(X)

    def linearize(self, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._impl.linearize(X0)

    def __getattr__(self, item):
        return getattr(self._impl, item)