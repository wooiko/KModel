# model.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Type
import inspect

# --- sklearn ---
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, Sum, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# ======================================================================
#                     БАЗОВИЙ ІНТЕРФЕЙС СТРАТЕГІЇ
# ======================================================================
class _BaseKernelModel(ABC):
    """Абстрактний клас-стратегія. Визначає обов’язкові методи та kernel із setter."""

    def __init__(self):
        self._kernel: str | None = None

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def linearize(self, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @property
    def kernel(self) -> str | None:
        return self._kernel

    @kernel.setter
    def kernel(self, value: str) -> None:
        self._kernel = value


# ======================================================================
#                    РЕАЛІЗАЦІЯ ДЛЯ Kernel Ridge
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
        self.kernel = kernel.lower()
        self.alpha = alpha
        self.gamma = gamma
        self.find_optimal_params = find_optimal_params
        self.n_iter_random_search = n_iter_random_search

        # наповнюються після fit
        self.model: KernelRidge | None = None
        self.X_train_: np.ndarray | None = None
        self.dual_coef_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        # Вибір із RandomizedSearch або пряма ініціалізація
        if self.find_optimal_params:
            self.model = self._run_random_search(X, Y)
        else:
            gamma_eff = (
                self.gamma
                if self.gamma is not None
                else (
                    self._calculate_median_heuristic_gamma(X)
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
            raise RuntimeError("Модель ще не навчена.")
        return self.model.predict(X)

    def linearize(self, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X0.ndim == 1:
            X0 = X0.reshape(1, -1)

        if self.kernel == "linear":
            return self.coef_, self.intercept_

        if self.kernel != "rbf":
            raise NotImplementedError(
                f"Лінеаризація KRR доступна тільки для 'linear' та 'rbf', отримано '{self.kernel}'."
            )

        gamma_eff = (
            getattr(self.model, "gamma", None)
            or self._calculate_median_heuristic_gamma(self.X_train_)
        )

        diffs = X0[:, None, :] - self.X_train_[None, :, :]
        sq_diffs = np.sum(diffs**2, axis=-1)
        K_row = np.exp(-gamma_eff * sq_diffs)
        dK_dX = -2 * gamma_eff * diffs * K_row[..., None]

        W_local = np.einsum("ijk,ji->ki", dK_dX, self.dual_coef_)
        y0 = self.predict(X0)
        b_local = (y0 - X0 @ W_local).flatten()

        W_local = np.clip(W_local, -1e3, 1e3)
        b_local = np.clip(b_local, -1e3, 1e3)
        return W_local, b_local

    def _run_random_search(self, X: np.ndarray, Y: np.ndarray) -> KernelRidge:
        base = KernelRidge(kernel=self.kernel)
        if self.kernel == "linear":
            param_distributions = {"alpha": loguniform(0.001, 100)}
        elif self.kernel == "rbf":
            param_distributions = {
                "alpha": loguniform(0.01, 100),
                "gamma": loguniform(0.001, 10),
            }
        else:
            raise ValueError(f"RandomizedSearch не підтримує ядро '{self.kernel}'.")

        rs = RandomizedSearchCV(
            base,
            param_distributions,
            n_iter=self.n_iter_random_search,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        rs.fit(X, Y)
        print(f"-> Найкращі параметри KRR: {rs.best_params_}")
        return rs.best_estimator_

    @staticmethod
    def _calculate_median_heuristic_gamma(X: np.ndarray) -> float:
        subset = X
        if X.shape[0] > 1000:
            rng = np.random.default_rng(42)
            subset = X[rng.choice(X.shape[0], size=1000, replace=False)]
        d2 = np.sum((subset[:, None] - subset[None, :]) ** 2, axis=2)
        upper = d2[np.triu_indices_from(d2, k=1)]
        median_sq = np.median(upper) if upper.size else 1.0
        return 1.0 / max(median_sq, 1e-9)


# ======================================================================
#                РЕАЛІЗАЦІЯ ДЛЯ Gaussian Process Regressor
# ======================================================================
class _GPRModel(_BaseKernelModel):
    def __init__(self):
        super().__init__()
        self.models: list[GaussianProcessRegressor] = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        n_targets = Y.shape[1]
        base_kernel = (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(0.1, 20.0))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        )
        self.models.clear()
        for i in range(n_targets):
            gpr = GaussianProcessRegressor(
                kernel=base_kernel,
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
            X0 = X0.reshape(1, -1)

        W_cols, b_elems = [], []
        for gpr in self.models:
            rbf_kernel = self._find_rbf_kernel(gpr.kernel_)
            if rbf_kernel is None:
                raise TypeError("Не знайдено RBF-компонент у ядрі GPR.")
            gamma = 1.0 / (2 * rbf_kernel.length_scale**2)

            Xtr = gpr.X_train_
            alpha = gpr.alpha_

            diffs = X0[:, None, :] - Xtr[None, :, :]
            sq_diffs = np.sum(diffs**2, axis=-1)
            K_row = np.exp(-gamma * sq_diffs)
            dK_dX = -2 * gamma * diffs * K_row[..., None]

            W_col = np.einsum("ji,j->i", dK_dX.squeeze(0), alpha.flatten()).reshape(-1, 1)
            y0 = gpr.predict(X0)
            b_col = (y0 - X0 @ W_col).flatten()

            W_cols.append(W_col)
            b_elems.append(b_col)

        W_local = np.hstack(W_cols)
        b_local = np.hstack(b_elems)
        return W_local, b_local

    @staticmethod
    def _find_rbf_kernel(kernel):
        if isinstance(kernel, RBF):
            return kernel
        if isinstance(kernel, (Product, Sum)):
            return _GPRModel._find_rbf_kernel(kernel.k1) or _GPRModel._find_rbf_kernel(kernel.k2)
        return None


# ======================================================================
#                         ФАСАД - KernelModel
# ======================================================================
class KernelModel:
    """
    Фасад. Інкапсулює конкретну реалізацію (_KRRModel, _GPRModel, …)
    та делегує всі виклики через механізм __getattr__.
    """

    _REGISTRY: Dict[str, Type[_BaseKernelModel]] = {
        "krr": _KRRModel,
        "gpr": _GPRModel,
        # для SVR: додати 'svr': _SVRModel
    }

    def __init__(self, model_type: str = "krr", **kwargs):
        mtype = model_type.lower()
        if mtype not in self._REGISTRY:
            raise ValueError(f"Невідома модель '{model_type}'")
        impl_cls = self._REGISTRY[mtype]

        # відфільтрувати тільки ті kwargs, що є в __init__ імплементації
        sig = inspect.signature(impl_cls.__init__)
        impl_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters and k != 'self'
        }

        self._impl = impl_cls(**impl_kwargs)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        return self._impl.fit(X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._impl.predict(X)

    def linearize(self, X0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._impl.linearize(X0)

    def __getattr__(self, item):
        return getattr(self._impl, item)