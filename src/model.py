# model.py

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class KernelModel:
    def __init__(
        self,
        model_type: str = 'krr',   # 'krr' або 'gpr'
        kernel: str     = 'rbf',   # 'rbf' або 'linear' (тільки для krr)
        alpha: float    = 1.0,
        gamma: float    = None
    ):
        self.model_type = model_type.lower()
        self.kernel     = kernel
        self.alpha      = alpha
        self.gamma      = gamma

        # Після fit:
        self.models     = None        # KernelRidge або список GPR
        self.X_train_   = None
        self.dual_coef_ = None
        self.coef_      = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        if self.model_type == 'krr':
            n_feats = X.shape[1]

            # Вибір ядра
            if self.kernel == 'rbf':
                if self.gamma is None:
                    self.gamma = 1.0 / (n_feats * X.var())
                krr = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=self.gamma)
            elif self.kernel == 'linear':
                krr = KernelRidge(alpha=self.alpha, kernel='linear')
            else:
                raise ValueError(f"Невідоме ядро '{self.kernel}' для KRR.")

            # Навчання
            krr.fit(X, Y)

            # Зберігаємо основні атрибути
            self.models     = krr
            self.X_train_   = X.copy()
            self.dual_coef_ = krr.dual_coef_   # shape = (n_samples, n_targets)

            if self.kernel == 'linear':
                # Обчислюємо ваги прямим методом: w = X^T · dual_coef_
                # (n_features, n_samples)·(n_samples, n_targets) → (n_features, n_targets)
                self.coef_      = X.T.dot(self.dual_coef_)
                # У KernelRidge за замовчуванням немає інтерсепта
                self.intercept_ = np.zeros(Y.shape[1])
            else:
                # RBF-ядро: будемо в прогнозі використовувати dual_coef_ та X_train_
                self.intercept_ = np.zeros(Y.shape[1])

        else:
            # Gaussian Process для кожної компоненти виходу
            n_targets = Y.shape[1]
            base_kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
            self.models = []
            for i in range(n_targets):
                gpr = GaussianProcessRegressor(
                    kernel=base_kernel,
                    alpha=self.alpha,
                    normalize_y=True
                )
                gpr.fit(X, Y[:, i])
                self.models.append(gpr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_type == 'krr':
            if self.kernel == 'linear':
                # Y = X · w + b
                return X.dot(self.coef_) + self.intercept_
            else:
                # RBF-KRR: прогноз через ядро
                diffs = np.sum((X[:, None, :] - self.X_train_[None, :, :])**2, axis=2)
                K     = np.exp(-self.gamma * diffs)
                return K.dot(self.dual_coef_) + self.intercept_
        else:
            # GPR: стекуємо прогнози по кожному виходу
            Ys = [gpr.predict(X) for gpr in self.models]
            return np.vstack(Ys).T