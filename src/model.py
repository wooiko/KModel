# model.py

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class KernelModel:
    """
    Універсальний обгорток для ядрових моделей:
    - Kernel Ridge Regression з RBF-ядром
    - Gaussian Process Regression з RBF-ядром

    Підтримує мультивихід: у випадку GPR будує по одному GaussianProcessRegressor
    на кожен вихідний канал.
    """

    def __init__(self,
                 model_type: str = 'krr',
                 alpha: float = 1.0,
                 gamma: float = None):
        """
        Args:
            model_type: 'krr' або 'gpr'
            alpha: регуляризаційний параметр (для KRR та шуму в GPR)
            gamma: параметр RBF-ядра для KRR (1 / (n_features * X.var()) за замовчуванням)
        """
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.gamma = gamma
        self.models = None

        if self.model_type not in ('krr', 'gpr'):
            raise ValueError(f"Невідомий model_type '{model_type}'. Виберіть 'krr' або 'gpr'.")

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Навчання моделі на X, Y.
        X.shape = (n_samples, n_features)
        Y.shape = (n_samples, n_targets)
        """
        if self.model_type == 'krr':
            # одна модель, підтримує мультивихід
            self.models = KernelRidge(
                alpha=self.alpha,
                kernel='rbf',
                gamma=self.gamma
            )
            self.models.fit(X, Y)

        else:  # 'gpr'
            # окремий GPR на кожен вихідний канал
            n_targets = Y.shape[1]
            base_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0,
                                                   length_scale_bounds=(1e-3, 1e3))
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
        """
        Прогноз Ŷ для вхідного X.
        Повертає масив форми (n_samples, n_targets).
        """
        if self.models is None:
            raise RuntimeError("Модель не навчена. Викличте спочатку fit().")

        if self.model_type == 'krr':
            return self.models.predict(X)

        # gpr: збираємо прогнози з кожної моделі в стовпці
        preds = [gpr.predict(X) for gpr in self.models]
        return np.vstack(preds).T