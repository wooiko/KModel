# model.py

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import GridSearchCV


class KernelModel:
    def __init__(
        self,
        model_type: str = 'krr',
        kernel: str = 'linear',
        alpha: float = 1.0,
        gamma: float = None,
        # >>> Новий параметр для активації пошуку
        find_optimal_params: bool = False
    ):
        self.model_type = model_type.lower()
        self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        # >>> Зберігаємо прапорець
        self.find_optimal_params = find_optimal_params

        # Після fit:
        self.models = None
        self.X_train_ = None
        self.dual_coef_ = None
        self.coef_ = None
        self.intercept_ = None


    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Навчає модель. Якщо find_optimal_params=True, виконує пошук по сітці
        для KernelRidge моделі перед фінальним навчанням.
        """
        # --- Логіка для Гаусівських процесів (залишається без змін) ---
        if self.model_type == 'gpr':
            n_targets = Y.shape[1]
            base_kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
            self.models = []
            for i in range(n_targets):
                gpr = GaussianProcessRegressor(
                    kernel=base_kernel, alpha=self.alpha, normalize_y=True
                )
                gpr.fit(X, Y[:, i])
                self.models.append(gpr)
            return

        # --- Логіка для KernelRidge ---
        krr_model = None  # Тут буде зберігатись фінальна навчена модель

        if self.find_optimal_params and self.model_type == 'krr':
            # --- Блок автоматичного підбору гіперпараметрів ---
            print(f"Запуск пошуку оптимальних гіперпараметрів для KernelRidge (ядро: {self.kernel})...")
            
            base_krr = KernelRidge(kernel=self.kernel)
            
            # Визначаємо сітку параметрів для пошуку
            if self.kernel == 'linear':
                param_grid = {'alpha': np.logspace(-3, 2, 6)}  # [0.001, 0.01, ..., 100]
            elif self.kernel == 'rbf':
                param_grid = {
                    'alpha': np.logspace(-2, 2, 5),  # [0.01, 0.1, 1, 10, 100]
                    'gamma': np.logspace(-3, 1, 5)   # [0.001, 0.01, 0.1, 1, 10]
                }
            else:
                raise ValueError(f"Пошук параметрів не підтримується для ядра '{self.kernel}'.")

            # Створюємо та запускаємо GridSearchCV
            # cv=3 означає 3-кратну перехресну валідацію
            # n_jobs=-1 використовує всі доступні ядра процесора для прискорення
            grid_search = GridSearchCV(base_krr, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X, Y)

            print(f"-> Найкращі параметри знайдено: {grid_search.best_params_}")
            
            # Найкраща модель, знайдена в процесі пошуку
            krr_model = grid_search.best_estimator_

            # Оновлюємо атрибути екземпляра знайденими оптимальними значеннями
            self.alpha = krr_model.alpha
            if hasattr(krr_model, 'gamma'):
                self.gamma = krr_model.gamma
        
        else:
            # --- Стара логіка: використання параметрів, заданих вручну ---
            gamma_to_use = self.gamma
            if self.kernel == 'rbf' and self.gamma is None:
                n_feats = X.shape[1]
                gamma_to_use = 1.0 / (n_feats * X.var()) if X.var() > 1e-9 else 1.0
            
            krr_model = KernelRidge(alpha=self.alpha, kernel=self.kernel, gamma=gamma_to_use)
            krr_model.fit(X, Y)

        # --- Загальні кроки після навчання для будь-якої KRR моделі ---
        self.models = krr_model
        self.X_train_ = X.copy()
        self.dual_coef_ = self.models.dual_coef_

        if self.kernel == 'linear':
            self.coef_ = X.T.dot(self.dual_coef_)
            self.intercept_ = np.zeros(Y.shape[1])
        else:  # Для RBF та інших нелінійних ядер коефіцієнти не розраховуємо
            self.coef_ = None
            self.intercept_ = np.zeros(Y.shape[1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_type == 'gpr':
            Ys = [gpr.predict(X) for gpr in self.models]
            return np.vstack(Ys).T
        
        # Для KRR використовуємо вбудований метод predict навченої моделі
        return self.models.predict(X)
    
    def linearize(self, X0: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Обчислює лінійну апроксимацію моделі y ≈ Wx + b навколо точки X0.
        Повертає локальну матрицю ваг W та зсув b.
        """
        if self.model_type != 'krr':
            raise NotImplementedError("Лінеаризація реалізована тільки для KRR.")

        # Якщо модель вже лінійна, просто повертаємо її глобальні параметри
        if self.kernel == 'linear':
            # intercept_ тут нульовий згідно з вашою реалізацією
            return self.coef_, self.intercept_ 

        if self.kernel == 'rbf':
            # Для RBF-ядра y(X) = K(X, X_train) @ dual_coef_
            # W = d(y)/d(X) | в точці X0
            
            # Переконуємось, що X0 має правильну форму (1, n_features)
            if X0.ndim == 1:
                X0 = X0.reshape(1, -1)
            
            # Обчислення Якобіана (градієнта)
            # d(K_ij)/d(X_i) = -2 * gamma * (X_i - X_train_j) * K_ij
            diffs = X0[:, None, :] - self.X_train_[None, :, :]
            sq_diffs = np.sum(diffs**2, axis=-1)
            K_row = np.exp(-self.gamma * sq_diffs)
            
            # Градієнт ядра по відношенню до X0
            # (n_targets, n_samples, n_features)
            dK_dX = -2 * self.gamma * diffs * K_row[..., None]
            
            # Локальна матриця ваг W (Якобіан)
            # W_ji = sum_k(dK_ik/dX_j * dual_coef_k) -> W_ij = sum_k(dK_ik/dX_i * dual_coef_k)
            # (n_features, n_targets)
            W_local = np.einsum('ijk,ji->ki', dK_dX, self.dual_coef_)
            
            # Обчислення локального зсуву b, щоб апроксимація була точною в точці X0
            # y0 = W_local * X0 + b_local  =>  b_local = y0 - W_local * X0
            y0 = self.predict(X0)
            b_local = y0 - X0 @ W_local
            
            return W_local, b_local.flatten()

        raise NotImplementedError(f"Лінеаризація для ядра '{self.kernel}' не реалізована.")