# model.py

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# Новий імпорт для пошуку по сітці
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