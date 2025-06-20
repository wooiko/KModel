# model.py

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, Sum, ConstantKernel as C
from sklearn.model_selection import RandomizedSearchCV # Changed from GridSearchCV
from scipy.stats import loguniform # For defining distributions in RandomizedSearchCV


class KernelModel:
    def __init__(
        self,
        model_type: str = 'krr',
        kernel: str = 'linear',
        alpha: float = 1.0,
        gamma: float = None,
        find_optimal_params: bool = False,
        n_iter_random_search: int = 20 # New parameter for RandomizedSearchCV iterations
    ):
        self.model_type = model_type.lower()
        self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        self.find_optimal_params = find_optimal_params
        self.n_iter_random_search = n_iter_random_search # Store the new parameter

        # Attributes after fit:
        self.models = None
        self.X_train_ = None
        self.dual_coef_ = None
        self.coef_ = None
        self.intercept_ = None


    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Навчає модель. Якщо find_optimal_params=True, виконує рандомізований пошук
        гіперпараметрів для KernelRidge моделі перед фінальним навчанням.
        """
        # --- Logic for Gaussian Processes (remains unchanged) ---
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

        # --- Logic for KernelRidge ---
        krr_model = None

        if self.find_optimal_params and self.model_type == 'krr':
            print(f"Запуск рандомізованого пошуку гіперпараметрів для KernelRidge (ядро: {self.kernel})...")
            
            base_krr = KernelRidge(kernel=self.kernel)
            
            # Define parameter distributions for RandomizedSearchCV
            if self.kernel == 'linear':
                param_distributions = {'alpha': loguniform(0.001, 100)}  
            elif self.kernel == 'rbf':
                param_distributions = {
                    'alpha': loguniform(0.01, 100),
                    'gamma': loguniform(0.001, 10)   
                }
            else:
                raise ValueError(f"Пошук параметрів не підтримується для ядра '{self.kernel}'.")

            # Create and run RandomizedSearchCV
            random_search = RandomizedSearchCV(
                base_krr, 
                param_distributions, 
                n_iter=self.n_iter_random_search, # Number of parameter settings that are sampled
                cv=3, 
                scoring='neg_mean_squared_error', 
                random_state=42, # For reproducibility
                n_jobs=-1,
                verbose=1 # More verbose output during search
            )
            random_search.fit(X, Y)

            print(f"-> Найкращі параметри знайдено: {random_search.best_params_}")
            
            # The best model found during the search
            krr_model = random_search.best_estimator_

            # Update instance attributes with the found optimal values
            self.alpha = krr_model.alpha
            if hasattr(krr_model, 'gamma'): # gamma might not exist for linear kernel
                self.gamma = krr_model.gamma
        
        else:
            # --- Original logic: use manually specified parameters or heuristic gamma ---
            gamma_to_use = self.gamma
            if self.kernel == 'rbf' and self.gamma is None:
                # Use the median-heuristic for gamma if not specified
                gamma_to_use = self._calculate_median_heuristic_gamma(X)
            
            krr_model = KernelRidge(alpha=self.alpha, kernel=self.kernel, gamma=gamma_to_use)
            krr_model.fit(X, Y)

        # --- Common steps after training for any KRR model ---
        self.models = krr_model
        self.X_train_ = X.copy()
        self.dual_coef_ = self.models.dual_coef_

        if self.kernel == 'linear':
            # For linear kernel, compute explicit coefficients and intercept
            self.coef_ = X.T.dot(self.dual_coef_)
            self.intercept_ = np.zeros(Y.shape[1])
        else:  # For RBF and other non-linear kernels, coefficients are not directly calculated this way
            self.coef_ = None
            self.intercept_ = np.zeros(Y.shape[1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_type == 'gpr':
            Ys = [gpr.predict(X) for gpr in self.models]
            return np.vstack(Ys).T
        
        # For KRR, use the predict method of the trained model
        return self.models.predict(X)
    
    def linearize(self, X0: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Обчислює лінійну апроксимацію моделі y ≈ Wx + b навколо точки X0.
        Повертає локальну матрицю ваг W та зсув b.
        Працює для KRR та GPR.
        """
        # Ensure X0 has the correct shape (1, n_features)
        if X0.ndim == 1:
            X0 = X0.reshape(1, -1)
            
        # --- Linearization for Kernel Ridge Regression ---
        if self.model_type == 'krr':
            if self.kernel == 'linear':
                # For linear kernel, W and b are constant
                return self.coef_, self.intercept_

            if self.kernel == 'rbf':
                # Analytical gradient for RBF kernel (as before)
                # Ensure gamma is correctly used, either found by search or heuristic
                current_gamma = self.gamma if self.gamma is not None else \
                                self._calculate_median_heuristic_gamma(self.X_train_) # Fallback if gamma somehow not set

                diffs = X0[:, None, :] - self.X_train_[None, :, :]
                sq_diffs = np.sum(diffs**2, axis=-1)
                K_row = np.exp(-current_gamma * sq_diffs)
                
                # Gradient of K with respect to X0
                dK_dX = -2 * current_gamma * diffs * K_row[..., None]
                # Local weights W_local = dY/dX0 = (dY/dK) * (dK/dX0)
                # dual_coef_ is dY/dK (in essence, up to a scalar factor)
                W_local = np.einsum('ijk,ji->ki', dK_dX, self.dual_coef_)
                
                # Calculate local intercept b_local = y0 - W_local * X0
                y0 = self.predict(X0)
                b_local = y0 - X0 @ W_local
                
                return W_local, b_local.flatten()

            raise NotImplementedError(f"Лінеаризація для KRR з ядром '{self.kernel}' не реалізована.")

        elif self.model_type == 'gpr':
            W_columns = []
            b_elements = []

            # Iterate through each GPR model (for each output target)
            for gpr_model in self.models:
                # Find the RBF component within the potentially complex kernel
                rbf_kernel = self._find_rbf_kernel(gpr_model.kernel_)
                if rbf_kernel is None:
                    raise TypeError("Не вдалося знайти компонент RBF у ядрі моделі GPR.")
                
                # Extract gamma from the RBF kernel's length_scale
                gamma = 1.0 / (2 * rbf_kernel.length_scale ** 2)

                X_train_ = gpr_model.X_train_
                alpha_ = gpr_model.alpha_ # Corresponds to dual_coef_ in KRR context

                # Gradient calculation logic is identical to KRR's RBF
                diffs = X0[:, None, :] - X_train_[None, :, :]
                sq_diffs = np.sum(diffs**2, axis=-1)
                K_row = np.exp(-gamma * sq_diffs)
                
                dK_dX = -2 * gamma * diffs * K_row[..., None]
                
                # W_col for each output target
                W_col = np.einsum('ji,j->i', dK_dX.squeeze(axis=0), alpha_.flatten()).reshape(-1, 1)

                # Calculate local intercept for each output target
                y0_col = gpr_model.predict(X0)
                b_col = y0_col - X0 @ W_col
                
                W_columns.append(W_col)
                b_elements.append(b_col)

            W_local = np.hstack(W_columns) # Stack columns to form the full W matrix
            b_local = np.array(b_elements).flatten() # Flatten elements to form the full b vector
            
            return W_local, b_local

        raise NotImplementedError(f"Лінеаризація для типу моделі '{self.model_type}' не реалізована.")

    def _find_rbf_kernel(self, kernel):
        """Recursively searches for an RBF component within a composite kernel."""
        if isinstance(kernel, RBF):
            return kernel
        elif isinstance(kernel, (Product, Sum)):
            # Recursive search in both components (k1 and k2)
            rbf_in_k1 = self._find_rbf_kernel(kernel.k1)
            if rbf_in_k1:
                return rbf_in_k1
            rbf_in_k2 = self._find_rbf_kernel(kernel.k2)
            if rbf_in_k2:
                return rbf_in_k2
        return None

    def _calculate_median_heuristic_gamma(self, X: np.ndarray) -> float:
        """
        Calculates gamma using the median heuristic: 1 / median(||x - x'||^2).
        Used for the RBF kernel when gamma is not specified or found via search.
        X is expected to be scaled.
        """
        if X.shape[0] > 1000: # Limit for large datasets to avoid excessive computations
            # Select a random subset for median calculation if data is too large
            rng = np.random.default_rng(42) # For reproducibility
            sample_indices = rng.choice(X.shape[0], size=1000, replace=False)
            X_sample = X[sample_indices]
        else:
            X_sample = X

        # Compute pairwise Euclidean distances squared
        # This is more efficient than iterating
        # (N, 1, D) - (1, N, D) -> (N, N, D) then sum along axis D -> (N, N)
        distances_sq = np.sum((X_sample[:, None, :] - X_sample[None, :, :])**2, axis=2)
        
        # Select only the upper triangle (without diagonal) to avoid duplicates and zeros
        upper_tri_indices = np.triu_indices(distances_sq.shape[0], k=1)
        upper_tri_distances = distances_sq[upper_tri_indices]
        
        if len(upper_tri_distances) == 0:
            # Case where X_sample contains only one element or is empty
            return 1.0 # Return a default value

        median_sq_dist = np.median(upper_tri_distances)
        
        if median_sq_dist < 1e-9: # Prevent division by zero or very small numbers
            return 1.0 
            
        return 1.0 / median_sq_dist