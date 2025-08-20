# sim_enh.py - Refactored simulation module

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from collections import deque
from datetime import datetime
import time
import json
from pathlib import Path

# Core imports
from data_gen import StatefulDataGenerator
from model import KernelModel
from objectives import MaxIronMassTrackingObjective
from mpc import MPCController
from ekf import ExtendedKalmanFilter
from anomaly_detector import SignalAnomalyDetector
from maf import MovingAverageFilter
# from retraining_fixes import update_params_for_robust_retraining

# Analysis and utilities
from utils import (
    run_post_simulation_analysis_enhanced, 
    diagnose_mpc_behavior, 
    diagnose_ekf_detailed
)
from evaluation_simple import (
    evaluate_simulation, 
    print_evaluation_report,
    diagnose_analysis_data
)
from evaluation_storage import quick_save, quick_load  
from evaluation_database import quick_add_to_database
from config_manager import (
    simulate_mpc_with_config, list_configs, create_default_configs,
    prompt_manual_adjustments, load_config, list_saved_results, get_config_info
)


class DataProcessor:
    """Handles data preparation and preprocessing"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        
    def prepare_simulation_data(
        self, 
        reference_df: pd.DataFrame
    ) -> Tuple[StatefulDataGenerator, pd.DataFrame, np.ndarray, np.ndarray]:
        """Generate and preprocess simulation data"""
        print("Step 1: Generating simulation data...")

        # Initialize data generator
        true_gen = self._create_data_generator(reference_df)
        
        # Generate anomaly configuration
        anomaly_cfg = self._generate_anomaly_config()
        
        # Generate base time series
        df_true_orig = true_gen.generate(
            T=self.params['N_data'],
            control_pts=self.params['control_pts'],
            n_neighbors=self.params['n_neighbors'],
            noise_level=self.params['noise_level'],
            anomaly_config=anomaly_cfg
        )
        
        # Apply nonlinear transformations if enabled
        df_true = self._apply_nonlinear_transformations(true_gen, df_true_orig)
        
        # Apply anomaly detection filtering
        df_true = self._apply_anomaly_filtering(df_true)
        
        # Create lagged dataset
        X, Y = self._create_lagged_features(df_true)
        
        return true_gen, df_true, X, Y
    
    def _create_data_generator(self, reference_df: pd.DataFrame) -> StatefulDataGenerator:
        """Create and configure data generator"""
        return StatefulDataGenerator(
            reference_df,
            ore_flow_var_pct=3.0,
            time_step_s=self.params['time_step_s'],
            time_constants_s=self.params['time_constants_s'],
            dead_times_s=self.params['dead_times_s'],
            true_model_type=self.params['plant_model_type'],
            seed=self.params['seed']
        )
    
    def _generate_anomaly_config(self) -> Dict:
        """Generate anomaly configuration"""
        return StatefulDataGenerator.generate_anomaly_config(
            N_data=self.params['N_data'],
            train_frac=self.params['train_size'],
            val_frac=self.params['val_size'],
            test_frac=self.params['test_size'],
            seed=self.params['seed']
        )
    
    def _apply_nonlinear_transformations(
        self, 
        true_gen: StatefulDataGenerator, 
        df_true_orig: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply nonlinear transformations if enabled"""
        if self.params['enable_nonlinear']:
            return true_gen.generate_nonlinear_variant(
                base_df=df_true_orig,
                non_linear_factors=self.params['nonlinear_config'],
                noise_level='none',
                anomaly_config=None
            )
        return df_true_orig
    
    def _apply_anomaly_filtering(self, df_true: pd.DataFrame) -> pd.DataFrame:
        """Apply offline anomaly filtering"""
        ad_config = self.params.get('anomaly_params', {})
        ad_feed_fe = SignalAnomalyDetector(**ad_config)
        ad_ore_flow = SignalAnomalyDetector(**ad_config)

        filtered_feed = [ad_feed_fe.update(val) for val in df_true['feed_fe_percent']]
        filtered_ore = [ad_ore_flow.update(val) for val in df_true['ore_mass_flow']]

        df_filtered = df_true.copy()
        df_filtered['feed_fe_percent'] = filtered_feed
        df_filtered['ore_mass_flow'] = filtered_ore
        
        return df_filtered
    
    def _create_lagged_features(self, df_true: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged feature matrices"""
        X, Y_full_np = StatefulDataGenerator.create_lagged_dataset(
            df_true, lags=self.params['lag']
        )
        # Select only concentrate_fe and concentrate_mass columns
        Y = Y_full_np[:, [0, 2]]
        return X, Y
    
    def split_and_scale_data(
        self, 
        X: np.ndarray, 
        Y: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], StandardScaler, StandardScaler]:
        """Split data into train/val/test sets and apply scaling"""
        n = X.shape[0]
        n_train = int(self.params['train_size'] * n)
        n_val = int(self.params['val_size'] * n)

        data_splits = {
            'X_train': X[:n_train], 
            'Y_train': Y[:n_train],
            'X_val': X[n_train:n_train + n_val], 
            'Y_val': Y[n_train:n_train + n_val],
            'X_test': X[n_train + n_val:], 
            'Y_test': Y[n_train + n_val:]
        }

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        data_splits['X_train_scaled'] = x_scaler.fit_transform(data_splits['X_train'])
        data_splits['Y_train_scaled'] = y_scaler.fit_transform(data_splits['Y_train'])
        data_splits['X_val_scaled'] = x_scaler.transform(data_splits['X_val'])
        data_splits['Y_val_scaled'] = y_scaler.transform(data_splits['Y_val'])
        data_splits['X_test_scaled'] = x_scaler.transform(data_splits['X_test'])
        data_splits['Y_test_scaled'] = y_scaler.transform(data_splits['Y_test'])
        
        return data_splits, x_scaler, y_scaler


class MPCControllerFactory:
    """Factory for creating and configuring MPC controllers"""
    
    @staticmethod
    def create_controller(
        params: Dict[str, Any],
        x_scaler: StandardScaler,
        y_scaler: StandardScaler
    ) -> MPCController:
        """Create and configure MPC controller"""
        print("Step 2: Initializing enhanced MPC controller...")
        
        # Create model based on type
        model = MPCControllerFactory._create_model(params)
        
        # Create objective function
        objective = MPCControllerFactory._create_objective(params, y_scaler)
        
        # Configure trust region parameters
        trust_params = MPCControllerFactory._configure_trust_region(params)
        
        # Calculate constraint weights
        constraint_weights = MPCControllerFactory._calculate_constraint_weights(params)
        
        # Create MPC controller
        return MPCController(
            model=model,
            objective=objective,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            n_targets=2,
            horizon=params['Np'],
            control_horizon=params['Nc'],
            lag=params['lag'],
            u_min=params['u_min'],
            u_max=params['u_max'],
            delta_u_max=params['delta_u_max'],
            use_disturbance_estimator=params['use_disturbance_estimator'],
            y_max=MPCControllerFactory._get_output_constraints(params, y_scaler),
            rho_y=constraint_weights['rho_y'],
            rho_delta_u=constraint_weights['rho_delta_u'],
            rho_trust=params['rho_trust'],
            **trust_params
        )
    
    @staticmethod
    def _create_model(params: Dict[str, Any]) -> KernelModel:
        """Create model based on configuration"""
        model_type = params['model_type'].lower()
        
        if model_type == 'linear':
            print(f"   🔧 L-MPC: {params.get('linear_type', 'ols')}")
            return KernelModel(
                model_type=model_type,
                linear_type=params.get('linear_type', 'ols'),
                poly_degree=params.get('poly_degree', 1),
                include_bias=params.get('include_bias', True),
                alpha=params.get('alpha', 1.0),
                find_optimal_params=params.get('find_optimal_params', False)
            )
        else:
            print(f"   🧠 K-MPC: {model_type}")
            return KernelModel(
                model_type=model_type,
                kernel=params.get('kernel', 'rbf'),
                find_optimal_params=params.get('find_optimal_params', True)
            )
    
    @staticmethod
    def _create_objective(params: Dict[str, Any], y_scaler: StandardScaler) -> MaxIronMassTrackingObjective:
        """Create objective function"""
        ref_point_scaled = y_scaler.transform(
            np.array([[params['ref_fe'], params['ref_mass']]])
        )[0]
        
        return MaxIronMassTrackingObjective(
            λ=params['λ_obj'],
            w_fe=params['w_fe'],
            w_mass=params['w_mass'],
            ref_fe=ref_point_scaled[0],
            ref_mass=ref_point_scaled[1],
            K_I=params['K_I']
        )
    
    @staticmethod
    def _configure_trust_region(params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure trust region parameters"""
        model_type = params['model_type'].lower()
        
        if model_type == 'linear':
            # Conservative settings for linear models
            defaults = {
                'adaptive_trust_region': False,
                'initial_trust_radius': 1.2,
                'min_trust_radius': 0.8,
                'max_trust_radius': 3.0,
                'trust_decay_factor': 0.9
            }
        else:
            # Standard settings for kernel models
            defaults = {
                'adaptive_trust_region': True,
                'initial_trust_radius': 1.0,
                'min_trust_radius': 0.1,
                'max_trust_radius': 5.0,
                'trust_decay_factor': 0.8
            }
        
        # Override with user parameters
        trust_params = {key: params.get(key, default) for key, default in defaults.items()}
        trust_params.update({
            'linearization_check_enabled': params.get('linearization_check_enabled', True),
            'max_linearization_distance': params.get('max_linearization_distance', 2.0)
        })
        
        return trust_params
    
    @staticmethod
    def _calculate_constraint_weights(params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate constraint weights"""
        avg_tracking_weight = (params['w_fe'] + params['w_mass']) / 2.0
        return {
            'rho_y': avg_tracking_weight * 1000,
            'rho_delta_u': params['λ_obj'] * 100
        }
    
    @staticmethod
    def _get_output_constraints(params: Dict[str, Any], y_scaler: StandardScaler) -> Optional[list]:
        """Get output constraints if soft constraints are enabled"""
        if params['use_soft_constraints']:
            y_max_scaled = y_scaler.transform(
                np.array([[params['y_max_fe'], params['y_max_mass']]])
            )[0]
            return list(y_max_scaled)
        return None


class EKFFactory:
    """Factory for creating and configuring Extended Kalman Filter"""
    
    @staticmethod
    def create_ekf(
        mpc: MPCController,
        scalers: Tuple[StandardScaler, StandardScaler],
        hist0_unscaled: np.ndarray,
        Y_train_scaled: np.ndarray,
        lag: int,
        params: Dict[str, Any]
    ) -> ExtendedKalmanFilter:
        """Create and configure EKF with intelligent initialization"""
        print("Step 4: Initializing Kalman Filter (EKF)...")
        
        x_scaler, y_scaler = scalers
        n_phys, n_dist = (lag + 1) * 3, 2
        
        # Intelligent disturbance initialization
        initial_disturbances = EKFFactory._estimate_initial_disturbances(Y_train_scaled)
        
        # Form augmented initial state
        x0_aug = np.hstack([hist0_unscaled.flatten(), initial_disturbances])
        
        # Configure covariance matrices
        P0, Q, R = EKFFactory._configure_covariance_matrices(
            n_phys, n_dist, initial_disturbances, Y_train_scaled, params
        )
        
        return ExtendedKalmanFilter(
            mpc.model, x_scaler, y_scaler, x0_aug, P0, Q, R, lag,
            beta_R=params.get('beta_R', 0.1),
            q_adaptive_enabled=params.get('q_adaptive_enabled', True),
            q_alpha=params.get('q_alpha', 0.995),
            q_nis_threshold=params.get('q_nis_threshold', 1.8)
        )
    
    @staticmethod
    def _estimate_initial_disturbances(Y_train_scaled: np.ndarray) -> np.ndarray:
        """Estimate initial disturbances from training data"""
        if len(Y_train_scaled) > 100:
            early_period = Y_train_scaled[:50]
            late_period = Y_train_scaled[-50:]
            
            fe_drift = np.mean(late_period[:, 0]) - np.mean(early_period[:, 0])
            mass_drift = np.mean(late_period[:, 1]) - np.mean(early_period[:, 1])
            
            # Limit disturbances to reasonable bounds
            fe_std = np.std(Y_train_scaled[:, 0])
            mass_std = np.std(Y_train_scaled[:, 1])
            
            max_disturbance_fe = 0.5 * fe_std
            max_disturbance_mass = 0.5 * mass_std
            
            fe_bias = np.clip(fe_drift, -max_disturbance_fe, max_disturbance_fe)
            mass_bias = np.clip(mass_drift, -max_disturbance_mass, max_disturbance_mass)
            
            print(f"   📊 Initial disturbance estimates: Fe={fe_bias:.3f}, Mass={mass_bias:.3f}")
            return np.array([fe_bias, mass_bias])
        else:
            print("   ⚠️ Limited training data, using default disturbances")
            return np.array([0.1, 0.0])
    
    @staticmethod
    def _configure_covariance_matrices(
        n_phys: int,
        n_dist: int,
        initial_disturbances: np.ndarray,
        Y_train_scaled: np.ndarray,
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Configure covariance matrices P0, Q, R"""
        # Initial covariance P0
        P0 = np.eye(n_phys + n_dist) * params['P0']
        
        # Adaptive disturbance uncertainty
        for i, dist in enumerate(initial_disturbances):
            uncertainty = max(0.1, abs(dist) * 2)
            P0[n_phys + i, n_phys + i] = uncertainty
        
        # Process noise Q
        Q_phys = np.eye(n_phys) * params['Q_phys']
        
        if len(Y_train_scaled) > 50:
            # Adaptive disturbance noise
            fe_variability = np.std(np.diff(Y_train_scaled[:, 0]))
            mass_variability = np.std(np.diff(Y_train_scaled[:, 1]))
            
            Q_dist_fe = max(params['Q_dist'], fe_variability * 0.1)
            Q_dist_mass = max(params['Q_dist'], mass_variability * 0.1)
            Q_dist = np.diag([Q_dist_fe, Q_dist_mass])
        else:
            Q_dist = np.eye(n_dist) * params['Q_dist']
        
        Q = np.block([[Q_phys, np.zeros((n_phys, n_dist))], 
                      [np.zeros((n_dist, n_phys)), Q_dist]])
        
        # Measurement noise R
        measurement_variances = np.var(Y_train_scaled, axis=0)
        min_R_values = np.array([1e-4, 1e-4])
        R_values = np.maximum(measurement_variances * params['R'], min_R_values)
        R = np.diag(R_values)
        
        return P0, Q, R


class ModelTrainer:
    """Handles model training and evaluation"""
    
    @staticmethod
    def train_and_evaluate(
        mpc: MPCController,
        data: Dict[str, np.ndarray],
        y_scaler: StandardScaler
    ) -> Tuple[Dict[str, float], Dict[str, list]]:
        """Train model and evaluate performance with timing metrics"""
        print("Step 3: Training and evaluating process model...")
        
        # Train model with timing
        start_time = time.time()
        mpc.fit(data['X_train_scaled'], data['Y_train_scaled'])
        training_time = time.time() - start_time
        
        print(f"   Training time: {training_time:.2f} sec")
        
        # Evaluate on test set
        y_pred_scaled = mpc.model.predict(data['X_test_scaled'])
        y_pred_orig = y_scaler.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        metrics = ModelTrainer._calculate_metrics(data['Y_test'], y_pred_orig)
        
        # Initialize timing metrics
        timing_metrics = {
            'initial_training_time': training_time,
            'retraining_times': [],
            'prediction_times': []
        }
        
        return metrics, timing_metrics
    
    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        test_mse = mean_squared_error(y_true, y_pred)
        
        metrics = {'test_mse_total': test_mse}
        output_columns = ['conc_fe', 'conc_mass']
        
        for i, col in enumerate(output_columns):
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            metrics[f'test_rmse_{col}'] = rmse
            print(f"   RMSE for {col}: {rmse:.3f}")
        
        return metrics


class SimulationLoop:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.setup_filters()
        self.setup_robust_retraining_strategy()  # ← ЗМІНЕНО: використовує robust
        
        # Enhanced tracking
        self.prediction_errors = []
        self.innovation_norms = []
        self.retraining_events = []
    
    def setup_robust_retraining_strategy(self):
        """
        НОВА МЕТОД: Налаштування robust retraining strategy
        
        ЗАМІНЮЄ: setup_retraining_strategy()
        """
        if not self.params.get('enable_retraining', False):
            self.retraining_manager = None
            self.robust_strategy = None
            print("❌ Retraining disabled")
            return
        
        # Спроба імпорту RobustRetrainingStrategy
        try:
            from retraining_fixes import RobustRetrainingStrategy
            self.use_robust_strategy = True
            print("🛡️ Loading RobustRetrainingStrategy...")
        except ImportError:
            print("⚠️ RobustRetrainingStrategy not found, using enhanced legacy")
            self.use_robust_strategy = False
            self.setup_enhanced_legacy_retraining()
            return
        
        # РЕАЛЬНЕ СТВОРЕННЯ RobustRetrainingStrategy
        self.robust_strategy = RobustRetrainingStrategy(
            window_size=self.params.get('retrain_window_size', 1000),
            retrain_period=self.params.get('retrain_period', 50),
            innovation_threshold=self.params.get('retrain_innov_threshold', 0.3),
            
            # Robust parameters для запобігання catastrophic forgetting
            stability_buffer_size=self.params.get('stability_buffer_size', 250),
            regime_change_detection=self.params.get('regime_change_detection', True),
            conservative_retrain_factor=self.params.get('conservative_retrain_factor', 0.75),
            
            # Multi-threshold system
            soft_innovation_threshold=self.params.get('soft_innovation_threshold', 0.2),
            hard_innovation_threshold=self.params.get('hard_innovation_threshold', 0.5),
            regime_change_threshold=self.params.get('regime_change_threshold', 2.0),
            
            # Adaptive window management
            min_stable_ratio=self.params.get('min_stable_ratio', 0.35),
            max_recent_ratio=self.params.get('max_recent_ratio', 0.65),
        )
        
        print("✅ RobustRetrainingStrategy initialized successfully")
        print(f"   • Window policy: ROBUST with catastrophic forgetting protection")
        print(f"   • Stability buffer: {self.params.get('stability_buffer_size', 250)}")
        print(f"   • Multi-threshold: {self.params.get('soft_innovation_threshold', 0.2)} < {self.params.get('retrain_innov_threshold', 0.3)} < {self.params.get('hard_innovation_threshold', 0.5)}")
    
    def setup_enhanced_legacy_retraining(self):
        """Fallback для випадку, коли RobustRetrainingStrategy недоступний"""
        print("🔄 Setting up enhanced legacy retraining...")
        
        self.retraining_buffer = deque(maxlen=self.params['retrain_window_size'])
        self.innovation_monitor = deque(maxlen=self.params['retrain_period'])
        self.retrain_cooldown_timer = 0
        
        # Додаємо базовий захист від catastrophic forgetting
        self.stability_buffer = deque(maxlen=200)
        self.performance_monitor = deque(maxlen=50)
        self.consecutive_failures = 0
        
        print(f"   • Enhanced legacy configured with basic protection")
    

    
    def _setup_robust_strategy_with_mpc(self, mpc, scalers, data):
        """
        НОВА МЕТОД: Налаштування robust strategy з MPC controller
        """
        x_scaler, y_scaler = scalers
        
        # Додаємо початкові тренувальні дані до robust strategy
        initial_data_count = min(100, len(data['X_train_scaled']))
        
        for i in range(initial_data_count):
            x_scaled = data['X_train_scaled'][i]
            y_scaled = data['Y_train_scaled'][i]
            
            # Конвертуємо назад в оригінальний масштаб для robust strategy
            x_unscaled = x_scaler.inverse_transform(x_scaled.reshape(1, -1))[0]
            y_unscaled = y_scaler.inverse_transform(y_scaled.reshape(1, -1))[0]
            
            # Створюємо штучні метрики для початкових даних
            metrics = {
                'innovation_norm': 0.1,
                'prediction_error': 0.05,
                'ekf_uncertainty': 0.1,
                'measurement_variability': 0.05
            }
            
            # РЕАЛЬНО ДОДАЄМО дані до RobustRetrainingStrategy
            self.robust_strategy.add_data_point(x_unscaled, y_unscaled, metrics)
        
        print(f"   • Added {initial_data_count} initial samples to RobustRetrainingStrategy")
        print(f"   • Stability buffer initialized: {len(self.robust_strategy.stability_buffer)} samples")
    
    def _handle_actual_robust_retraining(
        self, 
        ekf, 
        y_meas: np.ndarray, 
        y_pred_unscaled: np.ndarray, 
        measurements: np.ndarray, 
        t: int,
        x_unscaled: np.ndarray,
        step_metrics: list,
        retraining_log: list,
        mpc,
        timing_metrics: Dict[str, list]
    ) -> bool:
        """
        НОВА МЕТОД: РЕАЛЬНЕ використання RobustRetrainingStrategy
        
        ЦЯ МЕТОД ДІЙСНО ВИКЛИКАЄ RobustRetrainingStrategy методи!
        """
        
        # Обчислюємо комплексні метрики
        metrics = self._calculate_comprehensive_metrics(ekf, y_meas, y_pred_unscaled, measurements, t)
        step_metrics.append(metrics)
        
        # ↓ РЕАЛЬНИЙ ВИКЛИК RobustRetrainingStrategy.add_data_point() ↓
        self.robust_strategy.add_data_point(x_unscaled, y_meas, metrics)
        
        # ↓ РЕАЛЬНИЙ ВИКЛИК RobustRetrainingStrategy.should_retrain() ↓
        retrain_type = self.robust_strategy.should_retrain(t, metrics)
        
        if retrain_type != 'none':
            print(f"\n🛡️ ROBUST RETRAINING ACTIVATED: {retrain_type.upper()} at step {t}")
            
            # ↓ РЕАЛЬНИЙ ВИКЛИК RobustRetrainingStrategy.get_training_data() ↓
            X_train, Y_train = self.robust_strategy.get_training_data(retrain_type)
            
            if len(X_train) == 0:
                print("⚠️ RobustRetrainingStrategy returned no training data")
                return False
            
            # Конвертуємо дані в потрібний масштаб для MPC
            x_scaler, y_scaler = self._get_scalers_from_mpc(mpc)
            
            if x_scaler is not None and y_scaler is not None:
                X_train_scaled = x_scaler.transform(X_train)
                Y_train_scaled = y_scaler.transform(Y_train)
            else:
                # Припускаємо, що дані вже в потрібному масштабі
                X_train_scaled = X_train
                Y_train_scaled = Y_train
            
            # Виконуємо перенавчання з timing
            start_time = time.time()
            
            try:
                if retrain_type == 'soft':
                    print("🔄 Performing SOFT robust retraining...")
                    # Для soft retraining можна додати регуляризацію
                    mpc.fit(X_train_scaled, Y_train_scaled)
                else:
                    print(f"🚨 Performing {retrain_type.upper()} robust retraining...")
                    mpc.fit(X_train_scaled, Y_train_scaled)
                
                retrain_time = time.time() - start_time
                
                # Оцінюємо нову продуктивність
                if len(Y_train_scaled) > 10:
                    y_pred_test = mpc.model.predict(X_train_scaled[-10:])
                    new_performance = np.mean(np.linalg.norm(Y_train_scaled[-10:] - y_pred_test, axis=1))
                else:
                    new_performance = 0.0
                
                # ↓ РЕАЛЬНИЙ ВИКЛИК RobustRetrainingStrategy.update_after_retraining() ↓
                self.robust_strategy.update_after_retraining(t, retrain_time, new_performance, retrain_type)
                
                # Логуємо подію перенавчання
                retraining_event = {
                    'step': t,
                    'type': retrain_type,
                    'metrics': metrics.copy(),
                    'training_samples': len(X_train),
                    'performance': new_performance,
                    'robust_strategy': True  # Відмічаємо, що використовувався robust strategy
                }
                retraining_log.append(retraining_event)
                
                # Зберігаємо timing
                timing_metrics['retraining_times'].append(retrain_time)
                
                # Скидаємо trust region
                if hasattr(mpc, 'reset_trust_region'):
                    mpc.reset_trust_region()
                
                print(f"   ✅ Robust retraining completed in {retrain_time:.3f}s with performance {new_performance:.4f}")
                return True
                
            except Exception as e:
                print(f"   ❌ Robust retraining failed: {e}")
                return False
        
        return False
    
    def _get_scalers_from_mpc(self, mpc):
        """Отримати scalers з MPC controller"""
        try:
            return mpc.x_scaler, mpc.y_scaler
        except AttributeError:
            print("⚠️ MPC controller doesn't have scalers")
            return None, None
    
    def _compile_robust_retraining_statistics(
        self, 
        step_metrics: list, 
        retraining_log: list, 
        timing_metrics: Dict[str, list]
    ) -> Dict[str, Any]:
        """
        НОВА МЕТОД: Компіляція статистики з RobustRetrainingStrategy
        """
        
        # Базова статистика
        stats = {
            'total_steps': len(step_metrics),
            'retraining_events': len(retraining_log),
            'retraining_frequency': len(retraining_log) / len(step_metrics) if step_metrics else 0.0,
        }
        
        # Timing статистика
        if timing_metrics.get('retraining_times'):
            stats['avg_retrain_time'] = np.mean(timing_metrics['retraining_times'])
            stats['max_retrain_time'] = np.max(timing_metrics['retraining_times'])
            stats['total_retrain_time'] = np.sum(timing_metrics['retraining_times'])
        
        # Метрики продуктивності
        if step_metrics:
            innovation_norms = [m.get('innovation_norm', 0.0) for m in step_metrics]
            prediction_errors = [m.get('prediction_error', 0.0) for m in step_metrics]
            
            stats['avg_innovation_norm'] = np.mean(innovation_norms)
            stats['max_innovation_norm'] = np.max(innovation_norms)
            stats['avg_prediction_error'] = np.mean(prediction_errors)
            stats['max_prediction_error'] = np.max(prediction_errors)
        
        # ↓ РЕАЛЬНИЙ ВИКЛИК RobustRetrainingStrategy.get_enhanced_statistics() ↓
        if self.use_robust_strategy and hasattr(self, 'robust_strategy'):
            try:
                robust_stats = self.robust_strategy.get_enhanced_statistics()
                # Додаємо robust статистику з префіксом
                for key, value in robust_stats.items():
                    stats[f'robust_{key}'] = value
                
                print(f"   📊 Added robust strategy statistics: {len(robust_stats)} metrics")
            except Exception as e:
                print(f"   ⚠️ Failed to get robust statistics: {e}")
        
        # Аналіз типів перенавчання
        if retraining_log:
            retraining_types = [event.get('type', 'unknown') for event in retraining_log]
            for retrain_type in ['soft', 'hard', 'emergency']:
                count = retraining_types.count(retrain_type)
                stats[f'{retrain_type}_retrainings'] = count
        
        return stats
    
    def _print_robust_diagnostic_info(self, step: int, step_metrics: list, retraining_log: list):
        """НОВА МЕТОД: Діагностична інформація з robust strategy"""
        if not step_metrics:
            return
        
        recent_metrics = step_metrics[-min(10, len(step_metrics)):]
        avg_innovation = np.mean([m.get('innovation_norm', 0.0) for m in recent_metrics])
        avg_error = np.mean([m.get('prediction_error', 0.0) for m in recent_metrics])
        
        recent_retrains = len([r for r in retraining_log if r['step'] > step - 50])
        robust_retrains = len([r for r in retraining_log if r.get('robust_strategy', False) and r['step'] > step - 50])
        
        print(f"📊 Step {step}: Innovation={avg_innovation:.3f}, Error={avg_error:.3f}")
        print(f"   Retraining: {recent_retrains} total ({robust_retrains} robust) in last 50 steps")
        
        # ↓ РЕАЛЬНИЙ ВИКЛИК RobustRetrainingStrategy статистики ↓
        if self.use_robust_strategy and hasattr(self, 'robust_strategy'):
            try:
                stability_score = self.robust_strategy._calculate_stability_score()
                buffer_sizes = f"stable={len(self.robust_strategy.stability_buffer)}, recent={len(self.robust_strategy.recent_buffer)}"
                print(f"   Robust: stability={stability_score:.3f}, buffers=({buffer_sizes})")
            except Exception as e:
                print(f"   Robust diagnostics failed: {e}")
                
    def setup_filters(self):
        """Initialize filters for online processing"""
        from maf import MovingAverageFilter
        from anomaly_detector import SignalAnomalyDetector
        
        window_size = 4
        self.filt_feed = MovingAverageFilter(window_size)
        self.filt_ore = MovingAverageFilter(window_size)
        
        # Online anomaly detectors
        ad_config = self.params.get('anomaly_params', {})
        self.ad_feed_fe = SignalAnomalyDetector(**ad_config)
        self.ad_ore_flow = SignalAnomalyDetector(**ad_config)
    
    # def setup_retraining_strategy(self):
    #     """Setup advanced retraining strategy based on parameters"""
    #     if not self.params.get('enable_retraining', False):
    #         self.retraining_manager = None
    #         print("❌ Retraining disabled")
    #         return
        
    #     # Import sliding window components
    #     try:
    #         from sliding_window_retraining import SlidingWindowRetraining, TimeBasedRetraining, RetrainingManager
    #     except ImportError:
    #         print("⚠️ Sliding window retraining not available, using legacy approach")
    #         self.setup_legacy_retraining()
    #         return
        
    #     # Get strategy configuration
    #     strategy_type = self.params.get('retraining_strategy', 'sliding_window')
        
    #     print(f"🔄 Setting up {strategy_type} retraining strategy...")
        
    #     if strategy_type == 'sliding_window':
    #         # Create sliding window strategy with enhanced parameters
    #         strategy = SlidingWindowRetraining(
    #             window_size=self.params.get('retrain_window_size', 1000),
    #             retrain_period=self.params.get('retrain_period', 50),
    #             innovation_threshold=self.params.get('retrain_innov_threshold', 0.3),
    #             quality_threshold=self.params.get('retrain_quality_threshold', 1.5),
    #             window_policy=self.params.get('window_policy', 'adaptive'),
    #             min_retrain_samples=self.params.get('min_retrain_samples', 100),
    #             max_retrain_samples=self.params.get('max_retrain_samples', 2000),
    #             performance_lookback=self.params.get('performance_lookback', 10)
    #         )
            
    #         print(f"   ✅ Sliding window strategy configured:")
    #         print(f"      • Policy: {self.params.get('window_policy', 'adaptive')}")
    #         print(f"      • Window size: {self.params.get('retrain_window_size', 1000)}")
    #         print(f"      • Innovation threshold: {self.params.get('retrain_innov_threshold', 0.3)}")
            
    #     elif strategy_type == 'time_based':
    #         strategy = TimeBasedRetraining(
    #             retrain_interval=self.params.get('retrain_period', 100),
    #             window_size=self.params.get('retrain_window_size', 500)
    #         )
    #         print(f"   ✅ Time-based strategy configured with interval {self.params.get('retrain_period', 100)}")
            
    #     else:
    #         print(f"⚠️ Unknown strategy '{strategy_type}', falling back to legacy")
    #         self.setup_legacy_retraining()
    #         return
        
    #     # Store strategy for later manager creation
    #     self.retraining_strategy = strategy
        
    def setup_legacy_retraining(self):
        """Setup legacy retraining for backward compatibility"""
        if self.params['enable_retraining']:
            print(f"🔄 Setting up legacy retraining...")
            print(f"   • Window: {self.params['retrain_window_size']}")
            print(f"   • Period: {self.params['retrain_period']}")
            
            self.retraining_buffer = deque(maxlen=self.params['retrain_window_size'])
            self.innovation_monitor = deque(maxlen=self.params['retrain_period'])
            self.retrain_cooldown_timer = 0
            
            # Add initial training data to buffer
            self.legacy_retraining = True
        else:
            self.legacy_retraining = False
    
    def run(
        self,
        true_gen,
        mpc,
        ekf,
        df_true,
        data: Dict[str, np.ndarray],
        scalers: Tuple,
        timing_metrics: Dict[str, list],
        progress_callback: Optional[Callable] = None
    ) -> Tuple:
        """
        Основний цикл симуляції з РЕАЛЬНОЮ інтеграцією RobustRetrainingStrategy
        """
        print("🚀 Starting simulation with ROBUST retraining integration...")
        
        # Ініціалізація (без змін)
        x_scaler, y_scaler = scalers
        simulation_state = self._initialize_simulation_state(df_true, data, true_gen, mpc)
        
        # ↓ КРИТИЧНА ЗМІНА: Ініціалізація robust strategy з MPC ↓
        if self.use_robust_strategy and hasattr(self, 'robust_strategy'):
            # ЗАРАЗ robust_strategy готовий до використання
            print("✅ Using RobustRetrainingStrategy for simulation")
            self._setup_robust_strategy_with_mpc(mpc, scalers, data)
        elif hasattr(self, 'retraining_buffer'):
            # Fallback на legacy
            print("✅ Using enhanced legacy retraining")
            self._setup_legacy_training_data(data, scalers)
        else:
            print("❌ No retraining strategy available")
        
        # Storage для результатів
        records = []
        analysis_data = self._initialize_analysis_storage()
        step_metrics = []
        retraining_log = []
        
        print(f"📊 Starting {simulation_state['T_sim']} simulation steps...")
        
        # ↓ ГОЛОВНИЙ ЦИКЛ З ROBUST RETRAINING ↓
        for t in range(simulation_state['T_sim']):
            if progress_callback:
                progress_callback(t, simulation_state['T_sim'], f"Robust step {t + 1}/{simulation_state['T_sim']}")
            
            # Кроки 1-7: Стандартна обробка (без змін)
            measurements = self._process_measurements(simulation_state['d_all'][t, :])
            ekf.predict(simulation_state['u_prev'], measurements)
            self._update_mpc_with_ekf(mpc, ekf)
            
            current_state = ekf.x_hat[:ekf.n_phys].reshape(self.params['lag'] + 1, 3)
            current_state_flat = current_state.flatten().reshape(1, -1)
            current_state_scaled = x_scaler.transform(current_state_flat)
            y_pred_scaled = mpc.model.predict(current_state_scaled)[0]
            y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled.reshape(1, -1))[0]
            
            u_cur, prediction_time = self._optimize_control(mpc, measurements, simulation_state['u_prev'])
            timing_metrics['prediction_times'].append(prediction_time)
            
            y_full = true_gen.step(simulation_state['d_all'][t, 0], simulation_state['d_all'][t, 1], u_cur)
            y_meas = y_full[['concentrate_fe_percent', 'concentrate_mass_flow']].values.flatten()
            
            ekf.update(y_meas)
            
            # ↓ КРОК 8: РЕАЛЬНЕ ВИКОРИСТАННЯ RobustRetrainingStrategy ↓
            if self.use_robust_strategy and hasattr(self, 'robust_strategy'):
                retrained = self._handle_actual_robust_retraining(
                    ekf, y_meas, y_pred_unscaled, measurements, t, 
                    current_state_flat[0], step_metrics, retraining_log, mpc, timing_metrics
                )
            elif hasattr(self, 'retraining_buffer'):
                retrained = self._handle_enhanced_legacy_retraining(
                    mpc, ekf, scalers, current_state_flat[0], y_meas, t, timing_metrics
                )
            else:
                retrained = False
            
            # Кроки 9-13: Стандартне завершення (без змін)
            self._store_enhanced_analysis_data(
                analysis_data, ekf, mpc, y_meas, y_pred_unscaled, t, u_cur, 
                step_metrics[-1] if step_metrics else {}
            )
            
            records.append(self._create_simulation_record(y_full.iloc[0], u_cur))
            simulation_state['u_prev'] = u_cur
            
            if hasattr(self, 'retrain_cooldown_timer') and self.retrain_cooldown_timer > 0:
                self.retrain_cooldown_timer -= 1
            
            # Diagnostic output
            if t % 50 == 0 and t > 0:
                self._print_robust_diagnostic_info(t, step_metrics, retraining_log)
        
        print(f"\n🏁 Robust simulation completed: {simulation_state['T_sim']} steps")
        
        # Фіналізація з robust статистикою
        analysis_data['retraining_statistics'] = self._compile_robust_retraining_statistics(
            step_metrics, retraining_log, timing_metrics
        )
        
        self._finalize_enhanced_analysis_data(analysis_data, timing_metrics, step_metrics)
        
        return pd.DataFrame(records), analysis_data
    
    def _handle_advanced_retraining(
        self, 
        ekf, 
        y_meas: np.ndarray, 
        y_pred_unscaled: np.ndarray, 
        measurements: np.ndarray, 
        t: int,
        x_unscaled: np.ndarray,
        step_metrics: list,
        retraining_log: list
    ) -> bool:
        """Handle advanced sliding window retraining"""
        
        # Calculate comprehensive metrics for retraining decision
        metrics = self._calculate_comprehensive_metrics(
            ekf, y_meas, y_pred_unscaled, measurements, t
        )
        
        # Store metrics for analysis
        step_metrics.append(metrics)
        
        # Add data point to retraining strategy
        y_unscaled = y_meas
        self.retraining_manager.add_data_point(x_unscaled, y_unscaled, metrics)
        
        # Check and perform retraining if needed
        retrained = self.retraining_manager.check_and_retrain(t, metrics)
        
        if retrained:
            retraining_event = {
                'step': t,
                'metrics': metrics.copy(),
                'window_size': getattr(self.retraining_manager.strategy, 'adaptive_window_size', 'N/A'),
                'strategy': self.params.get('window_policy', 'unknown')
            }
            retraining_log.append(retraining_event)
            print(f"📊 Advanced retraining event logged at step {t}")
        
        # Update strategy state
        self.retraining_manager.step()
        
        return retrained
    
    def _handle_legacy_retraining(
        self,
        mpc,
        ekf,
        scalers: Tuple,
        x_unscaled: np.ndarray,
        y_meas: np.ndarray,
        t: int,
        timing_metrics: Dict[str, list]
    ) -> bool:
        """Handle legacy retraining for backward compatibility"""
        
        x_scaler, y_scaler = scalers
        
        # Add new data to buffer
        new_x_unscaled = x_unscaled.reshape(1, -1)
        new_y_unscaled = y_meas.reshape(1, -1)
        
        new_x_scaled = x_scaler.transform(new_x_unscaled)
        new_y_scaled = y_scaler.transform(new_y_unscaled)
        
        self.retraining_buffer.append((new_x_scaled[0], new_y_scaled[0]))
        
        # Monitor innovation
        if ekf.last_innovation is not None:
            innov_norm = np.linalg.norm(ekf.last_innovation)
            self.innovation_monitor.append(innov_norm)
        
        # Check retraining conditions (legacy logic)
        if (t > 0 and
            t % self.params['retrain_period'] == 0 and
            len(self.innovation_monitor) == self.params['retrain_period'] and
            self.retrain_cooldown_timer == 0):

            avg_innov = float(np.mean(self.innovation_monitor))

            if avg_innov > self.params['retrain_innov_threshold']:
                print(f"\n---> LEGACY RETRAINING TRIGGER at step {t}! "
                      f"Avg innovation: {avg_innov:.4f} > {self.params['retrain_innov_threshold']:.4f}")

                retrain_data = list(self.retraining_buffer)
                X_retrain = np.array([p[0] for p in retrain_data])
                Y_retrain = np.array([p[1] for p in retrain_data])

                print(f"--> Legacy mpc.fit() on {len(X_retrain)} samples...")
                
                start_time = time.time()
                mpc.fit(X_retrain, Y_retrain)
                retrain_time = time.time() - start_time
                timing_metrics['retraining_times'].append(retrain_time)
                
                print(f"--> Legacy retraining completed in {retrain_time:.3f} sec.")
                
                # Reset trust region if available
                if hasattr(mpc, 'reset_trust_region'):
                    mpc.reset_trust_region()
                
                self.innovation_monitor.clear()
                self.retrain_cooldown_timer = self.params['retrain_period'] * 2
                
                return True
        
        return False
    
    def _setup_legacy_training_data(self, data: Dict[str, np.ndarray], scalers: Tuple):
        """Setup initial training data for legacy retraining"""
        x_scaler, y_scaler = scalers
        
        # Add initial training data to buffer
        initial_train_data = list(zip(data['X_train_scaled'], data['Y_train_scaled']))
        self.retraining_buffer.extend(initial_train_data)
        
        print(f"   • Added {len(initial_train_data)} initial training samples to legacy buffer")
    
    def _calculate_comprehensive_metrics(
        self, 
        ekf, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        measurements: np.ndarray, 
        step: int
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics for retraining decisions"""
        
        metrics = {}
        
        # Innovation-based metrics
        if ekf.last_innovation is not None:
            metrics['innovation_norm'] = float(np.linalg.norm(ekf.last_innovation))
            metrics['innovation_fe'] = float(abs(ekf.last_innovation[0])) if len(ekf.last_innovation) > 0 else 0.0
            metrics['innovation_mass'] = float(abs(ekf.last_innovation[1])) if len(ekf.last_innovation) > 1 else 0.0
        else:
            metrics['innovation_norm'] = 0.0
            metrics['innovation_fe'] = 0.0
            metrics['innovation_mass'] = 0.0
        
        # Prediction error metrics
        prediction_error = np.linalg.norm(y_true - y_pred)
        metrics['prediction_error'] = float(prediction_error)
        metrics['prediction_error_fe'] = float(abs(y_true[0] - y_pred[0]))
        metrics['prediction_error_mass'] = float(abs(y_true[1] - y_pred[1]))
        
        # Relative errors
        if y_true[0] != 0:
            metrics['relative_error_fe'] = float(abs(y_true[0] - y_pred[0]) / abs(y_true[0]))
        else:
            metrics['relative_error_fe'] = 0.0
            
        if y_true[1] != 0:
            metrics['relative_error_mass'] = float(abs(y_true[1] - y_pred[1]) / abs(y_true[1]))
        else:
            metrics['relative_error_mass'] = 0.0
        
        # Measurement variability
        metrics['measurement_variability'] = float(np.std(measurements))
        
        # EKF confidence (trace of covariance matrix)
        if hasattr(ekf, 'P'):
            metrics['ekf_uncertainty'] = float(np.trace(ekf.P))
        else:
            metrics['ekf_uncertainty'] = 0.0
        
        # Step information
        metrics['step'] = step
        metrics['timestamp'] = time.time()
        
        return metrics
    
    def _process_measurements(self, raw_measurements: np.ndarray) -> np.ndarray:
        """Process raw measurements through filters"""
        feed_fe_raw, ore_flow_raw = raw_measurements
        
        # Online anomaly filtering
        feed_fe_filt = self.ad_feed_fe.update(feed_fe_raw)
        ore_flow_filt = self.ad_ore_flow.update(ore_flow_raw)
        
        # Moving average filtering
        return np.array([
            self.filt_feed.update(feed_fe_filt),
            self.filt_ore.update(ore_flow_filt)
        ])
    
    def _update_mpc_with_ekf(self, mpc, ekf):
        """Update MPC controller with EKF estimates"""
        x_est_phys_unscaled = ekf.x_hat[:ekf.n_phys].reshape(self.params['lag'] + 1, 3)
        mpc.reset_history(x_est_phys_unscaled)
        mpc.d_hat = ekf.x_hat[ekf.n_phys:]
    
    def _optimize_control(
        self,
        mpc,
        measurements: np.ndarray,
        u_prev: float
    ) -> Tuple[float, float]:
        """Optimize control action with timing measurement"""
        start_time = time.time()
        
        d_seq = np.repeat(measurements.reshape(1, -1), self.params['Np'], axis=0)
        u_seq = mpc.optimize(d_seq, u_prev)
        u_cur = u_prev if u_seq is None else float(u_seq[0])
        
        prediction_time = time.time() - start_time
        return u_cur, prediction_time
    
    def _store_enhanced_analysis_data(
        self, 
        analysis_data: Dict, 
        ekf, 
        mpc, 
        y_meas: np.ndarray, 
        y_pred: np.ndarray,
        t: int, 
        u_cur: float,
        step_metrics: Dict[str, float]
    ):
        """Store enhanced analysis data including predictions and metrics"""
        
        # Original data storage
        analysis_data['y_true_hist'].append(y_meas.copy())
        analysis_data['x_hat_hist'].append(ekf.x_hat.copy())
        analysis_data['P_hist'].append(ekf.P.copy())
        analysis_data['R_hist'].append(ekf.R.copy())
        
        # Enhanced data storage
        analysis_data['y_pred_seq'].append(y_pred.copy())
        analysis_data['step_metrics'].append(step_metrics.copy())
        
        # Innovation handling
        innovation = ekf.last_innovation.copy() if ekf.last_innovation is not None else np.zeros(ekf.n_dist)
        analysis_data['innov_hist'].append(innovation)
        
        if ekf.last_innovation is not None and ekf.last_innovation.shape[0] >= 2:
            analysis_data['innovation_seq'].append(ekf.last_innovation[:2].copy())
        else:
            analysis_data['innovation_seq'].append(np.zeros(2))
        
        # Trust region and quality data
        trust_stats = self._get_trust_region_stats(mpc, t)
        trust_stats['step_metrics'] = step_metrics
        analysis_data['trust_region_stats_hist'].append(trust_stats)
        
        # Store MPC control sequences
        u_seq = getattr(mpc, 'last_u_sequence', None)
        if u_seq is not None:
            try:
                plan_data = {
                    'plan': u_seq.copy(),
                    'step': t,
                    'horizon_length': len(u_seq),
                    'first_action': float(u_seq[0]) if len(u_seq) > 0 else None
                }
                analysis_data['u_seq_hist'].append(plan_data)
            except Exception:
                analysis_data['u_seq_hist'].append({
                    'plan': None,
                    'step': t,
                    'optimization_failed': True
                })
        else:
            analysis_data['u_seq_hist'].append({
                'plan': None,
                'step': t,
                'optimization_failed': True
            })
        
        # Store disturbance estimates
        if mpc.d_hat is not None:
            try:
                analysis_data['d_hat_hist'].append(mpc.d_hat.copy())
            except Exception:
                analysis_data['d_hat_hist'].append(np.zeros(2))
        else:
            analysis_data['d_hat_hist'].append(np.zeros(2))
        
        # Linearization quality
        if hasattr(mpc, 'linearization_quality_history') and mpc.linearization_quality_history:
            quality = mpc.linearization_quality_history[-1]
            if isinstance(quality, dict):
                analysis_data['linearization_quality_hist'].append(quality['euclidean_distance'])
            else:
                analysis_data['linearization_quality_hist'].append(quality)
        else:
            analysis_data['linearization_quality_hist'].append(0.0)
    
    def _get_trust_region_stats(self, mpc, t: int) -> Dict:
        """Get trust region statistics with fallback"""
        try:
            if hasattr(mpc, 'get_trust_region_stats'):
                return mpc.get_trust_region_stats()
        except Exception:
            pass
        
        # Fallback implementation
        return {
            'current_radius': getattr(mpc, 'current_trust_radius', 
                                    getattr(mpc, 'trust_radius', 1.0)),
            'radius_increased': False,
            'radius_decreased': False,
            'step': t,
            'optimization_success': True
        }
    
    def _create_simulation_record(self, y_meas, u_cur: float) -> Dict[str, float]:
        """Create simulation record for current timestep"""
        return {
            'feed_fe_percent': y_meas.feed_fe_percent,
            'ore_mass_flow': y_meas.ore_mass_flow,
            'solid_feed_percent': u_cur,
            'conc_fe': y_meas.concentrate_fe_percent,
            'tail_fe': y_meas.tailings_fe_percent,
            'conc_mass': y_meas.concentrate_mass_flow,
            'tail_mass': y_meas.tailings_mass_flow,
            'mass_pull_pct': y_meas.mass_pull_percent,
            'fe_recovery_percent': y_meas.fe_recovery_percent,
        }
    
    def _compile_retraining_statistics(
        self, 
        step_metrics: list, 
        retraining_log: list, 
        timing_metrics: Dict[str, list]
    ) -> Dict[str, Any]:
        """Compile comprehensive retraining statistics"""
        
        stats = {
            'total_steps': len(step_metrics),
            'retraining_events': len(retraining_log),
            'retraining_frequency': len(retraining_log) / len(step_metrics) if step_metrics else 0.0,
        }
        
        # Timing statistics
        if timing_metrics.get('retraining_times'):
            stats['avg_retrain_time'] = np.mean(timing_metrics['retraining_times'])
            stats['max_retrain_time'] = np.max(timing_metrics['retraining_times'])
            stats['total_retrain_time'] = np.sum(timing_metrics['retraining_times'])
            stats['retrain_time_std'] = np.std(timing_metrics['retraining_times'])
        
        # Performance metrics over time
        if step_metrics:
            innovation_norms = [m.get('innovation_norm', 0.0) for m in step_metrics]
            prediction_errors = [m.get('prediction_error', 0.0) for m in step_metrics]
            
            stats['avg_innovation_norm'] = np.mean(innovation_norms)
            stats['max_innovation_norm'] = np.max(innovation_norms)
            stats['avg_prediction_error'] = np.mean(prediction_errors)
            stats['max_prediction_error'] = np.max(prediction_errors)
            
            # Performance trends (last 25% vs first 25%)
            quarter_size = len(step_metrics) // 4
            if quarter_size > 10:
                early_innovation = np.mean(innovation_norms[:quarter_size])
                late_innovation = np.mean(innovation_norms[-quarter_size:])
                stats['innovation_trend'] = (late_innovation - early_innovation) / early_innovation if early_innovation > 0 else 0.0
                
                early_error = np.mean(prediction_errors[:quarter_size])
                late_error = np.mean(prediction_errors[-quarter_size:])
                stats['error_trend'] = (late_error - early_error) / early_error if early_error > 0 else 0.0
        
        # Retraining event analysis
        if retraining_log:
            retrain_steps = [event['step'] for event in retraining_log]
            retrain_intervals = np.diff(retrain_steps) if len(retrain_steps) > 1 else []
            
            if len(retrain_intervals) > 0:
                stats['avg_retrain_interval'] = np.mean(retrain_intervals)
                stats['min_retrain_interval'] = np.min(retrain_intervals)
                stats['max_retrain_interval'] = np.max(retrain_intervals)
        
        # Strategy-specific statistics
        if hasattr(self, 'retraining_manager') and self.retraining_manager:
            strategy_stats = self.retraining_manager.get_statistics()
            stats.update({'strategy_' + k: v for k, v in strategy_stats.items()})
        
        return stats
    
    def _print_diagnostic_info(self, step: int, step_metrics: list, retraining_log: list):
        """Print diagnostic information during simulation"""
        if not step_metrics:
            return
        
        recent_metrics = step_metrics[-min(10, len(step_metrics)):]
        avg_innovation = np.mean([m.get('innovation_norm', 0.0) for m in recent_metrics])
        avg_error = np.mean([m.get('prediction_error', 0.0) for m in recent_metrics])
        
        recent_retrains = len([r for r in retraining_log if r['step'] > step - 50])
        
        print(f"📊 Step {step}: Innovation={avg_innovation:.3f}, Error={avg_error:.3f}, "
              f"Recent retrains={recent_retrains}")
    
    def _finalize_enhanced_analysis_data(
        self, 
        analysis_data: Dict, 
        timing_metrics: Dict[str, list],
        step_metrics: list
    ):
        """Finalize enhanced analysis data structure"""
        
        # Convert lists to arrays
        analysis_data.update({
            "y_true": np.vstack(analysis_data['y_true_hist']),
            "x_hat": np.vstack(analysis_data['x_hat_hist']),
            "P": np.stack(analysis_data['P_hist']),
            "innov": np.vstack(analysis_data['innov_hist']),
            "R": np.stack(analysis_data['R_hist']),
            "u_seq": analysis_data['u_seq_hist'],
            "d_hat": np.vstack(analysis_data['d_hat_hist']) if analysis_data['d_hat_hist'] else np.array([]),
            "trust_region_stats": analysis_data['trust_region_stats_hist'],
            "linearization_quality": analysis_data['linearization_quality_hist'],
            "y_true_seq": analysis_data['y_true_seq'],
            "x_est_seq": analysis_data['x_est_seq'],
            "innovation_seq": analysis_data['innovation_seq'],
            "timing_metrics": timing_metrics
        })
        
        # Add enhanced arrays
        if analysis_data.get('y_pred_seq'):
            analysis_data["y_pred"] = np.vstack(analysis_data['y_pred_seq'])
        
        if step_metrics:
            analysis_data["step_metrics_array"] = step_metrics
        
        # Clean up temporary storage
        temp_keys = ['y_true_hist', 'x_hat_hist', 'P_hist', 'innov_hist', 'R_hist', 
                    'u_seq_hist', 'd_hat_hist', 'trust_region_stats_hist', 
                    'linearization_quality_hist', 'y_pred_seq', 'step_metrics']
        for key in temp_keys:
            if key in analysis_data:
                del analysis_data[key]
    
    def _initialize_simulation_state(self, df_true, data, true_gen, mpc):
        """Initialize simulation state variables"""
        n_total = len(df_true) - self.params['lag'] - 1
        n_train = int(self.params['train_size'] * n_total)
        n_val = int(self.params['val_size'] * n_total)
        test_idx_start = self.params['lag'] + 1 + n_train + n_val
        
        hist0_unscaled = df_true[['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']].iloc[
            test_idx_start - (self.params['lag'] + 1): test_idx_start
        ].values
        
        mpc.reset_history(hist0_unscaled)
        true_gen.reset_state(hist0_unscaled)
        
        df_run = df_true.iloc[test_idx_start:]
        d_all = df_run[['feed_fe_percent', 'ore_mass_flow']].values
        T_sim = len(df_run) - (self.params['lag'] + 1)
        u_prev = float(hist0_unscaled[-1, 2])
        
        return {
            'test_idx_start': test_idx_start,
            'd_all': d_all,
            'T_sim': T_sim,
            'u_prev': u_prev
        }
    
    def _initialize_analysis_storage(self):
        """Initialize storage for analysis data"""
        return {
            'y_true_hist': [],
            'x_hat_hist': [],
            'P_hist': [],
            'innov_hist': [],
            'R_hist': [],
            'u_seq_hist': [],
            'd_hat_hist': [],
            'trust_region_stats_hist': [],
            'linearization_quality_hist': [],
            'y_true_seq': [],
            'y_pred_seq': [],
            'x_est_seq': [],
            'innovation_seq': [],
            'step_metrics': []
        }
    

class SimulationOrchestrator:
    """Main orchestrator for MPC simulation"""
    
    def __init__(self, auto_save: bool = True, database_logging: bool = True):
        self.auto_save = auto_save
        self.database_logging = database_logging
    
    def simulate_mpc(
        self,
        reference_df: pd.DataFrame,
        **kwargs
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
        """
        Enhanced MPC simulation with support for both linear and kernel models.
        
        Args:
            reference_df: Reference data
            **kwargs: Simulation parameters (see default values in method)
            
        Returns:
            Tuple of (results_df, metrics) or (None, None) if error occurs
        """
        # Set default parameters
        params = self._set_default_parameters(kwargs)
        
        # Log simulation type
        self._log_simulation_start(params)
        
        try:
            # Execute simulation pipeline
            results_df, metrics = self._execute_simulation_pipeline(reference_df, params)
            
            # Post-processing
            if params.get('run_evaluation', True):
                self._run_evaluation(results_df, metrics, params)
            
            # Auto-save results
            if self.auto_save:
                self._auto_save_results(results_df, metrics, params)
            
            return results_df, metrics
            
        except Exception as e:
            print(f"❌ Critical error in simulate_mpc: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _set_default_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Set default parameters with user overrides"""
        defaults = {
            # Data generation
            'N_data': 5000,
            'control_pts': 1000,
            'time_step_s': 5,
            'dead_times_s': {
                'concentrate_fe_percent': 20.0,
                'tailings_fe_percent': 25.0,
                'concentrate_mass_flow': 20.0,
                'tailings_mass_flow': 25.0
            },
            'time_constants_s': {
                'concentrate_fe_percent': 8.0,
                'tailings_fe_percent': 10.0,
                'concentrate_mass_flow': 5.0,
                'tailings_mass_flow': 7.0
            },
            
            # Model configuration
            'lag': 2,
            'model_type': 'krr',
            'kernel': 'rbf',
            'find_optimal_params': True,
            
            # Linear model specific
            'linear_type': 'ols',
            'poly_degree': 1,
            'include_bias': True,
            'alpha': 1.0,
            
            # MPC parameters
            'Np': 6,
            'Nc': 4,
            'λ_obj': 0.1,
            'K_I': 0.01,
            'w_fe': 7.0,
            'w_mass': 1.0,
            'ref_fe': 53.5,
            'ref_mass': 57.0,
            
            # Constraints
            'u_min': 20.0,
            'u_max': 40.0,
            'delta_u_max': 1.0,
            'y_max_fe': 54.5,
            'y_max_mass': 58.0,
            'use_soft_constraints': True,
            
            # Trust region
            'adaptive_trust_region': True,
            'initial_trust_radius': 1.0,
            'min_trust_radius': 0.5,
            'max_trust_radius': 5.0,
            'trust_decay_factor': 0.8,
            'rho_trust': 0.1,
            'linearization_check_enabled': True,
            'max_linearization_distance': 2.0,
            
            # Data splits
            'train_size': 0.7,
            'val_size': 0.15,
            'test_size': 0.15,
            
            # Process configuration
            'n_neighbors': 5,
            'seed': 0,
            'noise_level': 'none',
            'plant_model_type': 'rf',
            'use_disturbance_estimator': True,
            
            # Retraining
            'enable_retraining': True,
            'retrain_period': 50,
            'retrain_window_size': 1000,
            'retrain_innov_threshold': 0.3,
            'retrain_linearization_threshold': 1.5,
            
            # Anomaly detection
            'anomaly_params': {
                'window': 25,
                'spike_z': 4.0,
                'drop_rel': 0.30,
                'freeze_len': 5,
                'enabled': True
            },
            
            # Nonlinearity
            'enable_nonlinear': False,
            'nonlinear_config': {
                'concentrate_fe_percent': ('pow', 2),
                'concentrate_mass_flow': ('pow', 1.5)
            },
            
            # EKF parameters
            'P0': 1e-2,
            'Q_phys': 1500,
            'Q_dist': 1,
            'R': 0.01,
            'q_adaptive_enabled': True,
            'q_alpha': 0.99,
            'q_nis_threshold': 1.5,
            'beta_R': 0.1,
            
            # Analysis and evaluation
            'run_analysis': True,
            'run_evaluation': True,
            'show_evaluation_plots': False,
            'tolerance_fe_percent': 2.0,
            'tolerance_mass_percent': 2.0,
            
            # Callbacks
            'progress_callback': None
        }
        
        # Update with user parameters
        defaults.update(kwargs)
        return defaults
    
    def _log_simulation_start(self, params: Dict[str, Any]):
        """Log simulation start information"""
        model_type = params['model_type'].lower()
        
        if model_type == 'linear':
            print(f"🎯 CONFIGURING L-MPC (Linear Model)")
            print(f"   • Type: {params['linear_type']}")
            print(f"   • Polynomial degree: {params['poly_degree']}")
            print(f"   • Bias: {params['include_bias']}")
            if params['linear_type'] in ['ridge', 'lasso']:
                print(f"   • Regularization: {params['alpha']}")
        else:
            print(f"🎯 CONFIGURING K-MPC (Kernel Model: {model_type})")
            print(f"   • Kernel: {params.get('kernel', 'rbf')}")
            print(f"   • Auto-tune parameters: {params.get('find_optimal_params', True)}")
    
    def _execute_simulation_pipeline(
        self,
        reference_df: pd.DataFrame,
        params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Execute the main simulation pipeline"""
        # 1. Data preparation
        data_processor = DataProcessor(params)
        true_gen, df_true, X, Y = data_processor.prepare_simulation_data(reference_df)
        data, x_scaler, y_scaler = data_processor.split_and_scale_data(X, Y)
        
        # 2. MPC controller initialization
        mpc = MPCControllerFactory.create_controller(params, x_scaler, y_scaler)
        
        # 3. Model training
        metrics, timing_metrics = ModelTrainer.train_and_evaluate(mpc, data, y_scaler)
        
        # 4. EKF initialization
        test_idx_start = params['lag'] + 1 + len(data['X_train']) + len(data['X_val'])
        hist0_unscaled = df_true[['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']].iloc[
            test_idx_start - (params['lag'] + 1): test_idx_start
        ].values
        
        ekf = EKFFactory.create_ekf(mpc, (x_scaler, y_scaler), hist0_unscaled, 
                                   data['Y_train_scaled'], params['lag'], params)
        
        # 5. Simulation loop
        simulation_loop = SimulationLoop(params)
        results_df, analysis_data = simulation_loop.run(
            true_gen, mpc, ekf, df_true, data, (x_scaler, y_scaler),
            timing_metrics, params.get('progress_callback')
        )
        
        # 6. Post-simulation analysis
        if params.get('run_analysis', True):
            self._run_post_analysis(results_df, analysis_data, params, df_true, test_idx_start)
        
        return results_df, metrics
    
    def _run_post_analysis(
        self,
        results_df: pd.DataFrame,
        analysis_data: Dict,
        params: Dict[str, Any],
        df_true: pd.DataFrame,
        test_idx_start: int
    ):
        """Run post-simulation analysis"""
        # Add test disturbances to analysis data
        analysis_data['d_all_test'] = df_true.iloc[test_idx_start:][['feed_fe_percent','ore_mass_flow']].values
        
        # Run enhanced analysis
        run_post_simulation_analysis_enhanced(results_df, analysis_data, params)
        
        # Diagnose analysis data
        try:
            diagnose_analysis_data(analysis_data)
        except Exception as e:
            print(f"⚠️ Analysis diagnosis error: {e}")
    
    def _run_evaluation(
        self,
        results_df: pd.DataFrame,
        metrics: Dict[str, float],
        params: Dict[str, Any]
    ):
        """Run performance evaluation"""
        print("\n" + "="*60)
        print("🎯 MPC SYSTEM EFFECTIVENESS EVALUATION")
        print("="*60)
        
        try:
            # Create dummy analysis_data for evaluation
            analysis_data = {'timing_metrics': {'prediction_times': []}}
            
            eval_results = evaluate_simulation(results_df, analysis_data, params)
            simulation_steps = len(results_df)
            print_evaluation_report(eval_results, detailed=True, simulation_steps=simulation_steps)
            
            if params.get('show_evaluation_plots', False):
                print("\n📊 Creating evaluation plots...")
                try:
                    from evaluation_simple import create_evaluation_plots
                    create_evaluation_plots(results_df, eval_results, params)
                except Exception as plot_error:
                    print(f"⚠️ Plot creation error: {plot_error}")
                    
        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
        
        print("="*60)
    
    def _auto_save_results(
        self,
        results_df: pd.DataFrame,
        metrics: Dict[str, float],
        params: Dict[str, Any]
    ):
        """Auto-save simulation results"""
        try:
            # Create dummy analysis_data for saving
            analysis_data = {'timing_metrics': {'prediction_times': []}}
            eval_results = evaluate_simulation(results_df, analysis_data, params)
            
            # Save to file
            file_path = quick_save(
                results_df=results_df,
                eval_results=eval_results,
                analysis_data=analysis_data,
                params=params,
                description=f"Auto-simulation {datetime.now()}"
            )
            
            # Save to database if enabled
            if self.database_logging:
                eval_id = quick_add_to_database(
                    package=quick_load(file_path),
                    series_id="production_runs",
                    tags=["auto", "production"]
                )
                print(f"✅ Simulation saved: file {file_path}, DB ID {eval_id}")
            else:
                print(f"✅ Simulation saved: {file_path}")
                
        except Exception as e:
            print(f"⚠️ Auto-save error: {e}")


# Convenience function for backward compatibility
def simulate_mpc(reference_df: pd.DataFrame, **kwargs) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """
    Main simulation function with enhanced capabilities for both linear and kernel models.
    
    This function provides a simplified interface to the full simulation pipeline,
    supporting both L-MPC (linear models) and K-MPC (kernel models) with intelligent
    parameter adaptation and comprehensive performance evaluation.
    
    Args:
        reference_df: Historical process data for model training
        **kwargs: Simulation parameters (see SimulationOrchestrator._set_default_parameters for full list)
        
    Key Parameters:
        model_type: 'linear' for L-MPC, 'krr'/'svr'/'gpr' for K-MPC
        linear_type: For L-MPC only - 'ols', 'ridge', 'lasso'
        poly_degree: For L-MPC only - polynomial degree (1-3)
        kernel: For K-MPC only - kernel type ('rbf', 'poly', etc.)
        
    Returns:
        Tuple of (results_dataframe, performance_metrics) or (None, None) if error
    """
    orchestrator = SimulationOrchestrator()
    return orchestrator.simulate_mpc(reference_df, **kwargs)


def main():
    """Main execution function for standalone usage"""
    def progress_callback(step, total, msg):
        """Simple progress callback for console output"""
        if step % 20 == 0 or step == total:
            print(f"[{step}/{total}] {msg}")

    # Load data
    try:
        hist_df = pd.read_parquet('processed.parquet')
        print("✅ Data loaded successfully")
    except FileNotFoundError:
        print("❌ Error: 'processed.parquet' file not found.")
        return

    # Setup configurations
    available_configs = list_configs()
    if not available_configs:
        print("🔧 Creating default configurations...")
        create_default_configs()
        available_configs = list_configs()

    # Display available configurations
    print(f"\n📋 AVAILABLE CONFIGURATIONS:")
    print("=" * 50)
    for i, config in enumerate(available_configs, 1):
        try:
            config_info = get_config_info(config)
            if config_info:
                model_type = config_info.get('model_type', 'unknown')
                description = config_info.get('description', 'No description')
                
                type_marker = "🔧 L-MPC" if model_type.lower() == 'linear' else "🧠 K-MPC"
                
                print(f"{i}. {config} {type_marker}")
                print(f"   📄 {description}")
                print(f"   ⚙️ Model: {model_type}, N_data: {config_info.get('N_data', '?')}")
                print()
        except Exception as e:
            print(f"{i}. {config} (error: {e})")

    # Configuration selection
    choice = input(f"Select base configuration (1-{len(available_configs)}, default 1): ").strip()
    
    try:
        config_index = int(choice) - 1 if choice else 0
        selected_config = available_configs[max(0, min(config_index, len(available_configs) - 1))]
    except (ValueError, IndexError):
        selected_config = available_configs[0]

    print(f"🎯 Selected configuration: {selected_config}")

    # Load and display configuration
    base_config = load_config(selected_config)
    model_type = base_config.get('model_type', 'krr').lower()

    # Display configuration details
    if model_type == 'linear':
        print(f"\n🔧 L-MPC CONFIGURATION:")
        print(f"   • Linear model type: {base_config.get('linear_type', 'ols')}")
        print(f"   • Polynomial degree: {base_config.get('poly_degree', 1)}")
        print(f"   • Include bias: {base_config.get('include_bias', True)}")
        if base_config.get('linear_type') in ['ridge', 'lasso']:
            print(f"   • Regularization coefficient: {base_config.get('alpha', 1.0)}")
    else:
        print(f"\n🧠 K-MPC CONFIGURATION:")
        print(f"   • Kernel model type: {model_type}")
        print(f"   • Kernel: {base_config.get('kernel', 'rbf')}")
        print(f"   • Auto-tune parameters: {base_config.get('find_optimal_params', True)}")

    # Manual adjustments
    want_adjustments = input(f"\nMake manual adjustments? (y/N): ").strip().lower()
    manual_overrides = {}
    
    if want_adjustments in ['y', 'yes']:
        manual_overrides = prompt_manual_adjustments(base_config)
        if manual_overrides:
            print(f"\n✅ Planned {len(manual_overrides)} adjustments")
            for key, value in manual_overrides.items():
                old_value = base_config.get(key, "not set")
                print(f"   • {key}: {old_value} → {value}")

    # Evaluation settings
    want_evaluation = input(f"\nEnable effectiveness evaluation? (Y/n): ").strip().lower()
    run_evaluation = want_evaluation not in ['n', 'no']
    
    show_evaluation_plots = False
    if run_evaluation:
        want_plots = input(f"Show evaluation plots? (Y/n): ").strip().lower()
        show_evaluation_plots = want_plots not in ['n', 'no']

    # Run simulation
    print(f"\n🚀 STARTING SIMULATION...")
    print(f"{'🔧 Mode: L-MPC (Linear Model)' if model_type == 'linear' else '🧠 Mode: K-MPC (Kernel Model)'}")
    print("=" * 50)

    try:
        result = simulate_mpc_with_config(
            hist_df,
            config_name=selected_config,
            manual_overrides=manual_overrides,
            progress_callback=progress_callback,
            run_evaluation=run_evaluation,
            show_evaluation_plots=show_evaluation_plots
        )

        if result is None:
            print("❌ simulate_mpc_with_config returned None")
            return

        results_df, metrics = result

        # Display results
        print("\n📊 SIMULATION RESULTS:")
        print("=" * 40)
        print(f"📈 Processed steps: {len(results_df)}")

        # Key metrics
        key_metrics = ['test_mse_total', 'test_rmse_conc_fe', 'test_rmse_conc_mass']
        for metric in key_metrics:
            if metric in metrics:
                print(f"📊 {metric}: {metrics[metric]:.6f}")

        # Model type summary
        print(f"\n🎯 USED MODEL:")
        if model_type == 'linear':
            linear_type = manual_overrides.get('linear_type', base_config.get('linear_type', 'ols'))
            print(f"   🔧 L-MPC: {linear_type} (degree {base_config.get('poly_degree', 1)})")
        else:
            kernel = manual_overrides.get('kernel', base_config.get('kernel', 'rbf'))
            print(f"   🧠 K-MPC: {model_type} + {kernel}")

        # Show saved files
        saved_results = list_saved_results()
        if saved_results:
            latest = saved_results[0]
            print(f"\n💾 SAVED:")
            print(f"   📁 {latest['file']}")
            print(f"   📊 Size: {latest['size_mb']:.2f} MB")

        print("\n✅ Simulation completed successfully!")

    except Exception as e:
        print(f"\n❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()