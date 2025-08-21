# sim.py - Refactored MPC Simulation Module

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from collections import deque
from datetime import datetime
import json
import time
from pathlib import Path

# Import required modules
from data_gen import StatefulDataGenerator
from model import KernelModel
from objectives import MaxIronMassTrackingObjective
from mpc import MPCController
from utils import (
    run_post_simulation_analysis_enhanced, 
    diagnose_mpc_behavior, 
    diagnose_ekf_detailed
)
from ekf import ExtendedKalmanFilter
from anomaly_detector import SignalAnomalyDetector
from maf import MovingAverageFilter
from evaluation_simple import evaluate_simulation, print_evaluation_report
from config_manager import (
    simulate_mpc_with_config, list_configs, create_default_configs,
    prompt_manual_adjustments, load_config, list_saved_results,
    get_config_info
)
from evaluation_storage import quick_save, quick_load  
from evaluation_database import quick_add_to_database


# =============================================================================
# Data Preparation Functions
# =============================================================================

def prepare_simulation_data(reference_df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[StatefulDataGenerator, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepare simulation data by creating generator, generating time series with anomalies,
    filtering them with SignalAnomalyDetector, and building lagged matrices X, Y.
    """
    print("Step 1: Generating simulation data...")

    # Initialize plant generator
    true_gen = StatefulDataGenerator(
        reference_df,
        ore_flow_var_pct=3.0,
        time_step_s=params['time_step_s'],
        time_constants_s=params['time_constants_s'],
        dead_times_s=params['dead_times_s'],
        true_model_type=params['plant_model_type'],
        seed=params['seed']
    )

    # Configure anomalies
    anomaly_config = StatefulDataGenerator.generate_anomaly_config(
        N_data=params['N_data'],
        train_frac=params['train_size'],
        val_frac=params['val_size'],
        test_frac=params['test_size'],
        seed=params['seed']
    )
    
    # Generate full time series with artifacts
    df_true_orig = true_gen.generate(
        T=params['N_data'],
        control_pts=params['control_pts'],
        n_neighbors=params['n_neighbors'],
        noise_level=params['noise_level'],
        anomaly_config=anomaly_config
    )
    
    # Apply nonlinear transformations if enabled
    if params['enable_nonlinear']:
        nonlinear_config = params['nonlinear_config']
        df_true = true_gen.generate_nonlinear_variant(
            base_df=df_true_orig,
            non_linear_factors=nonlinear_config,
            noise_level='none',
            anomaly_config=None
        )
    else:
        df_true = df_true_orig
    
    # Offline anomaly filtering
    anomaly_params = params.get('anomaly_params', {})
    feed_detector = SignalAnomalyDetector(**anomaly_params)
    ore_detector = SignalAnomalyDetector(**anomaly_params)

    filtered_feed = []
    filtered_ore = []
    for raw_fe, raw_ore in zip(df_true['feed_fe_percent'], df_true['ore_mass_flow']):
        filtered_feed.append(feed_detector.update(raw_fe))
        filtered_ore.append(ore_detector.update(raw_ore))

    # Replace raw columns with filtered ones
    df_true = df_true.copy()
    df_true['feed_fe_percent'] = filtered_feed
    df_true['ore_mass_flow'] = filtered_ore

    # Create lagged dataset
    X, Y_full = StatefulDataGenerator.create_lagged_dataset(df_true, lags=params['lag'])
    Y = Y_full[:, [0, 2]]  # Select concentrate_fe and concentrate_mass columns

    return true_gen, df_true, X, Y


def split_and_scale_data(X: np.ndarray, Y: np.ndarray, params: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], StandardScaler, StandardScaler]:
    """
    Split data into train/validation/test sets and scale them.
    """
    n = X.shape[0]
    n_train = int(params['train_size'] * n)
    n_val = int(params['val_size'] * n)

    # Split data
    splits = {
        'X_train': X[:n_train], 'Y_train': Y[:n_train],
        'X_val': X[n_train:n_train + n_val], 'Y_val': Y[n_train:n_train + n_val],
        'X_test': X[n_train + n_val:], 'Y_test': Y[n_train + n_val:]
    }

    # Initialize and fit scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Scale all splits
    splits['X_train_scaled'] = x_scaler.fit_transform(splits['X_train'])
    splits['Y_train_scaled'] = y_scaler.fit_transform(splits['Y_train'])
    splits['X_val_scaled'] = x_scaler.transform(splits['X_val'])
    splits['Y_val_scaled'] = y_scaler.transform(splits['Y_val'])
    splits['X_test_scaled'] = x_scaler.transform(splits['X_test'])
    splits['Y_test_scaled'] = y_scaler.transform(splits['Y_test'])
    
    return splits, x_scaler, y_scaler


# =============================================================================
# MPC and EKF Initialization Functions
# =============================================================================

def create_kernel_model(params: Dict[str, Any]) -> KernelModel:
    """Create and configure kernel model based on model type."""
    model_type = params['model_type'].lower()
    
    if model_type == 'linear':
        return KernelModel(
            model_type=model_type,
            linear_type=params.get('linear_type', 'ols'),
            poly_degree=params.get('poly_degree', 1),
            include_bias=params.get('include_bias', True),
            alpha=params.get('alpha', 1.0),
            find_optimal_params=params.get('find_optimal_params', False)
        )
    else:
        return KernelModel(
            model_type=model_type,
            kernel=params.get('kernel', 'rbf'),
            find_optimal_params=params.get('find_optimal_params', True)
        )


def configure_trust_region_params(model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Configure trust region parameters based on model type."""
    if model_type == 'linear':
        return {
            'adaptive_trust_region': params.get('adaptive_trust_region', False),
            'initial_trust_radius': params.get('initial_trust_radius', 1.2),
            'min_trust_radius': params.get('min_trust_radius', 0.8),
            'max_trust_radius': params.get('max_trust_radius', 3.0),
            'trust_decay_factor': params.get('trust_decay_factor', 0.9)
        }
    else:
        return {
            'adaptive_trust_region': params.get('adaptive_trust_region', True),
            'initial_trust_radius': params.get('initial_trust_radius', 1.0),
            'min_trust_radius': params.get('min_trust_radius', 0.1),
            'max_trust_radius': params.get('max_trust_radius', 5.0),
            'trust_decay_factor': params.get('trust_decay_factor', 0.8)
        }


def initialize_mpc_controller(params: Dict[str, Any], x_scaler: StandardScaler, y_scaler: StandardScaler) -> MPCController:
    """Initialize enhanced MPC controller with adaptive trust region and linear model support."""
    print("Step 2: Initializing enhanced MPC controller...")
    
    model_type = params['model_type'].lower()
    print(f"Setting up {model_type} model...")
    
    # Create process model
    kernel_model = create_kernel_model(params)
    
    # Scale setpoints and constraints
    ref_point_scaled = y_scaler.transform(np.array([[params['ref_fe'], params['ref_mass']]]))[0]
    y_max_scaled = y_scaler.transform(np.array([[params['y_max_fe'], params['y_max_mass']]]))[0]

    # Create objective function
    objective = MaxIronMassTrackingObjective(
        λ=params['λ_obj'], w_fe=params['w_fe'], w_mass=params['w_mass'],
        ref_fe=ref_point_scaled[0], ref_mass=ref_point_scaled[1], K_I=params['K_I']
    )
    
    # Calculate penalty weights
    avg_tracking_weight = (params['w_fe'] + params['w_mass']) / 2.0
    rho_y_val = avg_tracking_weight * 1000
    rho_du_val = params['λ_obj'] * 100

    # Configure trust region parameters
    trust_params = configure_trust_region_params(model_type, params)
    
    # Create enhanced controller
    mpc = MPCController(
        model=kernel_model, 
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
        y_max=list(y_max_scaled) if params['use_soft_constraints'] else None,
        rho_y=rho_y_val, 
        rho_delta_u=rho_du_val, 
        rho_trust=params['rho_trust'],
        linearization_check_enabled=params.get('linearization_check_enabled', True),
        max_linearization_distance=params.get('max_linearization_distance', 2.0),
        **trust_params
    )
    
    return mpc


def estimate_initial_disturbances(Y_train_scaled: np.ndarray) -> np.ndarray:
    """Estimate initial disturbances from training data."""
    if len(Y_train_scaled) > 100:
        early_period = Y_train_scaled[:50]
        late_period = Y_train_scaled[-50:]
        
        fe_drift = np.mean(late_period[:, 0]) - np.mean(early_period[:, 0])
        mass_drift = np.mean(late_period[:, 1]) - np.mean(early_period[:, 1])
        
        fe_std = np.std(Y_train_scaled[:, 0])
        mass_std = np.std(Y_train_scaled[:, 1])
        
        max_disturbance_fe = 0.5 * fe_std
        max_disturbance_mass = 0.5 * mass_std
        
        fe_bias = np.clip(fe_drift, -max_disturbance_fe, max_disturbance_fe)
        mass_bias = np.clip(mass_drift, -max_disturbance_mass, max_disturbance_mass)
        
        return np.array([fe_bias, mass_bias])
    else:
        return np.array([0.1, 0.0])


def configure_covariance_matrices(Y_train_scaled: np.ndarray, params: Dict[str, Any], initial_disturbances: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Configure P0, Q, and R covariance matrices."""
    n_phys = (params['lag'] + 1) * 3
    n_dist = 2
    
    # Initial covariance matrix P0
    base_p0 = params['P0']
    disturbance_uncertainty_fe = max(0.1, abs(initial_disturbances[0]) * 2)
    disturbance_uncertainty_mass = max(0.1, abs(initial_disturbances[1]) * 2)
    
    P0 = np.eye(n_phys + n_dist) * base_p0
    P0[n_phys, n_phys] = disturbance_uncertainty_fe
    P0[n_phys + 1, n_phys + 1] = disturbance_uncertainty_mass
    
    # Process noise matrix Q
    Q_phys = np.eye(n_phys) * params['Q_phys']
    
    if len(Y_train_scaled) > 50:
        fe_variability = np.std(np.diff(Y_train_scaled[:, 0]))
        mass_variability = np.std(np.diff(Y_train_scaled[:, 1]))
        
        Q_dist_fe = max(params['Q_dist'], fe_variability * 0.1)
        Q_dist_mass = max(params['Q_dist'], mass_variability * 0.1)
        Q_dist = np.diag([Q_dist_fe, Q_dist_mass])
    else:
        Q_dist = np.eye(n_dist) * params['Q_dist']
    
    Q = np.block([[Q_phys, np.zeros((n_phys, n_dist))], 
                  [np.zeros((n_dist, n_phys)), Q_dist]])
    
    # Measurement noise matrix R
    base_R_factor = params['R']
    measurement_variances = np.var(Y_train_scaled, axis=0)
    min_R_values = np.array([1e-4, 1e-4])
    R_values = np.maximum(measurement_variances * base_R_factor, min_R_values)
    R = np.diag(R_values)
    
    return P0, Q, R


def initialize_ekf(mpc: MPCController, scalers: Tuple[StandardScaler, StandardScaler], 
                  hist0_unscaled: np.ndarray, Y_train_scaled: np.ndarray, 
                  lag: int, params: Dict[str, Any]) -> ExtendedKalmanFilter:
    """Initialize Extended Kalman Filter with enhanced initial estimates."""
    print("Step 4: Initializing Kalman Filter (EKF)...")
       
    x_scaler, y_scaler = scalers
    n_phys, n_dist = (lag + 1) * 3, 2
    
    # Estimate initial disturbances
    initial_disturbances = estimate_initial_disturbances(Y_train_scaled)
    
    # Form augmented initial state
    x0_aug = np.hstack([hist0_unscaled.flatten(), initial_disturbances])
    
    # Configure covariance matrices
    P0, Q, R = configure_covariance_matrices(Y_train_scaled, params, initial_disturbances)
    
    # Create EKF with enhanced parameters
    ekf = ExtendedKalmanFilter(
        mpc.model, x_scaler, y_scaler, x0_aug, P0, Q, R, lag,
        beta_R=params.get('beta_R', 0.1),
        q_adaptive_enabled=params.get('q_adaptive_enabled', True),
        q_alpha=params.get('q_alpha', 0.995),
        q_nis_threshold=params.get('q_nis_threshold', 1.8)        
    )
    
    print("EKF initialized with enhanced parameters")
    return ekf


def train_and_evaluate_model(mpc: MPCController, data: Dict[str, np.ndarray], 
                           y_scaler: StandardScaler) -> Tuple[Dict[str, float], Dict[str, list]]:
    """Train process model and evaluate its quality with timing measurements."""
    print("Step 3: Training and evaluating process model...")
    
    # Measure training time
    start_time = time.time()
    mpc.fit(data['X_train_scaled'], data['Y_train_scaled'])
    training_time = time.time() - start_time
    
    print(f"Model training time: {training_time:.2f} sec")

    # Evaluate model performance
    y_pred_scaled = mpc.model.predict(data['X_test_scaled'])
    y_pred_orig = y_scaler.inverse_transform(y_pred_scaled)
    
    test_mse = mean_squared_error(data['Y_test'], y_pred_orig)
    print(f"Total model error on test data (MSE): {test_mse:.4f}")
    
    # Calculate detailed metrics
    metrics = {'test_mse_total': test_mse}
    output_columns = ['conc_fe', 'conc_mass']
    for i, col in enumerate(output_columns):
        rmse = np.sqrt(mean_squared_error(data['Y_test'][:, i], y_pred_orig[:, i]))
        metrics[f'test_rmse_{col}'] = rmse
        print(f"RMSE for {col}: {rmse:.3f}")
    
    # Initialize timing metrics
    timing_metrics = {
        'initial_training_time': training_time,
        'retraining_times': [],
        'prediction_times': []
    }
        
    return metrics, timing_metrics


# =============================================================================
# Simulation Loop Functions
# =============================================================================

def setup_simulation_environment(true_gen: StatefulDataGenerator, df_true: pd.DataFrame, 
                               data: Dict[str, np.ndarray], params: Dict[str, Any]) -> Tuple[int, np.ndarray, np.ndarray]:
    """Setup initial simulation environment."""
    n_total = len(df_true) - params['lag'] - 1
    n_train = int(params['train_size'] * n_total)
    n_val = int(params['val_size'] * n_total)
    test_idx_start = params['lag'] + 1 + n_train + n_val

    hist0_unscaled = df_true[['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']].iloc[
        test_idx_start - (params['lag'] + 1): test_idx_start
    ].values

    df_run = df_true.iloc[test_idx_start:]
    d_all = df_run[['feed_fe_percent', 'ore_mass_flow']].values

    return test_idx_start, hist0_unscaled, d_all


def initialize_filters_and_buffers(params: Dict[str, Any], data: Dict[str, np.ndarray]) -> Tuple[MovingAverageFilter, MovingAverageFilter, SignalAnomalyDetector, SignalAnomalyDetector, deque, list]:
    """Initialize online filters and retraining buffers."""
    window_size = 4
    feed_filter = MovingAverageFilter(window_size)
    ore_filter = MovingAverageFilter(window_size)

    # Online anomaly detectors
    anomaly_config = params.get('anomaly_params', {})
    feed_anomaly_detector = SignalAnomalyDetector(**anomaly_config)
    ore_anomaly_detector = SignalAnomalyDetector(**anomaly_config)

    # Retraining setup
    retraining_buffer = None
    innovation_monitor = []
    
    if params['enable_retraining']:
        print(f"Dynamic retraining ENABLED. Window: {params['retrain_window_size']}, Period: {params['retrain_period']}")
        retraining_buffer = deque(maxlen=params['retrain_window_size'])
        initial_train_data = list(zip(data['X_train_scaled'], data['Y_train_scaled']))
        retraining_buffer.extend(initial_train_data)
        innovation_monitor = deque(maxlen=params['retrain_period'])

    return feed_filter, ore_filter, feed_anomaly_detector, ore_anomaly_detector, retraining_buffer, innovation_monitor


def process_single_simulation_step(t: int, d_all: np.ndarray, feed_filter: MovingAverageFilter, 
                                 ore_filter: MovingAverageFilter, feed_detector: SignalAnomalyDetector, 
                                 ore_detector: SignalAnomalyDetector, ekf: ExtendedKalmanFilter, 
                                 mpc: MPCController, params: Dict[str, Any], u_prev: float, 
                                 timing_metrics: Dict[str, list]) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Process a single simulation step."""
    try:
        # Get raw measurements
        feed_fe_raw, ore_flow_raw = d_all[t, :]

        # Online anomaly filtering
        feed_fe_filtered = feed_detector.update(feed_fe_raw)
        ore_flow_filtered = ore_detector.update(ore_flow_raw)

        # Coarse smoothing
        d_filtered = np.array([
            feed_filter.update(feed_fe_filtered),
            ore_filter.update(ore_flow_filtered)
        ])

        # EKF prediction
        ekf.predict(u_prev, d_filtered)

        # Update MPC history
        x_est_phys_unscaled = ekf.x_hat[:ekf.n_phys].reshape(params['lag'] + 1, 3)
        mpc.reset_history(x_est_phys_unscaled)
        mpc.d_hat = ekf.x_hat[ekf.n_phys:]

        # Measure prediction time
        pred_start_time = time.time()
        
        # Get current state and predict next output
        current_state = x_est_phys_unscaled.flatten().reshape(1, -1)
        current_state_scaled = mpc.x_scaler.transform(current_state)
        y_pred_scaled = mpc.model.predict(current_state_scaled)[0]
        y_pred_unscaled = mpc.y_scaler.inverse_transform(y_pred_scaled.reshape(1, -1))[0]

        # MPC optimization
        d_seq = np.repeat(d_filtered.reshape(1, -1), params['Np'], axis=0)
        u_seq = mpc.optimize(d_seq, u_prev)
        u_current = u_prev if u_seq is None else float(u_seq[0])
        
        # Record prediction time
        prediction_time = time.time() - pred_start_time
        timing_metrics['prediction_times'].append(prediction_time)

        # Diagnostic output
        if t % 10 == 0:
            try:
                diagnose_mpc_behavior(mpc, t, u_seq, u_prev, d_seq)
            except Exception as diag_error:
                print(f"Warning: MPC diagnostics failed at step {t}: {diag_error}")

        return d_filtered, u_current, u_seq, y_pred_unscaled
        
    except Exception as e:
        print(f"ERROR in simulation step {t}: {e}")
        # Return safe fallback values
        d_filtered = np.array([feed_fe_raw if 'feed_fe_raw' in locals() else 0.0, 
                              ore_flow_raw if 'ore_flow_raw' in locals() else 0.0])
        u_current = u_prev
        u_seq = None
        y_pred_unscaled = np.array([0.0, 0.0])  # Safe fallback
        return d_filtered, u_current, u_seq, y_pred_unscaled


def handle_retraining(t: int, mpc: MPCController, ekf: ExtendedKalmanFilter, 
                     retraining_buffer: Optional[deque], innovation_monitor: list, 
                     params: Dict[str, Any], timing_metrics: Dict[str, list], 
                     retrain_cooldown_timer: int) -> int:
    """Handle model retraining logic."""
    if not params['enable_retraining'] or retraining_buffer is None:
        return retrain_cooldown_timer
        
    if ekf.last_innovation is not None:
        innov_norm = np.linalg.norm(ekf.last_innovation)
        innovation_monitor.append(innov_norm)

    # Check retraining conditions
    if (t > 0 and t % params['retrain_period'] == 0 and 
        len(innovation_monitor) == params['retrain_period'] and
        retrain_cooldown_timer == 0):

        avg_innovation = float(np.mean(innovation_monitor))
        should_retrain = avg_innovation > params['retrain_innov_threshold']
        
        # Additional check: linearization quality
        if (hasattr(mpc, 'linearization_quality_history') and 
            len(mpc.linearization_quality_history) > 10):
            
            if isinstance(mpc.linearization_quality_history[-1], dict):
                recent_distances = [h['euclidean_distance'] for h in mpc.linearization_quality_history[-10:]]
            else:
                recent_distances = mpc.linearization_quality_history[-10:]
            
            recent_lin_quality = np.mean(recent_distances)
            lin_threshold = params.get('retrain_linearization_threshold', 1.5)
            
            if recent_lin_quality > lin_threshold:
                print(f"Additional trigger: poor linearization quality ({recent_lin_quality:.3f} > {lin_threshold})")
                should_retrain = True

        if should_retrain:
            print(f"\nRETRAINING TRIGGER at step {t}! Average innovation: {avg_innovation:.4f} > {params['retrain_innov_threshold']:.4f}")

            retrain_data = list(retraining_buffer)
            X_retrain = np.array([p[0] for p in retrain_data])
            Y_retrain = np.array([p[1] for p in retrain_data])

            print(f"mpc.fit() on {len(X_retrain)} samples...")
            
            # Measure retraining time
            retrain_start_time = time.time()
            mpc.fit(X_retrain, Y_retrain)
            retrain_time = time.time() - retrain_start_time
            timing_metrics['retraining_times'].append(retrain_time)
            
            print(f"Retraining completed in {retrain_time:.3f} sec.")
            
            # Reset trust region after retraining
            if hasattr(mpc, 'reset_trust_region'):
                mpc.reset_trust_region()
                print("Trust region reset.\n")

            innovation_monitor.clear()
            retrain_cooldown_timer = params['retrain_period'] * 2

    return retrain_cooldown_timer


def collect_simulation_data(t: int, mpc: MPCController, ekf: ExtendedKalmanFilter, 
                          true_gen: StatefulDataGenerator, feed_fe_raw: float, 
                          ore_flow_raw: float, u_current: float, u_seq: Optional[np.ndarray], 
                          y_pred_unscaled: np.ndarray, retraining_buffer: Optional[deque], 
                          params: Dict[str, Any]) -> Dict[str, Any]:
    """Collect data from current simulation step."""
    # Real process step
    y_full = true_gen.step(feed_fe_raw, ore_flow_raw, u_current)

    # EKF correction
    y_measured_unscaled = y_full[['concentrate_fe_percent', 'concentrate_mass_flow']].values.flatten()
    ekf.update(y_measured_unscaled)

    # Collect data for diagnostics
    step_data = {
        'y_true': y_measured_unscaled.copy(),
        'y_pred': y_pred_unscaled.copy(),
        'x_est': ekf.x_hat.copy(),
        'innovation': ekf.last_innovation.copy() if ekf.last_innovation is not None else np.zeros(2)
    }

    # Collect trust region statistics
    trust_stats = None
    if hasattr(mpc, 'get_trust_region_stats'):
        try:
            trust_stats = mpc.get_trust_region_stats()
        except Exception as e:
            print(f"Error getting trust region statistics: {e}")
    
    if trust_stats is None:
        trust_stats = {
            'current_radius': getattr(mpc, 'current_trust_radius', getattr(mpc, 'trust_radius', 1.0)),
            'radius_increased': False,
            'radius_decreased': False,
            'step': t,
            'optimization_success': u_seq is not None
        }
    
    if isinstance(trust_stats, dict):
        trust_stats['step'] = t
        trust_stats['optimization_success'] = u_seq is not None
    
    step_data['trust_stats'] = trust_stats

    # Collect MPC plan - FIXED: Store both dict format and simple array for compatibility
    if u_seq is not None:
        try:
            # Store dict format for detailed analysis
            step_data['u_plan_detailed'] = {
                'plan': u_seq.copy(),
                'step': t,
                'horizon_length': len(u_seq),
                'first_action': float(u_seq[0]) if len(u_seq) > 0 else None
            }
            # Store simple array for plotting compatibility
            step_data['u_plan'] = u_seq.copy()
        except Exception as e:
            step_data['u_plan'] = u_seq.copy() if hasattr(u_seq, 'copy') else list(u_seq)
            step_data['u_plan_detailed'] = None
            print(f"Step {t}: error saving MPC plan - {e}")
    else:
        # Store None for simple format, dict for detailed
        step_data['u_plan'] = None
        step_data['u_plan_detailed'] = {
            'plan': None,
            'step': t,
            'optimization_failed': True
        }

    # Collect disturbance estimates
    if mpc.d_hat is not None:
        try:
            d_hat_orig = mpc.y_scaler.inverse_transform(mpc.d_hat.reshape(1, -1))[0]
            step_data['d_hat'] = d_hat_orig.copy()
        except Exception as e:
            step_data['d_hat'] = mpc.d_hat.copy()
            if t % 50 == 0:
                print(f"Step {t}: error scaling d_hat - {e}")
    else:
        step_data['d_hat'] = np.zeros(2)

    # Update retraining buffer
    if params['enable_retraining'] and retraining_buffer is not None:
        new_x_unscaled = mpc.x_hist.flatten().reshape(1, -1)
        new_y_unscaled = y_measured_unscaled.reshape(1, -1)

        new_x_scaled = mpc.x_scaler.transform(new_x_unscaled)
        new_y_scaled = mpc.y_scaler.transform(new_y_unscaled)

        retraining_buffer.append((new_x_scaled[0], new_y_scaled[0]))

    # Record result for output
    y_measured = y_full.iloc[0]
    record = {
        'feed_fe_percent': y_measured.feed_fe_percent,
        'ore_mass_flow': y_measured.ore_mass_flow,
        'solid_feed_percent': u_current,
        'conc_fe': y_measured.concentrate_fe_percent,
        'tail_fe': y_measured.tailings_fe_percent,
        'conc_mass': y_measured.concentrate_mass_flow,
        'tail_mass': y_measured.tailings_mass_flow,
        'mass_pull_pct': y_measured.mass_pull_percent,
        'fe_recovery_percent': y_measured.fe_recovery_percent,
    }

    return {**step_data, 'record': record}


def run_simulation_loop(true_gen: StatefulDataGenerator, mpc: MPCController, 
                       ekf: ExtendedKalmanFilter, df_true: pd.DataFrame, 
                       data: Dict[str, np.ndarray], scalers: Tuple[StandardScaler, StandardScaler], 
                       params: Dict[str, Any], timing_metrics: Dict[str, list], 
                       progress_callback: Optional[Callable] = None) -> Tuple[pd.DataFrame, Dict]:
    """Enhanced simulation loop with trust region monitoring and timing measurements."""
    print("Step 5: Running enhanced simulation loop...")
    
    try:
        x_scaler, y_scaler = scalers

        # Setup simulation environment
        test_idx_start, hist0_unscaled, d_all = setup_simulation_environment(true_gen, df_true, data, params)
        
        # Initialize MPC and generator states
        mpc.reset_history(hist0_unscaled)
        true_gen.reset_state(hist0_unscaled)

        df_run = df_true.iloc[test_idx_start:]
        T_sim = len(df_run) - (params['lag'] + 1)
        
        if T_sim <= 0:
            print("ERROR: Not enough data for simulation")
            return pd.DataFrame(), {}

        # Initialize filters and buffers
        feed_filter, ore_filter, feed_detector, ore_detector, retraining_buffer, innovation_monitor = initialize_filters_and_buffers(params, data)

        # Initialize data collection structures
        records = []
        simulation_data = {
            'y_true_hist': [], 'x_hat_hist': [], 'P_hist': [], 'innov_hist': [], 'R_hist': [],
            'u_seq_hist': [], 'd_hat_hist': [], 'trust_region_stats_hist': [], 'linearization_quality_hist': [],
            'y_true_seq': [], 'y_pred_seq': [], 'x_est_seq': [], 'innovation_seq': []
        }

        u_prev = float(hist0_unscaled[-1, 2])
        retrain_cooldown_timer = 0

        # Main simulation loop
        for t in range(T_sim):
            try:
                if progress_callback:
                    progress_callback(t, T_sim, f"Simulation step {t + 1}/{T_sim}")

                # Process single step
                d_filtered, u_current, u_seq, y_pred_unscaled = process_single_simulation_step(
                    t, d_all, feed_filter, ore_filter, feed_detector, ore_detector, 
                    ekf, mpc, params, u_prev, timing_metrics
                )

                # Handle retraining
                retrain_cooldown_timer = handle_retraining(
                    t, mpc, ekf, retraining_buffer, innovation_monitor, 
                    params, timing_metrics, retrain_cooldown_timer
                )
                
                if retrain_cooldown_timer > 0:
                    retrain_cooldown_timer -= 1

                # Collect all step data
                step_data = collect_simulation_data(
                    t, mpc, ekf, true_gen, d_all[t, 0], d_all[t, 1], u_current, 
                    u_seq, y_pred_unscaled, retraining_buffer, params
                )

                # Store collected data
                simulation_data['y_true_hist'].append(step_data['y_true'])
                simulation_data['x_hat_hist'].append(step_data['x_est'])
                simulation_data['P_hist'].append(ekf.P.copy())
                simulation_data['innov_hist'].append(step_data['innovation'])
                simulation_data['R_hist'].append(ekf.R.copy())
                # FIXED: Store simple format for plotting compatibility
                simulation_data['u_seq_hist'].append(step_data['u_plan'])
                simulation_data['d_hat_hist'].append(step_data['d_hat'])
                simulation_data['trust_region_stats_hist'].append(step_data['trust_stats'])
                
                # Store diagnostic sequences
                simulation_data['y_true_seq'].append(step_data['y_true'])
                simulation_data['y_pred_seq'].append(step_data['y_pred'])
                simulation_data['x_est_seq'].append(step_data['x_est'])
                simulation_data['innovation_seq'].append(step_data['innovation'])
                
                # Store linearization quality if available
                if hasattr(mpc, 'linearization_quality_history') and mpc.linearization_quality_history:
                    if isinstance(mpc.linearization_quality_history[-1], dict):
                        simulation_data['linearization_quality_hist'].append(mpc.linearization_quality_history[-1]['euclidean_distance'])
                    else:
                        simulation_data['linearization_quality_hist'].append(mpc.linearization_quality_history[-1])

                records.append(step_data['record'])
                u_prev = u_current
                
            except Exception as step_error:
                print(f"ERROR in simulation step {t}: {step_error}")
                # Continue with next step instead of stopping completely
                continue

        if progress_callback:
            progress_callback(T_sim, T_sim, "Simulation completed")

        # Check if we have any results
        if not records:
            print("ERROR: No simulation records were collected")
            return pd.DataFrame(), {}

        # Detailed EKF diagnostics
        try:
            diagnose_ekf_detailed(ekf, simulation_data['y_true_seq'], simulation_data['y_pred_seq'], 
                                 simulation_data['x_est_seq'], simulation_data['innovation_seq'])
        except Exception as diag_error:
            print(f"Warning: EKF diagnostics failed: {diag_error}")
        
        # Print timing statistics
        print(f"\nTIMING STATISTICS:")
        print(f"   • Initial training: {timing_metrics['initial_training_time']:.2f} sec")
        if timing_metrics['retraining_times']:
            avg_retrain = np.mean(timing_metrics['retraining_times'])
            print(f"   • Average retraining time: {avg_retrain:.3f} sec ({len(timing_metrics['retraining_times'])} times)")
        else:
            print(f"   • Retraining: not performed")
        
        if timing_metrics['prediction_times']:
            avg_pred = np.mean(timing_metrics['prediction_times']) * 1000
            print(f"   • Average prediction time: {avg_pred:.2f} ms")
            print(f"   • Throughput capacity: {1000/avg_pred:.1f} predictions/sec")

        # Prepare extended analysis data with error handling
        try:
            analysis_data = {
                "y_true": np.vstack(simulation_data['y_true_hist']) if simulation_data['y_true_hist'] else np.array([]),
                "x_hat": np.vstack(simulation_data['x_hat_hist']) if simulation_data['x_hat_hist'] else np.array([]),
                "P": np.stack(simulation_data['P_hist']) if simulation_data['P_hist'] else np.array([]),
                "innov": np.vstack(simulation_data['innov_hist']) if simulation_data['innov_hist'] else np.array([]),
                "R": np.stack(simulation_data['R_hist']) if simulation_data['R_hist'] else np.array([]),
                "u_seq": simulation_data['u_seq_hist'],
                "d_hat": np.vstack(simulation_data['d_hat_hist']) if simulation_data['d_hat_hist'] else np.array([]),
                "trust_region_stats": simulation_data['trust_region_stats_hist'],
                "linearization_quality": simulation_data['linearization_quality_hist'],
                "y_true_seq": simulation_data['y_true_seq'],
                "y_pred_seq": simulation_data['y_pred_seq'],
                "x_est_seq": simulation_data['x_est_seq'],
                "innovation_seq": simulation_data['innovation_seq'],
                "timing_metrics": timing_metrics
            }

            # Add test disturbances data
            test_idx_start = params['lag'] + 1 + len(data['X_train']) + len(data['X_val'])
            if test_idx_start < len(df_true):
                analysis_data['d_all_test'] = df_true.iloc[test_idx_start:][['feed_fe_percent','ore_mass_flow']].values
            else:
                analysis_data['d_all_test'] = np.array([])

        except Exception as data_error:
            print(f"Warning: Error preparing analysis data: {data_error}")
            analysis_data = {"timing_metrics": timing_metrics}

        # Diagnostic analysis
        try:
            from evaluation_simple import diagnose_analysis_data
            diagnose_analysis_data(analysis_data)
        except ImportError:
            print("diagnose_analysis_data not available")
        except Exception as e:
            print(f"Diagnostic error: {e}")

        return pd.DataFrame(records), analysis_data
        
    except Exception as e:
        print(f"CRITICAL ERROR in simulation loop: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), {}


# =============================================================================
# Main Orchestrator Function
# =============================================================================

def simulate_mpc(reference_df: pd.DataFrame, **kwargs) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """
    Enhanced version of main orchestrator function with linear model support and timing metrics.
    
    Returns:
        Tuple of (results_df, metrics) or (None, None) in case of error
    """
    # Collect all parameters into dictionary
    params = dict(kwargs)
    
    # FIXED: Only set defaults for missing parameters, don't overwrite config values
    defaults = {
        'N_data': 5000, 'control_pts': 1000, 'time_step_s': 5,
        'dead_times_s': {'concentrate_fe_percent': 20.0, 'tailings_fe_percent': 25.0,
                        'concentrate_mass_flow': 20.0, 'tailings_mass_flow': 25.0},
        'time_constants_s': {'concentrate_fe_percent': 8.0, 'tailings_fe_percent': 10.0,
                            'concentrate_mass_flow': 5.0, 'tailings_mass_flow': 7.0},
        'lag': 2, 'Np': 6, 'Nc': 4, 'n_neighbors': 5, 'seed': 0, 'noise_level': 'none',
        'model_type': 'krr', 'kernel': 'rbf', 'find_optimal_params': True,
        'linear_type': 'ols', 'poly_degree': 1, 'include_bias': True, 'alpha': 1.0,
        'λ_obj': 0.1, 'K_I': 0.01, 'w_fe': 7.0, 'w_mass': 1.0,
        'ref_fe': 53.5, 'ref_mass': 57.0, 'train_size': 0.7, 'val_size': 0.15, 'test_size': 0.15,
        'u_min': 20.0, 'u_max': 40.0, 'delta_u_max': 1.0, 'use_disturbance_estimator': True,
        'y_max_fe': 54.5, 'y_max_mass': 58.0, 'rho_trust': 0.1, 'max_trust_radius': 5.0,
        'adaptive_trust_region': True, 'initial_trust_radius': 1.0, 'min_trust_radius': 0.5,
        'trust_decay_factor': 0.8, 'linearization_check_enabled': True, 'max_linearization_distance': 2.0,
        'retrain_linearization_threshold': 1.5, 'use_soft_constraints': True, 'plant_model_type': 'rf',
        'enable_retraining': True, 'retrain_period': 50, 'retrain_window_size': 1000,
        'retrain_innov_threshold': 0.3, 'enable_nonlinear': False, 'run_analysis': True,
        'run_evaluation': True, 'show_evaluation_plots': False, 'tolerance_fe_percent': 2.0,
        'tolerance_mass_percent': 2.0, 'P0': 1e-2, 'Q_phys': 1500, 'Q_dist': 1, 'R': 0.01,
        'q_adaptive_enabled': True, 'q_alpha': 0.99, 'q_nis_threshold': 1.5,
        'anomaly_params': {'window': 25, 'spike_z': 4.0, 'drop_rel': 0.30, 'freeze_len': 5, 'enabled': True},
        'nonlinear_config': {'concentrate_fe_percent': ('pow', 2), 'concentrate_mass_flow': ('pow', 1.5)}
    }
    
    # CRITICAL FIX: Only set defaults for missing keys, preserve config values
    for key, default_value in defaults.items():
        if key not in params:
            params[key] = default_value

    # Print received configuration for debugging
    config_params = ['N_data', 'model_type', 'Np', 'Nc', 'λ_obj', 'initial_trust_radius', 'retrain_period']
    print(f"📋 RECEIVED CONFIGURATION PARAMETERS:")
    for param in config_params:
        if param in params:
            print(f"   • {param}: {params[param]}")

    # Validate linear model parameters
    model_type = params['model_type'].lower()
    if model_type == 'linear':
        print(f"🔧 Setting up L-MPC (Linear model)")
        print(f"   • Type: {params['linear_type']}")
        print(f"   • Polynomial degree: {params['poly_degree']}")
        print(f"   • Bias: {params['include_bias']}")
        
        if params['linear_type'] not in ['ols', 'ridge', 'lasso']:
            print(f"⚠️ Invalid linear_type '{params['linear_type']}', using 'ols'")
            params['linear_type'] = 'ols'
            
        if not (1 <= params['poly_degree'] <= 3):
            print(f"⚠️ Invalid poly_degree {params['poly_degree']}, using 1")
            params['poly_degree'] = 1
            
        if params['linear_type'] in ['ridge', 'lasso'] and params['alpha'] <= 0:
            print(f"⚠️ Invalid alpha {params['alpha']}, using 1.0")
            params['alpha'] = 1.0
    else:
        print(f"🧠 Setting up K-MPC (Kernel model: {params['model_type']})")
        print(f"   • Kernel: {params.get('kernel', 'rbf')}")
        print(f"   • Auto parameter search: {params.get('find_optimal_params', True)}")
    
    try:
        # Data preparation
        true_gen, df_true, X, Y = prepare_simulation_data(reference_df, params)
        data, x_scaler, y_scaler = split_and_scale_data(X, Y, params)

        # Initialize enhanced MPC
        mpc = initialize_mpc_controller(params, x_scaler, y_scaler)
        
        # Train and evaluate model
        metrics, timing_metrics = train_and_evaluate_model(mpc, data, y_scaler)
        
        # Initialize EKF
        n_train_pts = len(data['X_train'])
        n_val_pts = len(data['X_val'])
        test_idx_start = params['lag'] + 1 + n_train_pts + n_val_pts
        hist0_unscaled = df_true[['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']].iloc[
            test_idx_start - (params['lag'] + 1): test_idx_start
        ].values
        
        ekf = initialize_ekf(mpc, (x_scaler, y_scaler), hist0_unscaled, data['Y_train_scaled'], params['lag'], params)

        # Run simulation
        results_df, analysis_data = run_simulation_loop(
            true_gen, mpc, ekf, df_true, data, (x_scaler, y_scaler), params, 
            timing_metrics, params.get('progress_callback')
        )
        
        # Check if simulation was successful
        if results_df is None or len(results_df) == 0:
            print("ERROR: Simulation failed to produce results")
            return None, None
        
        # Extended analysis
        if params.get('run_analysis', True):
            try:
                run_post_simulation_analysis_enhanced(results_df, analysis_data, params)
            except Exception as analysis_error:
                print(f"Warning: Post-simulation analysis failed: {analysis_error}")
                print("Continuing without detailed analysis...")
        
        # Performance evaluation
        if params.get('run_evaluation', True):
            print("\n" + "="*60)
            print("🎯 EVALUATING MPC SYSTEM EFFECTIVENESS")
            print("="*60)
            try:
                eval_results = evaluate_simulation(results_df, analysis_data, params)
                simulation_steps = len(results_df)
                print_evaluation_report(eval_results, detailed=True, simulation_steps=simulation_steps)
                
                # Visualization
                if params.get('show_evaluation_plots', False):
                    print("\n📊 Creating evaluation plots...")
                    try:
                        from evaluation_simple import create_evaluation_plots
                        create_evaluation_plots(results_df, eval_results, params)
                    except Exception as plot_error:
                        print(f"⚠️ Error creating plots: {plot_error}")
                        
            except Exception as e:
                print(f"⚠️ Error during evaluation: {e}")
                print("Continuing without evaluation...")
                import traceback
                traceback.print_exc()
            print("="*60)
            
        # Automatic evaluation and saving
        try:
            eval_results = evaluate_simulation(results_df, analysis_data, params)
            
            # Save to file
            file_path = quick_save(
                results_df=results_df,
                eval_results=eval_results,
                analysis_data=analysis_data,
                params=params,
                description=f"Auto-simulation {datetime.now()}"
            )
            
            # Add to database
            eval_id = quick_add_to_database(
                package=quick_load(file_path),
                series_id="production_runs",
                tags=["auto", "production"]
            )
            
            print(f"✅ Simulation saved: file {file_path}, DB ID {eval_id}")
        except Exception as save_error:
            print(f"Warning: Could not save results: {save_error}")
            # Continue without saving instead of failing
        
        return results_df, metrics
        
    except Exception as e:
        print(f"❌ Critical error in simulate_mpc: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == '__main__':
    
    def progress_callback(step, total, msg):
        """Simple callback for console progress output"""
        if step % 20 == 0 or step == total:
            print(f"[{step}/{total}] {msg}")

    # Load data
    try:
        hist_df = pd.read_parquet('processed.parquet')
        print("Data loaded successfully")
    except FileNotFoundError:
        print("Error: file 'processed.parquet' not found.")
        exit(1)
    
    # Create default configs if they don't exist
    available_configs = list_configs()
    if not available_configs:
        print("Creating default configurations...")
        create_default_configs()
        available_configs = list_configs()
    
    # Show available configurations with model types
    print(f"\nAVAILABLE CONFIGURATIONS:")
    print("=" * 50)
    for i, config in enumerate(available_configs, 1):
        try:
            config_info = get_config_info(config)
            if config_info:
                model_type = config_info.get('model_type', 'unknown')
                description = config_info.get('description', 'Description missing')
                
                if model_type.lower() == 'linear':
                    type_marker = "L-MPC"
                elif model_type.lower() in ['krr', 'svr', 'gpr']:
                    type_marker = "K-MPC"
                else:
                    type_marker = ""
                
                print(f"{i}. {config} {type_marker}")
                print(f"   {description}")
                print(f"   Model: {model_type}, N_data: {config_info.get('N_data', '?')}, "
                      f"Np: {config_info.get('Np', '?')}, Nc: {config_info.get('Nc', '?')}")
                print()
            else:
                print(f"{i}. {config} (loading error)")
        except Exception as e:
            print(f"{i}. {config} (error: {e})")
    
    # Select base configuration
    print(f"Select base configuration:")
    choice = input(f"Your choice (1-{len(available_configs)}, default 1): ").strip()
    
    try:
        config_index = int(choice) - 1 if choice else 0
        if 0 <= config_index < len(available_configs):
            selected_config = available_configs[config_index]
        else:
            selected_config = available_configs[0]
    except (ValueError, IndexError):
        selected_config = available_configs[0]
    
    print(f"Selected base configuration: {selected_config}")
    
    # Load base configuration for display
    base_config = load_config(selected_config)
    
    # Show specific parameters for selected model type
    model_type = base_config.get('model_type', 'krr').lower()
    
    if model_type == 'linear':
        print(f"\nL-MPC CONFIGURATION:")
        print(f"   • Linear model type: {base_config.get('linear_type', 'ols')}")
        print(f"   • Polynomial degree: {base_config.get('poly_degree', 1)}")
        print(f"   • Include bias: {base_config.get('include_bias', True)}")
        if base_config.get('linear_type') in ['ridge', 'lasso']:
            print(f"   • Regularization coefficient: {base_config.get('alpha', 1.0)}")
        print(f"   • Auto parameter search: {base_config.get('find_optimal_params', False)}")
    else:
        print(f"\nK-MPC CONFIGURATION:")
        print(f"   • Kernel model type: {model_type}")
        print(f"   • Kernel: {base_config.get('kernel', 'rbf')}")
        print(f"   • Auto parameter search: {base_config.get('find_optimal_params', True)}")
    
    # Show current key parameters
    key_params = ['Np', 'Nc', 'ref_fe', 'ref_mass', 'w_fe', 'w_mass', 'λ_obj', 'N_data', 'control_pts']
    print(f"\nKEY PARAMETERS:")
    for param in key_params:
        if param in base_config:
            print(f"   • {param}: {base_config[param]}")
    
    # Ask about manual adjustments
    want_adjustments = input(f"\nWould you like to make manual adjustments? (y/N): ").strip().lower()
    
    manual_overrides = {}
    if want_adjustments in ['y', 'yes']:
        manual_overrides = prompt_manual_adjustments(base_config)
        
        if manual_overrides:
            print(f"\nPlanned {len(manual_overrides)} adjustments")
            for key, value in manual_overrides.items():
                old_value = base_config.get(key, "not set")
                print(f"   • {key}: {old_value} → {value}")
        else:
            print("No adjustments entered")
    
    # Ask about evaluation and visualization
    want_evaluation = input(f"\nEnable effectiveness evaluation? (Y/n): ").strip().lower()
    run_evaluation = want_evaluation not in ['n', 'no']
    
    show_evaluation_plots = False
    if run_evaluation:
        want_plots = input(f"Show evaluation plots? (Y/n): ").strip().lower()
        show_evaluation_plots = want_plots not in ['n', 'no']
    
    # Run simulation
    print(f"\nSTARTING SIMULATION...")
    if model_type == 'linear':
        print(f"Mode: L-MPC (Linear model)")
    else:
        print(f"Mode: K-MPC (Kernel model)")
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
            print("simulate_mpc_with_config returned None")
            exit(1)
        
        results_df, metrics = result
        
        # Check if results are valid
        if results_df is None or metrics is None:
            print("ERROR: Invalid simulation results")
            exit(1)
        
        if len(results_df) == 0:
            print("ERROR: No simulation data generated")
            exit(1)
        
        # Display results
        print("\nSIMULATION RESULTS:")
        print("=" * 40)
        print(f"Steps processed: {len(results_df)}")
        
        # Key metrics
        key_metrics = ['test_mse_total', 'test_rmse_conc_fe', 'test_rmse_conc_mass']
        for metric in key_metrics:
            if metric in metrics:
                print(f"{metric}: {metrics[metric]:.6f}")
        
        # Show used model type
        print(f"\nUSED MODEL:")
        if model_type == 'linear':
            linear_type = base_config.get('linear_type', 'ols')
            if manual_overrides.get('linear_type'):
                linear_type = manual_overrides['linear_type']
            print(f"   L-MPC: {linear_type} (degree {base_config.get('poly_degree', 1)})")
        else:
            kernel = base_config.get('kernel', 'rbf')
            if manual_overrides.get('kernel'):
                kernel = manual_overrides['kernel']
            print(f"   K-MPC: {model_type} + {kernel}")
        
        # Show saved files
        saved_results = list_saved_results()
        if saved_results:
            latest = saved_results[0]
            print(f"\nSAVED:")
            print(f"   {latest['file']}")
            print(f"   Size: {latest['size_mb']:.2f} MB")
        
        print("\nSimulation completed successfully!")
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()