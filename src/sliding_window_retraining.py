# sliding_window_retraining.py - Enhanced retraining with sliding window strategies

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
import time

class RetrainingStrategy(ABC):
    """Abstract base class for retraining strategies"""
    
    @abstractmethod
    def should_retrain(self, current_step: int, metrics: Dict[str, float]) -> bool:
        """Determine if retraining should occur"""
        pass
    
    @abstractmethod
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get data for retraining"""
        pass
    
    @abstractmethod
    def add_data_point(self, x: np.ndarray, y: np.ndarray, metrics: Dict[str, float]):
        """Add new data point to strategy"""
        pass


class SlidingWindowRetraining(RetrainingStrategy):
    """
    Sliding window retraining strategy with multiple window management policies
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        retrain_period: int = 50,
        innovation_threshold: float = 0.3,
        quality_threshold: float = 1.5,
        window_policy: str = 'fifo',  # 'fifo', 'quality_based', 'adaptive', 'exponential'
        min_retrain_samples: int = 100,
        max_retrain_samples: int = 2000,
        quality_alpha: float = 0.95,
        performance_lookback: int = 10
    ):
        self.window_size = window_size
        self.retrain_period = retrain_period
        self.innovation_threshold = innovation_threshold
        self.quality_threshold = quality_threshold
        self.window_policy = window_policy
        self.min_retrain_samples = min_retrain_samples
        self.max_retrain_samples = max_retrain_samples
        self.quality_alpha = quality_alpha
        self.performance_lookback = performance_lookback
        
        # Data storage
        self.data_buffer = deque(maxlen=max_retrain_samples)
        self.quality_buffer = deque(maxlen=max_retrain_samples)
        self.innovation_monitor = deque(maxlen=retrain_period)
        self.performance_history = deque(maxlen=performance_lookback)
        
        # State tracking
        self.last_retrain_step = 0
        self.cooldown_timer = 0
        self.retrain_count = 0
        self.adaptive_window_size = window_size
        self.current_quality_threshold = quality_threshold
        
        print(f"🔄 Initialized {window_policy.upper()} sliding window retraining:")
        print(f"   • Window size: {window_size}")
        print(f"   • Retrain period: {retrain_period}")
        print(f"   • Innovation threshold: {innovation_threshold}")
        print(f"   • Quality threshold: {quality_threshold}")
    
    def should_retrain(self, current_step: int, metrics: Dict[str, float]) -> bool:
        """Enhanced retraining decision with multiple criteria"""
        # Basic conditions
        if (current_step <= 0 or 
            current_step - self.last_retrain_step < self.retrain_period or
            len(self.data_buffer) < self.min_retrain_samples or
            self.cooldown_timer > 0):
            return False
        
        # Innovation-based trigger
        innovation_trigger = self._check_innovation_trigger()
        
        # Quality-based trigger
        quality_trigger = self._check_quality_trigger()
        
        # Performance degradation trigger
        performance_trigger = self._check_performance_trigger()
        
        # Adaptive threshold adjustment
        if self.window_policy == 'adaptive':
            self._adjust_adaptive_thresholds()
        
        should_retrain = innovation_trigger or quality_trigger or performance_trigger
        
        if should_retrain:
            print(f"\n🔄 RETRAINING TRIGGERED at step {current_step}:")
            print(f"   • Innovation: {innovation_trigger}")
            print(f"   • Quality: {quality_trigger}")
            print(f"   • Performance: {performance_trigger}")
            print(f"   • Available samples: {len(self.data_buffer)}")
        
        return should_retrain
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data based on window policy"""
        if self.window_policy == 'fifo':
            return self._get_fifo_data()
        elif self.window_policy == 'quality_based':
            return self._get_quality_based_data()
        elif self.window_policy == 'adaptive':
            return self._get_adaptive_data()
        elif self.window_policy == 'exponential':
            return self._get_exponential_weighted_data()
        else:
            return self._get_fifo_data()
    
    def add_data_point(self, x: np.ndarray, y: np.ndarray, metrics: Dict[str, float]):
        """Add new data point with quality assessment"""
        # Calculate data quality score
        quality_score = self._calculate_quality_score(x, y, metrics)
        
        # Store data with metadata
        data_point = {
            'x': x.copy(),
            'y': y.copy(),
            'quality': quality_score,
            'timestamp': time.time(),
            'innovation_norm': metrics.get('innovation_norm', 0.0),
            'prediction_error': metrics.get('prediction_error', 0.0)
        }
        
        self.data_buffer.append(data_point)
        self.quality_buffer.append(quality_score)
        
        # Update monitoring
        if 'innovation_norm' in metrics:
            self.innovation_monitor.append(metrics['innovation_norm'])
        
        if 'prediction_error' in metrics:
            self.performance_history.append(metrics['prediction_error'])
    
    def update_after_retraining(self, current_step: int, retrain_time: float, new_performance: float):
        """Update strategy state after retraining"""
        self.last_retrain_step = current_step
        self.cooldown_timer = self.retrain_period
        self.retrain_count += 1
        self.innovation_monitor.clear()
        
        # Adapt parameters based on retraining success
        if self.window_policy == 'adaptive':
            self._adapt_parameters_post_retrain(retrain_time, new_performance)
        
        print(f"   ✅ Retraining #{self.retrain_count} completed in {retrain_time:.3f}s")
        print(f"   📊 New performance: {new_performance:.4f}")
    
    def step(self):
        """Step function to update cooldown and adaptive parameters"""
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
    
    # Private methods for different window policies
    
    def _get_fifo_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Simple FIFO sliding window"""
        recent_data = list(self.data_buffer)[-self.adaptive_window_size:]
        
        X = np.array([point['x'] for point in recent_data])
        Y = np.array([point['y'] for point in recent_data])
        
        return X, Y
    
    def _get_quality_based_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Quality-based data selection"""
        # Sort by quality and take best samples
        all_data = list(self.data_buffer)
        quality_sorted = sorted(all_data, key=lambda x: x['quality'], reverse=True)
        
        # Take top quality samples up to window size
        selected_data = quality_sorted[:self.adaptive_window_size]
        
        # Ensure we have recent data (at least 20% from last 100 samples)
        recent_data = all_data[-100:]
        recent_count = max(1, int(0.2 * len(selected_data)))
        
        # Replace lowest quality with recent data
        if len(recent_data) >= recent_count:
            selected_data = selected_data[:-recent_count] + recent_data[-recent_count:]
        
        X = np.array([point['x'] for point in selected_data])
        Y = np.array([point['y'] for point in selected_data])
        
        print(f"   📊 Quality-based selection: {len(selected_data)} samples")
        return X, Y
    
    def _get_adaptive_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Adaptive window size based on performance"""
        # Adjust window size based on recent performance
        if len(self.performance_history) > 5:
            recent_perf = np.mean(list(self.performance_history)[-5:])
            older_perf = np.mean(list(self.performance_history)[:-5]) if len(self.performance_history) > 10 else recent_perf
            
            if recent_perf > older_perf * 1.1:  # Performance degrading
                self.adaptive_window_size = min(self.max_retrain_samples, 
                                               int(self.adaptive_window_size * 1.2))
            elif recent_perf < older_perf * 0.9:  # Performance improving
                self.adaptive_window_size = max(self.min_retrain_samples,
                                               int(self.adaptive_window_size * 0.9))
        
        return self._get_fifo_data()
    
    def _get_exponential_weighted_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Exponentially weighted data selection"""
        all_data = list(self.data_buffer)[-self.adaptive_window_size:]
        n_samples = len(all_data)
        
        if n_samples == 0:
            return np.array([]), np.array([])
        
        # Calculate exponential weights (more recent = higher weight)
        weights = np.exp(np.linspace(-2, 0, n_samples))
        weights = weights / np.sum(weights)
        
        # Sample based on weights
        sample_indices = np.random.choice(
            n_samples, 
            size=min(n_samples, self.adaptive_window_size),
            p=weights,
            replace=False
        )
        
        selected_data = [all_data[i] for i in sorted(sample_indices)]
        
        X = np.array([point['x'] for point in selected_data])
        Y = np.array([point['y'] for point in selected_data])
        
        return X, Y
    
    def _check_innovation_trigger(self) -> bool:
        """Check if innovation threshold is exceeded"""
        if len(self.innovation_monitor) < self.retrain_period:
            return False
        
        avg_innovation = np.mean(list(self.innovation_monitor))
        return avg_innovation > self.innovation_threshold
    
    def _check_quality_trigger(self) -> bool:
        """Check if data quality has degraded"""
        if len(self.quality_buffer) < 20:
            return False
        
        recent_quality = np.mean(list(self.quality_buffer)[-10:])
        older_quality = np.mean(list(self.quality_buffer)[-20:-10])
        
        return recent_quality < older_quality * 0.8
    
    def _check_performance_trigger(self) -> bool:
        """Check if performance has degraded significantly"""
        if len(self.performance_history) < self.performance_lookback:
            return False
        
        recent_performance = np.mean(list(self.performance_history)[-5:])
        baseline_performance = np.mean(list(self.performance_history)[:-5])
        
        return recent_performance > baseline_performance * 1.3
    
    def _calculate_quality_score(self, x: np.ndarray, y: np.ndarray, metrics: Dict[str, float]) -> float:
        """Calculate quality score for data point"""
        # Base quality from innovation (lower innovation = higher quality)
        innovation_score = 1.0 / (1.0 + metrics.get('innovation_norm', 1.0))
        
        # Prediction error component
        error_score = 1.0 / (1.0 + metrics.get('prediction_error', 1.0))
        
        # Data diversity component (distance from recent data)
        diversity_score = self._calculate_diversity_score(x)
        
        # Combine scores
        quality = 0.4 * innovation_score + 0.4 * error_score + 0.2 * diversity_score
        
        return quality
    
    def _calculate_diversity_score(self, x: np.ndarray) -> float:
        """Calculate diversity score based on distance from existing data"""
        if len(self.data_buffer) < 10:
            return 1.0
        
        # Sample recent data points for comparison
        recent_points = [point['x'] for point in list(self.data_buffer)[-10:]]
        
        if len(recent_points) == 0:
            return 1.0
        
        # Calculate minimum distance to existing points
        distances = [np.linalg.norm(x - point) for point in recent_points]
        min_distance = min(distances)
        
        # Convert to score (higher distance = higher diversity)
        diversity_score = min(1.0, min_distance / 2.0)
        
        return diversity_score
    
    def _adjust_adaptive_thresholds(self):
        """Adjust thresholds based on retraining history"""
        if self.retrain_count < 3:
            return
        
        # If retraining too frequently, increase thresholds
        steps_since_retrain = len(self.performance_history)
        if steps_since_retrain < self.retrain_period * 0.8:
            self.innovation_threshold *= 1.1
            self.current_quality_threshold *= 1.1
        
        # If not retraining enough and performance poor, decrease thresholds
        elif (steps_since_retrain > self.retrain_period * 2.0 and 
              len(self.performance_history) > 5 and
              np.mean(list(self.performance_history)[-5:]) > self.innovation_threshold):
            self.innovation_threshold *= 0.9
            self.current_quality_threshold *= 0.9
    
    def _adapt_parameters_post_retrain(self, retrain_time: float, new_performance: float):
        """Adapt parameters after retraining based on results"""
        # Adjust window size based on retraining time
        if retrain_time > 5.0:  # Too slow
            self.adaptive_window_size = max(self.min_retrain_samples,
                                           int(self.adaptive_window_size * 0.9))
        elif retrain_time < 1.0:  # Very fast, can handle more data
            self.adaptive_window_size = min(self.max_retrain_samples,
                                           int(self.adaptive_window_size * 1.1))
        
        # Store performance for future adaptation
        if hasattr(self, 'post_retrain_performance'):
            self.post_retrain_performance.append(new_performance)
        else:
            self.post_retrain_performance = deque([new_performance], maxlen=10)


class TimeBasedRetraining(RetrainingStrategy):
    """Simple time-based retraining strategy"""
    
    def __init__(self, retrain_interval: int = 100, window_size: int = 500):
        self.retrain_interval = retrain_interval
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.last_retrain_step = 0
    
    def should_retrain(self, current_step: int, metrics: Dict[str, float]) -> bool:
        return (current_step - self.last_retrain_step) >= self.retrain_interval
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.data_buffer) == 0:
            return np.array([]), np.array([])
        
        X = np.array([point['x'] for point in self.data_buffer])
        Y = np.array([point['y'] for point in self.data_buffer])
        return X, Y
    
    def add_data_point(self, x: np.ndarray, y: np.ndarray, metrics: Dict[str, float]):
        self.data_buffer.append({'x': x.copy(), 'y': y.copy()})


class RetrainingManager:
    """Manager class to handle different retraining strategies"""
    
    def __init__(self, strategy: RetrainingStrategy, mpc_controller, scalers: Tuple[StandardScaler, StandardScaler]):
        self.strategy = strategy
        self.mpc_controller = mpc_controller
        self.x_scaler, self.y_scaler = scalers
        self.timing_metrics = {'retraining_times': []}
        
    def add_data_point(self, x_unscaled: np.ndarray, y_unscaled: np.ndarray, metrics: Dict[str, float]):
        """Add new data point to retraining strategy"""
        # Scale data before storing
        x_scaled = self.x_scaler.transform(x_unscaled.reshape(1, -1))[0]
        y_scaled = self.y_scaler.transform(y_unscaled.reshape(1, -1))[0]
        
        self.strategy.add_data_point(x_scaled, y_scaled, metrics)
    
    def check_and_retrain(self, current_step: int, metrics: Dict[str, float]) -> bool:
        """Check if retraining should occur and perform it"""
        if not self.strategy.should_retrain(current_step, metrics):
            return False
        
        # Get training data
        X_train, Y_train = self.strategy.get_training_data()
        
        if len(X_train) == 0:
            print("⚠️ No training data available for retraining")
            return False
        
        # Perform retraining with timing
        start_time = time.time()
        self.mpc_controller.fit(X_train, Y_train)
        retrain_time = time.time() - start_time
        
        # Calculate new performance (simplified)
        if len(Y_train) > 10:
            y_pred = self.mpc_controller.model.predict(X_train[-10:])
            new_performance = np.mean(np.linalg.norm(Y_train[-10:] - y_pred, axis=1))
        else:
            new_performance = 0.0
        
        # Update strategy
        if hasattr(self.strategy, 'update_after_retraining'):
            self.strategy.update_after_retraining(current_step, retrain_time, new_performance)
        
        # Store timing
        self.timing_metrics['retraining_times'].append(retrain_time)
        
        # Reset trust region if available
        if hasattr(self.mpc_controller, 'reset_trust_region'):
            self.mpc_controller.reset_trust_region()
        
        return True
    
    def step(self):
        """Step function for strategy updates"""
        if hasattr(self.strategy, 'step'):
            self.strategy.step()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retraining statistics"""
        stats = {
            'total_retrainings': len(self.timing_metrics['retraining_times']),
            'avg_retrain_time': np.mean(self.timing_metrics['retraining_times']) if self.timing_metrics['retraining_times'] else 0.0,
            'total_retrain_time': np.sum(self.timing_metrics['retraining_times']),
        }
        
        if hasattr(self.strategy, 'retrain_count'):
            stats['strategy_retrain_count'] = self.strategy.retrain_count
        
        if hasattr(self.strategy, 'adaptive_window_size'):
            stats['current_window_size'] = self.strategy.adaptive_window_size
        
        return stats

# Example usage and configuration
def create_retraining_config():
    """Example configuration for different retraining strategies"""
    
    configs = {
        'aggressive_adaptive': {
            'enable_retraining': True,
            'retraining_strategy': 'sliding_window',
            'window_policy': 'adaptive',
            'retrain_window_size': 800,
            'retrain_period': 30,
            'retrain_innov_threshold': 0.2,
            'retrain_quality_threshold': 1.2,
            'min_retrain_samples': 100,
            'max_retrain_samples': 1500
        },
        
        'conservative_quality': {
            'enable_retraining': True,
            'retraining_strategy': 'sliding_window',
            'window_policy': 'quality_based',
            'retrain_window_size': 1200,
            'retrain_period': 80,
            'retrain_innov_threshold': 0.4,
            'retrain_quality_threshold': 1.8,
            'min_retrain_samples': 200,
            'max_retrain_samples': 2000
        },
        
        'exponential_weighted': {
            'enable_retraining': True,
            'retraining_strategy': 'sliding_window',
            'window_policy': 'exponential',
            'retrain_window_size': 1000,
            'retrain_period': 50,
            'retrain_innov_threshold': 0.3,
            'min_retrain_samples': 150,
            'max_retrain_samples': 1800
        },
        
        'simple_time_based': {
            'enable_retraining': True,
            'retraining_strategy': 'time_based',
            'retrain_period': 100,
            'retrain_window_size': 500
        }
    }
    
    return configs