# retraining_fixes.py - Solutions for catastrophic forgetting in retraining

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import time

class RobustRetrainingStrategy:
    """
    Enhanced retraining strategy that prevents catastrophic forgetting
    and handles regime changes smoothly
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        retrain_period: int = 50,
        innovation_threshold: float = 0.3,
        # NEW: Catastrophic forgetting prevention
        stability_buffer_size: int = 200,        # Always keep stable data
        regime_change_detection: bool = True,     # Detect regime changes
        conservative_retrain_factor: float = 0.7, # More conservative retraining
        gradient_smoothing: bool = True,          # Smooth parameter updates
        ensemble_voting: bool = True,             # Use ensemble for decisions
        
        # NEW: Multi-threshold system
        soft_innovation_threshold: float = 0.2,   # Warning level
        hard_innovation_threshold: float = 0.5,   # Emergency level
        regime_change_threshold: float = 2.0,     # Significant change
        
        # NEW: Adaptive window management
        min_stable_ratio: float = 0.3,           # Min % of stable data
        max_recent_ratio: float = 0.7,           # Max % of recent data
        stability_lookback: int = 100,           # Steps to assess stability
    ):
        self.window_size = window_size
        self.retrain_period = retrain_period
        self.innovation_threshold = innovation_threshold
        
        # Catastrophic forgetting prevention
        self.stability_buffer_size = stability_buffer_size
        self.regime_change_detection = regime_change_detection
        self.conservative_retrain_factor = conservative_retrain_factor
        self.gradient_smoothing = gradient_smoothing
        self.ensemble_voting = ensemble_voting
        
        # Multi-threshold system
        self.soft_innovation_threshold = soft_innovation_threshold
        self.hard_innovation_threshold = hard_innovation_threshold
        self.regime_change_threshold = regime_change_threshold
        
        # Adaptive window management
        self.min_stable_ratio = min_stable_ratio
        self.max_recent_ratio = max_recent_ratio
        self.stability_lookback = stability_lookback
        
        # Data storage with categorization
        self.data_buffer = deque(maxlen=window_size * 2)  # Larger buffer
        self.stability_buffer = deque(maxlen=stability_buffer_size)  # Always keep stable data
        self.recent_buffer = deque(maxlen=retrain_period * 3)  # Recent data
        
        # Monitoring and detection
        self.innovation_monitor = deque(maxlen=retrain_period)
        self.stability_monitor = deque(maxlen=stability_lookback)
        self.regime_indicators = deque(maxlen=50)
        
        # State tracking
        self.last_retrain_step = 0
        self.cooldown_timer = 0
        self.retrain_count = 0
        self.consecutive_failures = 0
        self.regime_change_detected = False
        self.last_stable_performance = None
        
        # Model ensemble for robust decisions
        self.performance_history = deque(maxlen=100)
        self.prediction_ensemble = []
        
        print(f"🛡️ Robust retraining strategy initialized:")
        print(f"   • Stability buffer: {stability_buffer_size}")
        print(f"   • Regime change detection: {regime_change_detection}")
        print(f"   • Conservative factor: {conservative_retrain_factor}")
        print(f"   • Multi-threshold: soft={soft_innovation_threshold}, hard={hard_innovation_threshold}")
    
    def should_retrain(self, current_step: int, metrics: Dict[str, float]) -> str:
        """
        Enhanced retraining decision with multiple levels
        
        Returns:
            'none': No retraining needed
            'soft': Soft retraining (partial update)
            'hard': Full retraining needed
            'emergency': Emergency retraining (regime change)
        """
        # Basic conditions
        if (current_step <= 0 or 
            current_step - self.last_retrain_step < self.retrain_period or
            len(self.data_buffer) < 50 or
            self.cooldown_timer > 0):
            return 'none'
        
        # Detect regime change
        regime_change = self._detect_regime_change(metrics)
        
        # Calculate current innovation level
        current_innovation = metrics.get('innovation_norm', 0.0)
        avg_recent_innovation = np.mean(list(self.innovation_monitor)) if self.innovation_monitor else 0.0
        
        # Multi-level decision logic
        if regime_change:
            print(f"🚨 REGIME CHANGE DETECTED at step {current_step}")
            return 'emergency'
        elif current_innovation > self.hard_innovation_threshold:
            print(f"⚠️ HARD THRESHOLD EXCEEDED: {current_innovation:.3f} > {self.hard_innovation_threshold}")
            return 'hard'
        elif avg_recent_innovation > self.innovation_threshold:
            print(f"📊 STANDARD THRESHOLD EXCEEDED: {avg_recent_innovation:.3f} > {self.innovation_threshold}")
            return 'hard'
        elif current_innovation > self.soft_innovation_threshold:
            stability_score = self._calculate_stability_score()
            if stability_score < 0.7:  # Low stability
                print(f"🔄 SOFT RETRAINING: innovation={current_innovation:.3f}, stability={stability_score:.3f}")
                return 'soft'
        
        return 'none'
    
    def get_training_data(self, retrain_type: str = 'hard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data based on retraining type and prevent catastrophic forgetting
        """
        if retrain_type == 'emergency':
            return self._get_emergency_training_data()
        elif retrain_type == 'hard':
            return self._get_robust_training_data()
        elif retrain_type == 'soft':
            return self._get_soft_training_data()
        else:
            return self._get_robust_training_data()
    
    def _get_emergency_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Emergency retraining data for regime changes"""
        print("🚨 Preparing emergency retraining data...")
        
        # Combine stability buffer (40%) + recent data (60%)
        stable_data = list(self.stability_buffer)
        recent_data = list(self.recent_buffer)
        
        # Calculate optimal mix
        total_samples = min(self.window_size, len(stable_data) + len(recent_data))
        stable_count = max(50, int(total_samples * 0.4))  # At least 50 stable samples
        recent_count = total_samples - stable_count
        
        # Select data
        selected_stable = stable_data[-stable_count:] if len(stable_data) >= stable_count else stable_data
        selected_recent = recent_data[-recent_count:] if len(recent_data) >= recent_count else recent_data
        
        # Combine and balance
        combined_data = selected_stable + selected_recent
        
        if not combined_data:
            print("⚠️ No emergency data available, using all available data")
            combined_data = list(self.data_buffer)[-200:]
        
        X = np.array([point['x'] for point in combined_data])
        Y = np.array([point['y'] for point in combined_data])
        
        print(f"   • Emergency data: {len(X)} samples (stable: {len(selected_stable)}, recent: {len(selected_recent)})")
        return X, Y
    
    def _get_robust_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Robust training data preventing catastrophic forgetting"""
        print("🛡️ Preparing robust retraining data...")
        
        all_data = list(self.data_buffer)
        if not all_data:
            return np.array([]), np.array([])
        
        # Categorize data by quality and recency
        stable_data = [d for d in all_data if d.get('quality', 0) > 0.7]
        recent_data = list(self.recent_buffer)
        
        # Calculate adaptive mix ratios
        stability_score = self._calculate_stability_score()
        
        if stability_score > 0.8:  # High stability - more recent data
            stable_ratio = 0.3
            recent_ratio = 0.7
        elif stability_score < 0.5:  # Low stability - more stable data
            stable_ratio = 0.6
            recent_ratio = 0.4
        else:  # Medium stability - balanced
            stable_ratio = 0.5
            recent_ratio = 0.5
        
        # Select data
        target_samples = min(self.window_size, len(all_data))
        stable_count = int(target_samples * stable_ratio)
        recent_count = int(target_samples * recent_ratio)
        
        # Quality-based selection for stable data
        if stable_data:
            stable_data_sorted = sorted(stable_data, key=lambda x: x.get('quality', 0), reverse=True)
            selected_stable = stable_data_sorted[:stable_count]
        else:
            selected_stable = all_data[:stable_count]
        
        # Recent data selection
        selected_recent = recent_data[-recent_count:] if len(recent_data) >= recent_count else recent_data
        
        # Fill remaining with quality data if needed
        used_indices = set()
        combined_data = selected_stable + selected_recent
        
        if len(combined_data) < target_samples:
            remaining = target_samples - len(combined_data)
            quality_sorted = sorted(all_data, key=lambda x: x.get('quality', 0), reverse=True)
            for item in quality_sorted:
                if len(combined_data) >= target_samples:
                    break
                if item not in combined_data:
                    combined_data.append(item)
        
        X = np.array([point['x'] for point in combined_data])
        Y = np.array([point['y'] for point in combined_data])
        
        print(f"   • Robust data: {len(X)} samples (stable: {len(selected_stable)}, recent: {len(selected_recent)})")
        print(f"   • Stability score: {stability_score:.3f}, ratios: stable={stable_ratio:.1f}, recent={recent_ratio:.1f}")
        
        return X, Y
    
    def _get_soft_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Soft retraining data for incremental updates"""
        print("🔄 Preparing soft retraining data...")
        
        # Use mostly stable data with small amount of recent data
        stable_data = list(self.stability_buffer)
        recent_data = list(self.recent_buffer)[-50:]  # Only last 50 recent samples
        
        # 80% stable, 20% recent for soft updates
        target_samples = min(500, len(stable_data) + len(recent_data))
        stable_count = int(target_samples * 0.8)
        recent_count = target_samples - stable_count
        
        selected_stable = stable_data[-stable_count:] if len(stable_data) >= stable_count else stable_data
        selected_recent = recent_data[-recent_count:] if len(recent_data) >= recent_count else recent_data
        
        combined_data = selected_stable + selected_recent
        
        X = np.array([point['x'] for point in combined_data])
        Y = np.array([point['y'] for point in combined_data])
        
        print(f"   • Soft data: {len(X)} samples (stable: {len(selected_stable)}, recent: {len(selected_recent)})")
        return X, Y
    
    def add_data_point(self, x: np.ndarray, y: np.ndarray, metrics: Dict[str, float]):
        """Enhanced data point addition with categorization"""
        # Calculate enhanced quality score
        quality_score = self._calculate_enhanced_quality_score(x, y, metrics)
        
        # Create data point with enhanced metadata
        data_point = {
            'x': x.copy(),
            'y': y.copy(),
            'quality': quality_score,
            'timestamp': time.time(),
            'innovation_norm': metrics.get('innovation_norm', 0.0),
            'prediction_error': metrics.get('prediction_error', 0.0),
            'stability_indicator': self._calculate_local_stability(metrics),
            'regime_indicator': self._calculate_regime_indicator(metrics)
        }
        
        # Add to appropriate buffers
        self.data_buffer.append(data_point)
        self.recent_buffer.append(data_point)
        
        # Add to stability buffer if high quality and stable
        if quality_score > 0.75 and data_point['stability_indicator'] > 0.8:
            self.stability_buffer.append(data_point)
        
        # Update monitoring
        self.innovation_monitor.append(metrics.get('innovation_norm', 0.0))
        self.stability_monitor.append(data_point['stability_indicator'])
        self.regime_indicators.append(data_point['regime_indicator'])
        
        # Update performance tracking
        if 'prediction_error' in metrics:
            self.performance_history.append(metrics['prediction_error'])
    
    def _detect_regime_change(self, metrics: Dict[str, float]) -> bool:
        """Enhanced regime change detection"""
        if not self.regime_change_detection or len(self.regime_indicators) < 20:
            return False
        
        # Statistical change detection
        recent_indicators = list(self.regime_indicators)[-10:]
        older_indicators = list(self.regime_indicators)[-20:-10]
        
        if len(older_indicators) < 5:
            return False
        
        recent_mean = np.mean(recent_indicators)
        older_mean = np.mean(older_indicators)
        
        # Detect significant change
        relative_change = abs(recent_mean - older_mean) / (older_mean + 1e-6)
        
        # Multiple indicators for regime change
        innovation_spike = metrics.get('innovation_norm', 0.0) > self.regime_change_threshold
        performance_drop = self._detect_performance_drop()
        statistical_change = relative_change > 0.5
        
        regime_change = innovation_spike and (performance_drop or statistical_change)
        
        if regime_change:
            self.regime_change_detected = True
            print(f"🚨 Regime change indicators:")
            print(f"   • Innovation spike: {innovation_spike} ({metrics.get('innovation_norm', 0.0):.3f})")
            print(f"   • Performance drop: {performance_drop}")
            print(f"   • Statistical change: {statistical_change} ({relative_change:.3f})")
        
        return regime_change
    
    def _detect_performance_drop(self) -> bool:
        """Detect significant performance degradation"""
        if len(self.performance_history) < 20:
            return False
        
        recent_perf = np.mean(list(self.performance_history)[-10:])
        baseline_perf = np.mean(list(self.performance_history)[-20:-10])
        
        # Significant performance drop
        return recent_perf > baseline_perf * 1.5
    
    def _calculate_stability_score(self) -> float:
        """Calculate overall system stability score"""
        if len(self.stability_monitor) < 10:
            return 0.5  # Default moderate stability
        
        recent_stability = np.mean(list(self.stability_monitor)[-10:])
        stability_variance = np.var(list(self.stability_monitor)[-20:]) if len(self.stability_monitor) >= 20 else 0.1
        
        # Higher score = more stable (lower variance, higher mean stability)
        stability_score = recent_stability * (1.0 - min(stability_variance, 0.5))
        
        return np.clip(stability_score, 0.0, 1.0)
    
    def _calculate_enhanced_quality_score(self, x: np.ndarray, y: np.ndarray, metrics: Dict[str, float]) -> float:
        """Enhanced quality score calculation"""
        # Base quality components
        innovation_score = 1.0 / (1.0 + metrics.get('innovation_norm', 1.0))
        error_score = 1.0 / (1.0 + metrics.get('prediction_error', 1.0))
        
        # Stability component
        stability_score = self._calculate_local_stability(metrics)
        
        # Diversity component
        diversity_score = self._calculate_diversity_score(x)
        
        # Regime consistency component (NEW)
        regime_consistency = 1.0 - abs(self._calculate_regime_indicator(metrics) - 
                                     np.mean(list(self.regime_indicators)[-5:]) if len(self.regime_indicators) >= 5 else 0.5)
        
        # Weighted combination with emphasis on stability during regime changes
        if self.regime_change_detected:
            # During regime changes, prioritize stability and consistency
            quality = (0.2 * innovation_score + 0.2 * error_score + 
                      0.4 * stability_score + 0.1 * diversity_score + 0.1 * regime_consistency)
        else:
            # Normal operation - balanced approach
            quality = (0.3 * innovation_score + 0.3 * error_score + 
                      0.2 * stability_score + 0.1 * diversity_score + 0.1 * regime_consistency)
        
        return np.clip(quality, 0.0, 1.0)
    
    def _calculate_local_stability(self, metrics: Dict[str, float]) -> float:
        """Calculate local stability indicator"""
        # Combine multiple stability indicators
        innovation_stability = 1.0 / (1.0 + metrics.get('innovation_norm', 1.0))
        
        # EKF uncertainty as stability indicator
        ekf_stability = 1.0 / (1.0 + metrics.get('ekf_uncertainty', 1.0))
        
        # Measurement variability
        meas_stability = 1.0 / (1.0 + metrics.get('measurement_variability', 1.0))
        
        # Combined stability
        stability = (innovation_stability + ekf_stability + meas_stability) / 3.0
        
        return np.clip(stability, 0.0, 1.0)
    
    def _calculate_regime_indicator(self, metrics: Dict[str, float]) -> float:
        """Calculate regime indicator for change detection"""
        # Combine innovation and prediction error for regime detection
        innovation_component = metrics.get('innovation_norm', 0.0)
        error_component = metrics.get('prediction_error', 0.0)
        
        # Normalize to 0-1 range
        regime_indicator = min(1.0, (innovation_component + error_component) / 2.0)
        
        return regime_indicator
    
    def _calculate_diversity_score(self, x: np.ndarray) -> float:
        """Calculate diversity score with recent data consideration"""
        if len(self.data_buffer) < 10:
            return 1.0
        
        # Sample recent data points for comparison
        recent_points = [point['x'] for point in list(self.data_buffer)[-10:]]
        
        if len(recent_points) == 0:
            return 1.0
        
        # Calculate minimum distance to existing points
        distances = [np.linalg.norm(x - point) for point in recent_points]
        min_distance = min(distances) if distances else 0.0
        
        # Convert to score (higher distance = higher diversity)
        diversity_score = min(1.0, min_distance / 2.0)
        
        return diversity_score
    
    def update_after_retraining(self, current_step: int, retrain_time: float, new_performance: float, retrain_type: str):
        """Enhanced post-retraining update"""
        self.last_retrain_step = current_step
        self.retrain_count += 1
        
        # Adaptive cooldown based on retraining type and success
        if retrain_type == 'emergency':
            self.cooldown_timer = max(20, self.retrain_period // 2)  # Shorter cooldown for emergencies
        elif retrain_type == 'soft':
            self.cooldown_timer = self.retrain_period // 4  # Very short cooldown for soft updates
        else:
            self.cooldown_timer = self.retrain_period
        
        # Track success/failure
        if hasattr(self, 'last_stable_performance') and self.last_stable_performance is not None:
            if new_performance > self.last_stable_performance * 1.2:  # Performance got worse
                self.consecutive_failures += 1
                print(f"⚠️ Retraining degraded performance: {new_performance:.4f} vs {self.last_stable_performance:.4f}")
            else:
                self.consecutive_failures = 0
                self.last_stable_performance = new_performance
        else:
            self.last_stable_performance = new_performance
        
        # Reset regime change flag after emergency retraining
        if retrain_type == 'emergency':
            self.regime_change_detected = False
            print("🔄 Regime change flag reset after emergency retraining")
        
        # Clear monitoring buffers after successful retraining
        if self.consecutive_failures == 0:
            self.innovation_monitor.clear()
        
        print(f"   ✅ Retraining #{self.retrain_count} ({retrain_type}) completed in {retrain_time:.3f}s")
        print(f"   📊 Performance: {new_performance:.4f}, Consecutive failures: {self.consecutive_failures}")
        print(f"   ⏰ Cooldown: {self.cooldown_timer} steps")
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retraining statistics"""
        stats = {
            'total_retrainings': self.retrain_count,
            'consecutive_failures': self.consecutive_failures,
            'regime_changes_detected': self.regime_change_detected,
            'stability_score': self._calculate_stability_score(),
            'data_quality_avg': np.mean([d.get('quality', 0) for d in self.data_buffer]) if self.data_buffer else 0.0,
            'stability_buffer_size': len(self.stability_buffer),
            'recent_buffer_size': len(self.recent_buffer),
        }
        
        if self.performance_history:
            stats['performance_trend'] = self._calculate_performance_trend()
        
        return stats
    
    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend over recent history"""
        if len(self.performance_history) < 20:
            return 0.0
        
        recent = np.mean(list(self.performance_history)[-10:])
        older = np.mean(list(self.performance_history)[-20:-10])
        
        # Negative trend = performance improving (lower error)
        trend = (recent - older) / (older + 1e-6)
        return trend
