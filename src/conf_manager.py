# conf_manager.py

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

class MPCConfigManager:
    def __init__(self, config_dir: str = "mpc_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Базові параметри за замовчуванням
        self.default_config = {
            # ---- Блок даних
            'N_data': 4000,
            'control_pts': 400, 
            'seed': 42,
            
            # ---- MPC параметри
            'horizon': 10,
            'prediction_horizon': 20,
            'control_horizon': 5,
            'sampling_time': 0.1,
            
            # ---- Модель
            'model_type': 'rbf',
            'kernel_type': 'gaussian',
            'regularization': 1e-6,
            'n_centers': 100,
            
            # ---- Оптимізація
            'optimizer': 'scipy',
            'max_iterations': 100,
            'tolerance': 1e-6,
            'trust_region_radius': 1.0,
            
            # ---- Обмеження
            'input_bounds': [(-2.0, 2.0), (-1.5, 1.5)],
            'output_bounds': [(-3.0, 3.0), (-2.0, 2.0)],
            'rate_limits': [(0.5, 0.5), (0.3, 0.3)],
            
            # ---- Ваги функції вартості
            'tracking_weight': 1.0,
            'control_weight': 0.1,
            'rate_weight': 0.01,
            'terminal_weight': 10.0,
            
            # ---- Специфічні для магнітної сепарації
            'magnetic_field_range': (0.1, 2.5),
            'flow_rate_range': (5.0, 50.0),
            'separation_efficiency_target': 0.85,
            'recovery_rate_target': 0.90,
            
            # ---- Збурення та шум
            'process_noise_std': 0.01,
            'measurement_noise_std': 0.005,
            'disturbance_amplitude': 0.1,
            
            # ---- Режим симуляції
            'verbose': True,
            'save_trajectory': True,
            'plot_results': False
        }
    
    def save_config(self, config: Dict[str, Any], name: str) -> Path:
        """Збереження конфігурації"""
        config_path = self.config_dir / f"{name}.json"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"✅ Конфігурація '{name}' збережена: {config_path}")
        return config_path
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """Завантаження конфігурації"""
        config_path = self.config_dir / f"{name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Конфігурація '{name}' не знайдена: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Конфігурація '{name}' завантажена")
        return config
    
    def create_config(self, name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Створення нової конфігурації з override'ами"""
        config = self.default_config.copy()
        config.update(overrides)
        self.save_config(config, name)
        return config
    
    def list_configs(self) -> list:
        """Список всіх доступних конфігурацій"""
        configs = list(self.config_dir.glob("*.json"))
        return [c.stem for c in configs]
    
    def merge_config(self, base_config: str, overrides: Dict[str, Any], 
                    new_name: str) -> Dict[str, Any]:
        """Створення конфігурації на основі існуючої + зміни"""
        base = self.load_config(base_config)
        base.update(overrides)
        self.save_config(base, new_name)
        return base

# Ініціалізуємо менеджер
config_manager = MPCConfigManager()
print("✅ MPCConfigManager ініціалізовано")

def main():
 
    your_original_params = {
        # ---- Блок даних
        'N_data': 4000, 
        'control_pts': 400,
        'seed': 42,
        
        'plant_model_type': 'rf',
        
        'train_size': 0.75,
        'val_size': 0.2,
        'test_size': 0.05,
    
        # ---- Налаштування моделі
        'noise_level': 'low',
        
        'model_type': 'linear',          # L-MPC
        'linear_type': 'ridge',          # ols, ridge, lasso
        'poly_degree': 2,                # 1=лінійна, 2=квадратична, 3=кубічна
        'alpha': 1.0,                    # Регуляризація для ridge/lasso
        
        'find_optimal_params': True,      # Автопошук параметрів
        'use_soft_constraints': True,
        
        # ---- Налаштування EKF
        'P0': 1e-2,
        'Q_phys': 600,
        'Q_dist': 1,
        'R': 1.0,
        'q_adaptive_enabled': False,
        'q_alpha': 0.90,
        'q_nis_threshold': 3.0,
    
        # Адаптивний Trust Region
        'adaptive_trust_region': True,
        'initial_trust_radius': 3.0,
        'min_trust_radius': 0.5,
        'max_trust_radius': 2.0,
        'trust_decay_factor': 0.9,
        'rho_trust': 0.5,
        
        # Контроль лінеаризації
        'linearization_check_enabled': True,
        'max_linearization_distance': 0.8,
        'retrain_linearization_threshold': 1.0,
    
        # ---- Налаштування аномалій
        'anomaly_params': {
            'window': 25,
            'spike_z': 4.0,
            'drop_rel': 0.30,
            'freeze_len': 5,
            'enabled': True
        },
    
        # ---- Нелінійні параметри
        'nonlinear_config': {
            'concentrate_fe_percent': ('pow', 2),
            'concentrate_mass_flow': ('pow', 1.5)
        },
        'enable_nonlinear': True, 
    
        # ---- Параметри затримки, часові параметри
        'time_step_s': 1800,
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
        
        # ---- Обмеження моделі
        'delta_u_max': 0.6,
        'λ_obj': 0.2,
        
        # ---- MPC горизонти
        'Nc': 6,
        'Np': 8,
        'lag': 2,
        
        # ---- Цільові параметри/ваги
        'w_fe': 1.0,
        'w_mass': 1.0,
        'ref_fe': 54.5,
        'ref_mass': 57.0,
        'y_max_fe': 55.0,
        'y_max_mass': 60.0,
        
        # ---- Блок перенавчання
        'enable_retraining': True,
        'retrain_period': 50,
        'retrain_window_size': 1000,
        'retrain_innov_threshold': 0.25,
        
        'run_analysis': False
    }
    
    # Зберігаємо як новий профіль "oleksandr_original"
    saved_config = config_manager.create_config('oleksandr_original', your_original_params)
    
    print("✅ Твої параметри збережено в профіль: 'oleksandr_original'")
    print(f"📊 Збережено {len(your_original_params)} параметрів")
if __name__ == '__main__':
    main()