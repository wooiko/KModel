# config_manager.py - Модуль для роботи з конфігураціями MPC

import json
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import pandas as pd

# =============================================================================
# === ВНУТРІШНІ ДОПОМІЖНІ ФУНКЦІЇ ===
# =============================================================================

def _get_config_dir() -> Path:
    """Повертає шлях до папки з конфігураціями."""
    return Path("mpc_configs")

def _get_results_dir() -> Path:
    """Повертає шлях до папки з результатами."""
    return Path("mpc_results")

def _ensure_config_dir_exists() -> None:
    """Створює папку конфігурацій якщо її немає."""
    config_dir = _get_config_dir()
    config_dir.mkdir(exist_ok=True)

def _ensure_results_dir_exists() -> None:
    """Створює папку результатів якщо її немає."""
    results_dir = _get_results_dir()
    results_dir.mkdir(exist_ok=True)

def _validate_config_file(config_file: Path) -> bool:
    """Перевіряє чи є файл валідним JSON."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, FileNotFoundError):
        return False

def _filter_for_simulate_mpc(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Фільтрує конфігурацію, залишаючи тільки валідні параметри для simulate_mpc з підтримкою L-MPC.
    """
    # Імпортуємо тут щоб уникнути циклічних імпортів
    from sim import simulate_mpc
    
    # Отримуємо валідні параметри через інспекцію
    sig = inspect.signature(simulate_mpc)
    valid_params = set(sig.parameters.keys())
    valid_params.discard('reference_df')  # Передається окремо
    
    # Фільтруємо
    filtered_config = {}
    invalid_params = []
    
    for key, value in config.items():
        if key in valid_params:
            filtered_config[key] = value
        elif key in ['name', 'description']:
            # Пропускаємо службові поля без повідомлення
            continue
        elif key.startswith('_') and key.endswith('_'):
            # Пропускаємо роздільники секцій (_SIMULATION_, _MODEL_, etc.)
            continue
        elif isinstance(value, str) and value.startswith('='):
            # Пропускаємо значення-роздільники ("======")
            continue
        else:
            invalid_params.append(key)
    
    # 🆕 СПЕЦІАЛЬНА ОБРОБКА ЛІНІЙНИХ МОДЕЛЕЙ
    if filtered_config.get('model_type', '').lower() == 'linear':
        # Додаємо L-MPC специфічні параметри якщо вони відсутні
        linear_defaults = {
            'linear_type': 'ols',
            'poly_degree': 1,
            'include_bias': True,
            'alpha': 1.0
        }
        
        for param, default_value in linear_defaults.items():
            if param not in filtered_config:
                filtered_config[param] = default_value
                print(f"ℹ️ Додано параметр L-MPC за замовчуванням: {param}={default_value}")
        
        # Валідація L-MPC параметрів
        if filtered_config.get('linear_type') not in ['ols', 'ridge', 'lasso']:
            print(f"⚠️ Некоректний linear_type '{filtered_config.get('linear_type')}', встановлено 'ols'")
            filtered_config['linear_type'] = 'ols'
            
        if not (1 <= filtered_config.get('poly_degree', 1) <= 3):
            print(f"⚠️ Некоректний poly_degree {filtered_config.get('poly_degree')}, встановлено 1")
            filtered_config['poly_degree'] = 1
    
    # 🆕 ВАЛІДАЦІЯ ЯДЕРНИХ МОДЕЛЕЙ
    elif filtered_config.get('model_type', '').lower() in ['krr', 'svr', 'gpr']:
        # Забезпечуємо наявність kernel для ядерних моделей
        if 'kernel' not in filtered_config:
            filtered_config['kernel'] = 'rbf'
            print(f"ℹ️ Додано kernel за замовчуванням для K-MPC: rbf")
    
    if invalid_params:
        print(f"ℹ️ Пропущено невалідні параметри: {', '.join(invalid_params)}")
    
    return filtered_config


# =============================================================================
# === ПУБЛІЧНІ ФУНКЦІЇ ===
# =============================================================================

def load_config(config_name: str) -> Dict[str, Any]:
    """
    Завантажує конфігурацію MPC з файлу.
    
    Args:
        config_name: Назва конфігурації (без розширення .json)
        
    Returns:
        Словник з параметрами конфігурації
        
    Raises:
        FileNotFoundError: Якщо конфігурація не знайдена
        ValueError: Якщо файл має неправильний формат JSON
    """
    config_dir = _get_config_dir()
    config_file = config_dir / f"{config_name}.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Конфігурація '{config_name}' не знайдена в {config_dir}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Помилка у форматі JSON конфігурації '{config_name}': {e}")

def save_config(config: Dict[str, Any], config_name: str) -> None:
    """
    Зберігає конфігурацію MPC у файл.
    
    Args:
        config: Словник з параметрами конфігурації
        config_name: Назва конфігурації (без розширення .json)
    """
    _ensure_config_dir_exists()
    
    config_dir = _get_config_dir()
    config_file = config_dir / f"{config_name}.json"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def list_configs() -> List[str]:
    """
    Повертає список доступних конфігурацій.
    
    Returns:
        Список назв конфігурацій (без розширення .json)
    """
    config_dir = _get_config_dir()
    if not config_dir.exists():
        return []
    
    configs = []
    for config_file in config_dir.glob("*.json"):
        if _validate_config_file(config_file):
            configs.append(config_file.stem)
    
    return sorted(configs)

def create_default_configs() -> None:
    """
    Створює стандартні конфігурації MPC з підтримкою L-MPC.
    """
    _ensure_config_dir_exists()
    
    # Консервативна конфігурація (K-MPC)
    conservative_config = {
        "name": "conservative",
        "description": "Консервативна конфігурація для стабільної роботи (K-MPC)",
        
        # Основні параметри
        "N_data": 2000,
        "control_pts": 200,
        "seed": 42,
        
        # Модель
        "model_type": "krr",
        "kernel": "rbf",
        "find_optimal_params": True,
        
        # MPC параметри
        "Np": 6,
        "Nc": 4,
        "lag": 2,
        "λ_obj": 0.2,
        
        # Ваги та уставки
        "w_fe": 5.0,
        "w_mass": 1.0,
        "ref_fe": 53.5,
        "ref_mass": 57.0,
        "tolerance_fe_percent": 1.5,
        "tolerance_mass_percent": 2.0,
        
        # Trust region
        "adaptive_trust_region": True,
        "initial_trust_radius": 1.0,
        "min_trust_radius": 0.5,
        "max_trust_radius": 3.0,
        "trust_decay_factor": 0.9,
        "rho_trust": 0.3,
        
        # EKF параметри
        "P0": 0.01,
        "Q_phys": 800,
        "Q_dist": 1,
        "R": 0.5,
        "q_adaptive_enabled": True,
        "q_alpha": 0.95,
        "q_nis_threshold": 2.0,
        
        # Перенавчання
        "enable_retraining": True,
        "retrain_period": 50,
        "retrain_innov_threshold": 0.3,
        "retrain_window_size": 1000,
        
        # Обмеження
        "use_soft_constraints": True,
        "delta_u_max": 0.8,
        "u_min": 20.0,
        "u_max": 40.0,
        
        # Процес
        "plant_model_type": "rf",
        "noise_level": "low",
        "enable_nonlinear": False,
        
        # Аномалії
        "anomaly_params": {
            "window": 25,
            "spike_z": 4.0,
            "drop_rel": 0.30,
            "freeze_len": 5,
            "enabled": True
        },
        
        "run_analysis": True
    }
    
    # Агресивна конфігурація (K-MPC)
    aggressive_config = {
        "name": "aggressive",
        "description": "Агресивна конфігурація для швидкого відгуку (K-MPC)",
        
        # Основні параметри
        "N_data": 3000,
        "control_pts": 300,
        "seed": 42,
        
        # Модель
        "model_type": "svr",
        "kernel": "rbf", 
        "find_optimal_params": True,
        
        # MPC параметри
        "Np": 8,
        "Nc": 6,
        "lag": 2,
        "λ_obj": 0.05,
        
        # Ваги та уставки
        "w_fe": 10.0,
        "w_mass": 2.0,
        "ref_fe": 54.0,
        "ref_mass": 58.0,
        "tolerance_fe_percent": 2.5,
        "tolerance_mass_percent": 3.0,
        
        # Trust region
        "adaptive_trust_region": True,
        "initial_trust_radius": 2.0,
        "min_trust_radius": 0.8,
        "max_trust_radius": 5.0,
        "trust_decay_factor": 0.8,
        "rho_trust": 0.1,
        
        # EKF параметри
        "P0": 0.01,
        "Q_phys": 1200,
        "Q_dist": 1,
        "R": 0.1,
        "q_adaptive_enabled": True,
        "q_alpha": 0.90,
        "q_nis_threshold": 3.0,
        
        # Перенавчання
        "enable_retraining": True,
        "retrain_period": 30,
        "retrain_innov_threshold": 0.2,
        "retrain_window_size": 800,
        
        # Обмеження
        "use_soft_constraints": True,
        "delta_u_max": 1.2,
        "u_min": 18.0,
        "u_max": 42.0,
        
        # Процес
        "plant_model_type": "rf",
        "noise_level": "medium",
        "enable_nonlinear": True,
        
        # Нелінійність
        "nonlinear_config": {
            "concentrate_fe_percent": ("pow", 2),
            "concentrate_mass_flow": ("pow", 1.5)
        },
        
        # Аномалії
        "anomaly_params": {
            "window": 20,
            "spike_z": 3.5,
            "drop_rel": 0.25,
            "freeze_len": 3,
            "enabled": True
        },
        
        "run_analysis": True
    }
    
    # 🆕 ЛІНІЙНА КОНФІГУРАЦІЯ (L-MPC) - OLS
    linear_ols_config = {
        "name": "linear_ols",
        "description": "Лінійна модель з OLS регресією (L-MPC)",
        
        # Основні параметри
        "N_data": 2000,
        "control_pts": 200,
        "seed": 42,
        
        # 🎯 ЛІНІЙНА МОДЕЛЬ
        "model_type": "linear",
        "linear_type": "ols",
        "poly_degree": 1,
        "include_bias": True,
        "find_optimal_params": False,
        
        # MPC параметри
        "Np": 6,
        "Nc": 4,
        "lag": 2,
        "λ_obj": 0.15,
        
        # Ваги та уставки
        "w_fe": 6.0,
        "w_mass": 1.0,
        "ref_fe": 53.5,
        "ref_mass": 57.0,
        "tolerance_fe_percent": 2.5,
        "tolerance_mass_percent": 3.0,
        
        # Trust region (менш агресивні налаштування для лінійної моделі)
        "adaptive_trust_region": False,
        "initial_trust_radius": 1.5,
        "min_trust_radius": 0.8,
        "max_trust_radius": 3.0,
        "trust_decay_factor": 0.9,
        "rho_trust": 0.2,
        
        # EKF параметри
        "P0": 0.01,
        "Q_phys": 600,
        "Q_dist": 1,
        "R": 0.3,
        "q_adaptive_enabled": False,
        "q_alpha": 0.98,
        "q_nis_threshold": 2.5,
        
        # Перенавчання (частіше для лінійної моделі)
        "enable_retraining": True,
        "retrain_period": 40,
        "retrain_innov_threshold": 0.25,
        "retrain_window_size": 800,
        
        # Обмеження
        "use_soft_constraints": True,
        "delta_u_max": 1.0,
        "u_min": 20.0,
        "u_max": 40.0,
        
        # Процес
        "plant_model_type": "rf",
        "noise_level": "low",
        "enable_nonlinear": False,
        
        # Аномалії
        "anomaly_params": {
            "window": 25,
            "spike_z": 4.0,
            "drop_rel": 0.30,
            "freeze_len": 5,
            "enabled": True
        },
        
        "run_analysis": False
    }
    
    # 🆕 ПОЛІНОМІАЛЬНА КОНФІГУРАЦІЯ (L-MPC) - Ridge
    linear_poly_config = {
        "name": "linear_poly",
        "description": "Поліноміальна модель ступеня 2 з Ridge регуляризацією (L-MPC)",
        
        # Основні параметри
        "N_data": 2500,
        "control_pts": 250,
        "seed": 42,
        
        # 🎯 ПОЛІНОМІАЛЬНА МОДЕЛЬ
        "model_type": "linear",
        "linear_type": "ridge",
        "poly_degree": 2,
        "include_bias": True,
        "find_optimal_params": True,
        "alpha": 1.0,  # Початкове значення для Ridge
        
        # MPC параметри
        "Np": 5,
        "Nc": 3,
        "lag": 2,
        "λ_obj": 0.1,
        
        # Ваги та уставки
        "w_fe": 8.0,
        "w_mass": 1.5,
        "ref_fe": 53.8,
        "ref_mass": 57.5,
        "tolerance_fe_percent": 2.0,
        "tolerance_mass_percent": 2.5,
        
        # Trust region
        "adaptive_trust_region": True,
        "initial_trust_radius": 1.2,
        "min_trust_radius": 0.6,
        "max_trust_radius": 4.0,
        "trust_decay_factor": 0.85,
        "rho_trust": 0.15,
        
        # EKF параметри
        "P0": 0.01,
        "Q_phys": 900,
        "Q_dist": 1,
        "R": 0.2,
        "q_adaptive_enabled": True,
        "q_alpha": 0.96,
        "q_nis_threshold": 2.2,
        
        # Перенавчання
        "enable_retraining": True,
        "retrain_period": 35,
        "retrain_innov_threshold": 0.22,
        "retrain_window_size": 900,
        
        # Обмеження
        "use_soft_constraints": True,
        "delta_u_max": 1.1,
        "u_min": 19.0,
        "u_max": 41.0,
        
        # Процес
        "plant_model_type": "rf",
        "noise_level": "medium",
        "enable_nonlinear": False,
        
        # Аномалії
        "anomaly_params": {
            "window": 22,
            "spike_z": 3.8,
            "drop_rel": 0.28,
            "freeze_len": 4,
            "enabled": True
        },
        
        "run_analysis": True
    }
    
    # Швидка конфігурація для тестування
    fast_test_config = {
        "name": "fast_test",
        "description": "Швидка конфігурація для тестування та відладки",
        
        # Основні параметри
        "N_data": 1000,
        "control_pts": 100,
        "seed": 42,
        
        # Модель
        "model_type": "linear",
        "linear_type": "ols",
        "poly_degree": 1,
        "find_optimal_params": False,
        
        # MPC параметри
        "Np": 4,
        "Nc": 3,
        "lag": 1,
        "λ_obj": 0.1,
        
        # Ваги та уставки
        "w_fe": 7.0,
        "w_mass": 1.0,
        "ref_fe": 53.5,
        "ref_mass": 57.0,
        "tolerance_fe_percent": 5.0,
        "tolerance_mass_percent": 5.0,
        
        # Trust region
        "adaptive_trust_region": False,
        "rho_trust": 0.2,
        
        # EKF параметри
        "P0": 0.01,
        "Q_phys": 600,
        "Q_dist": 1,
        "R": 1.0,
        "q_adaptive_enabled": False,
        
        # Перенавчання
        "enable_retraining": False,
        
        # Обмеження
        "use_soft_constraints": False,
        "delta_u_max": 1.0,
        
        # Процес
        "plant_model_type": "rf",
        "noise_level": "none",
        "enable_nonlinear": False,
        
        # Аномалії
        "anomaly_params": {
            "enabled": False
        },
        
        "run_analysis": False
    }
    
    # Зберігаємо конфігурації
    configs = [conservative_config, aggressive_config, linear_ols_config, linear_poly_config, fast_test_config]
    
    for config in configs:
        config_name = config["name"]
        save_config(config, config_name)
    
    print(f"✅ Створено {len(configs)} стандартних конфігурацій (включаючи L-MPC)")

def create_default_configs_ext() -> None:
    """
    Створює стандартні конфігурації MPC з підтримкою L-MPC.
    """
    _ensure_config_dir_exists()
    
    # KRR-MPC
    krr_mpc = {
        "name": "krr-mpc",
        "description": "KRR-MPC",
        
        # Основні параметри
        "N_data": 5000,
        "control_pts": 500,
        "seed": 42,
        
        # Модель
        "model_type": "krr",
        "kernel": "rbf",
        "find_optimal_params": True,
        
        # MPC параметри
        "Np": 6,
        "Nc": 4,
        "lag": 2,
        "λ_obj": 0.1,
        
        # Ваги та уставки
        "w_fe": 5.0,
        "w_mass": 2.0,
        "ref_fe": 54.0,
        "ref_mass": 58.0,
        "tolerance_fe_percent": 1.5,
        "tolerance_mass_percent": 2.0,
        
        # Trust region
        "adaptive_trust_region": True,
        "initial_trust_radius": 2.0,
        "min_trust_radius": 0.5,
        "max_trust_radius": 2.0,
        "trust_decay_factor": 0.9,
        "rho_trust": 0.3,
        
        # EKF параметри
        "P0": 0.01,
        "Q_phys": 600,
        "Q_dist": 1,
        "R": 1.0,
        "q_adaptive_enabled": True,
        "q_alpha": 0.90,
        "q_nis_threshold": 3.0,
        
        # Перенавчання
        "enable_retraining": True,
        "retrain_period": 50,
        "retrain_innov_threshold": 0.3,
        "retrain_window_size": 1000,
        
        # Обмеження
        "use_soft_constraints": True,
        "delta_u_max": 0.8,
        "u_min": 20.0,
        "u_max": 40.0,
        
        # Процес
        "plant_model_type": "rf",
        "noise_level": "low",

        # Нелінійність
        "enable_nonlinear": True,
        "nonlinear_config": {
            "concentrate_fe_percent": ["pow", 2],
            "concentrate_mass_flow": ["pow", 1.5]
        },
        
        # Аномалії
        "anomaly_params": {
            "window": 25,
            "spike_z": 4.0,
            "drop_rel": 0.30,
            "freeze_len": 5,
            "enabled": True
        },
        
        "run_analysis": False
    }
    
    # SVR-MPC
    svr_mpc = {
        "name": "svr-mpc",
        "description": "svr-MPC",
        
        # Основні параметри
        "N_data": 5000,
        "control_pts": 500,
        "seed": 42,
        
        # Модель
        "model_type": "svr",
        "kernel": "rbf",
        "find_optimal_params": True,
        
        # MPC параметри
        "Np": 6,
        "Nc": 4,
        "lag": 2,
        "λ_obj": 0.1,
        
        # Ваги та уставки
        "w_fe": 5.0,
        "w_mass": 2.0,
        "ref_fe": 54.0,
        "ref_mass": 58.0,
        "tolerance_fe_percent": 1.5,
        "tolerance_mass_percent": 2.0,
        
        # Trust region
        "adaptive_trust_region": True,
        "initial_trust_radius": 2.0,
        "min_trust_radius": 0.5,
        "max_trust_radius": 2.0,
        "trust_decay_factor": 0.9,
        "rho_trust": 0.3,
        
        # EKF параметри
        "P0": 0.01,
        "Q_phys": 600,
        "Q_dist": 1,
        "R": 1.0,
        "q_adaptive_enabled": True,
        "q_alpha": 0.90,
        "q_nis_threshold": 3.0,
        
        # Перенавчання
        "enable_retraining": True,
        "retrain_period": 50,
        "retrain_innov_threshold": 0.3,
        "retrain_window_size": 1000,
        
        # Обмеження
        "use_soft_constraints": True,
        "delta_u_max": 0.8,
        "u_min": 20.0,
        "u_max": 40.0,
        
        # Процес
        "plant_model_type": "rf",
        "noise_level": "low",

        # Нелінійність
        "enable_nonlinear": True,
        "nonlinear_config": {
            "concentrate_fe_percent": ["pow", 2],
            "concentrate_mass_flow": ["pow", 1.5]
        },
        
        # Аномалії
        "anomaly_params": {
            "window": 25,
            "spike_z": 4.0,
            "drop_rel": 0.30,
            "freeze_len": 5,
            "enabled": True
        },
        
        "run_analysis": False
    }

    # LIN-MPC
    lin_mpc = {
        "name": "lin-mpc",
        "description": "lin-MPC",
        
        # Основні параметри
        "N_data": 5000,
        "control_pts": 500,
        "seed": 42,
        
        # 🎯 ПОЛІНОМІАЛЬНА МОДЕЛЬ
        "model_type": "linear",
        "linear_type": "ridge",
        "poly_degree": 2,
        "include_bias": True,
        "find_optimal_params": True,
        "alpha": 1.0,  # Початкове значення для Ridge
        
        # MPC параметри
        "Np": 6,
        "Nc": 4,
        "lag": 2,
        "λ_obj": 0.1,
        
        # Ваги та уставки
        "w_fe": 5.0,
        "w_mass": 2.0,
        "ref_fe": 54.0,
        "ref_mass": 58.0,
        "tolerance_fe_percent": 1.5,
        "tolerance_mass_percent": 2.0,
        
        # Trust region
        "adaptive_trust_region": True,
        "initial_trust_radius": 2.0,
        "min_trust_radius": 0.5,
        "max_trust_radius": 2.0,
        "trust_decay_factor": 0.9,
        "rho_trust": 0.3,
        
        # EKF параметри
        "P0": 0.01,
        "Q_phys": 600,
        "Q_dist": 1,
        "R": 1.0,
        "q_adaptive_enabled": True,
        "q_alpha": 0.90,
        "q_nis_threshold": 3.0,
        
        # Перенавчання
        "enable_retraining": True,
        "retrain_period": 50,
        "retrain_innov_threshold": 0.3,
        "retrain_window_size": 1000,
        
        # Обмеження
        "use_soft_constraints": True,
        "delta_u_max": 0.8,
        "u_min": 20.0,
        "u_max": 40.0,
        
        # Процес
        "plant_model_type": "rf",
        "noise_level": "low",

        # Нелінійність
        "enable_nonlinear": True,
        "nonlinear_config": {
            "concentrate_fe_percent": ["pow", 2],
            "concentrate_mass_flow": ["pow", 1.5]
        },
        
        # Аномалії
        "anomaly_params": {
            "window": 25,
            "spike_z": 4.0,
            "drop_rel": 0.30,
            "freeze_len": 5,
            "enabled": True
        },
        
        "run_analysis": False
    }
    
    # KRR-MPC-TEST
    krr_test = {
        "name": "krr-test",
        "description": "KRR-MPC-TEST",
        
        # Основні параметри
        "N_data": 5000,
        "control_pts": 500,
        "seed": 42,
        "train_size": 0.9,
        "val_size": 0.08,
        "test_size": 0.02,
        
        # Модель
        "model_type": "krr",
        "kernel": "rbf",
        "find_optimal_params": True,
        
        # MPC параметри
        "Np": 6,
        "Nc": 4,
        "lag": 2,
        "λ_obj": 0.5,
        
        # Ваги та уставки
        "w_fe": 5.0,
        "w_mass": 2.0,
        "ref_fe": 54.0,
        "ref_mass": 58.0,
        "tolerance_fe_percent": 1.5,
        "tolerance_mass_percent": 2.0,
        
        # Trust region
        "adaptive_trust_region": True,
        "initial_trust_radius": 0.5,
        "min_trust_radius": 0.42,
        "max_trust_radius": 1.0,
        "trust_decay_factor": 0.96,
        "rho_trust": 0.015,
        "linearization_check_enabled": True,
        "max_linearization_distance": 0.8,
        "retrain_linearization_threshold": 1.0,
        
        # EKF параметри
        "P0": 0.01,
        "Q_phys": 600,
        "Q_dist": 1,
        "R": 1.0,
        "q_adaptive_enabled": True,
        "q_alpha": 0.90,
        "q_nis_threshold": 3.0,
        
        # Перенавчання
        "enable_retraining": True,
        "retrain_period": 50,
        "retrain_innov_threshold": 0.3,
        "retrain_window_size": 1000,
        
        # Обмеження
        "use_soft_constraints": True,
        "delta_u_max": 0.8,
        "u_min": 20.0,
        "u_max": 40.0,
        
        # Процес
        "plant_model_type": "rf",
        "noise_level": "low",

        # Нелінійність
        "enable_nonlinear": True,
        "nonlinear_config": {
            "concentrate_fe_percent": ["pow", 2],
            "concentrate_mass_flow": ["pow", 1.5]
        },
        
        # Аномалії
        "anomaly_params": {
            "window": 15,
            "spike_z": 2.5,
            "drop_rel": 0.15,
            "freeze_len": 4,
            "enabled": True
        },
        
        "run_analysis": False
    }
    
    # Зберігаємо конфігурації
    configs = [krr_mpc, svr_mpc, lin_mpc, krr_test]
    
    for config in configs:
        config_name = config["name"]
        save_config(config, config_name)
    
    print(f"✅ Створено {len(configs)} стандартних конфігурацій (включаючи L-MPC)")


def prompt_manual_adjustments(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Запитує користувача про ручні корегування базової конфігурації з підтримкою L-MPC параметрів.
    
    Args:
        base_config: Базова конфігурація
        
    Returns:
        Словник з ручними корегуваннями
    """
    print(f"\n🔧 РУЧНЕ КОРЕГУВАННЯ ПАРАМЕТРІВ")
    print("=" * 50)
    print("Натисніть Enter щоб залишити значення без змін")
    
    adjustments = {}
    
    # Групуємо параметри за категоріями з підтримкою L-MPC
    categories = {
        "📊 Основні параметри": [
            ("N_data", "Кількість точок даних", int),
            ("control_pts", "Кроків керування", int),
        ],
        "🤖 Модель": [
            ("model_type", "Тип моделі (krr/svr/linear)", str),
        ],
        "🔧 Лінійна модель (якщо model_type=linear)": [
            ("linear_type", "Тип лінійної моделі (ols/ridge/lasso)", str),
            ("poly_degree", "Ступінь поліному (1-3)", int),
            ("alpha", "Коефіцієнт регуляризації", float),
        ],
        "🔄 Ядерна модель (якщо model_type=krr/svr)": [
            ("kernel", "Тип ядра (rbf/linear/poly)", str),
            ("find_optimal_params", "Автопошук параметрів (True/False)", str),
        ],
        "🎯 MPC": [
            ("Np", "Горизонт прогнозування", int),
            ("Nc", "Горизонт керування", int),
            ("λ_obj", "Коефіцієнт згладжування", float)
        ],
        "📍 Уставки": [
            ("ref_fe", "Уставка Fe %", float),
            ("ref_mass", "Уставка масового потоку", float),
            ("w_fe", "Вага для Fe", float),
            ("w_mass", "Вага для масового потоку", float)
        ],
        "🎯 Толерантності оцінки": [
            ("tolerance_fe_percent", "Толерантність Fe (%)", float),
            ("tolerance_mass_percent", "Толерантність Mass (%)", float)
        ]
    }
    
    for category_name, params_list in categories.items():
        print(f"\n{category_name}:")
        
        for param_name, description, param_type in params_list:
            if param_name not in base_config:
                continue
                
            current_value = base_config[param_name]
            
            try:
                prompt = f"  {description} (поточне: {current_value}): "
                user_input = input(prompt).strip()
                
                if user_input:  # Користувач ввів щось
                    if param_type == str:
                        # Спеціальна обробка для boolean значень
                        if param_name == "find_optimal_params":
                            if user_input.lower() in ['true', 't', '1', 'так']:
                                adjustments[param_name] = True
                            elif user_input.lower() in ['false', 'f', '0', 'ні']:
                                adjustments[param_name] = False
                            else:
                                adjustments[param_name] = user_input
                        else:
                            adjustments[param_name] = user_input
                    elif param_type in [int, float]:
                        adjustments[param_name] = param_type(user_input)
                        
            except (ValueError, TypeError) as e:
                print(f"    ⚠️ Некоректне значення для {param_name}: {e}")
                continue
    
    return adjustments

# =============================================================================
# === ГОЛОВНА ФУНКЦІЯ ===
# =============================================================================

def simulate_mpc_with_config(
    reference_df: pd.DataFrame,
    config_name: str = "conservative",
    manual_overrides: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable] = None,
    save_results: bool = True,
    show_evaluation_plots: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Запускає симуляцію MPC з базовою конфігурацією та ручними корегуваннями.
    
    Args:
        reference_df: Референсні дані
        config_name: Назва базової конфігурації (за замовчуванням "conservative")
        manual_overrides: Словник для ручного перевизначення параметрів
        progress_callback: Функція зворотного виклику для прогресу
        save_results: Чи зберігати результати автоматично (за замовчуванням True)
        show_evaluation_plots: Чи показувати графіки оцінки (за замовчуванням False)  # ✅ ДОДАЄМО ОПИС
        **kwargs: Додаткові параметри для перевизначення
        
    Returns:
        Кортеж (результати, метрики)
        
    Raises:
        FileNotFoundError: Якщо конфігурація не знайдена
        ImportError: Якщо не вдається імпортувати simulate_mpc
    """
    
    try:
        # Імпортуємо функцію симуляції
        from sim import simulate_mpc
        
        # 1. Завантажуємо базову конфігурацію
        try:
            params = load_config(config_name)
            print(f"📋 Завантажено базову конфігурацію: {config_name}")
            
            # Показуємо ключові параметри
            key_params = ['model_type', 'Np', 'Nc', 'ref_fe', 'ref_mass', 'w_fe', 'w_mass', 'λ_obj']
            print("📊 Ключові параметри базової конфігурації:")
            for param in key_params:
                if param in params:
                    print(f"   • {param}: {params[param]}")
                    
        except FileNotFoundError:
            print(f"⚠️ Конфігурація '{config_name}' не знайдена")
            available = list_configs()
            if available:
                print(f"Доступні конфігурації: {', '.join(available)}")
                print("Створюємо стандартні конфігурації...")
                create_default_configs()
                params = load_config("conservative")
            else:
                raise FileNotFoundError(f"Не вдається завантажити конфігурацію '{config_name}'")
        
        # 2. Застосовуємо ручні корегування
        if manual_overrides:
            print(f"\n🔧 Застосовуємо {len(manual_overrides)} ручних корегувань:")
            for key, value in manual_overrides.items():
                old_value = params.get(key, "не задано")
                params[key] = value
                print(f"   • {key}: {old_value} → {value}")
        
        # 3. Застосовуємо додаткові kwargs
        if kwargs:
            print(f"\n⚙️ Застосовуємо {len(kwargs)} додаткових параметрів:")
            for key, value in kwargs.items():
                if manual_overrides and key not in manual_overrides:
                    old_value = params.get(key, "не задано")
                    params[key] = value
                    print(f"   • {key}: {old_value} → {value}")
                elif not manual_overrides:
                    old_value = params.get(key, "не задано")
                    params[key] = value
                    print(f"   • {key}: {old_value} → {value}")
        
        # 4. ✅ ДОДАЄМО progress_callback та show_evaluation_plots
        if progress_callback:
            params['progress_callback'] = progress_callback
        if show_evaluation_plots:  # ✅ ДОДАЄМО ПЕРЕДАЧУ ПАРАМЕТРА
            params['show_evaluation_plots'] = show_evaluation_plots
        
        # 5. Показуємо фінальну конфігурацію
        print(f"\n✅ Фінальна конфігурація для запуску:")
        key_params = ['model_type', 'Np', 'Nc', 'ref_fe', 'ref_mass', 'w_fe', 'w_mass', 'λ_obj']
        for param in key_params:
            if param in params:
                print(f"   • {param}: {params[param]}")
        
        # 6. Фільтруємо параметри для simulate_mpc
        filtered_params = _filter_for_simulate_mpc(params)
        
        # 7. Запускаємо основну симуляцію
        print("🚀 Викликаємо simulate_mpc...")
        results_df, metrics = simulate_mpc(reference_df, **filtered_params)
        
        # ✅ ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
        if save_results:
            print("\n💾 Зберігаємо результати симуляції...")
            
            # Створюємо назву конфігурації з корегуваннями
            config_save_name = config_name
            if manual_overrides:
                config_save_name += "_modified"
            
            try:
                saved_path = save_simulation_results(results_df, config_save_name, metrics)
                print(f"✅ Результати збережено: {saved_path}")
                
                # Додаткова інформація
                file_size = Path(saved_path).stat().st_size / (1024 * 1024)
                print(f"📁 Розмір файлу: {file_size:.2f} MB")
                
            except Exception as save_error:
                print(f"⚠️ Помилка при збереженні: {save_error}")
        
        return results_df, metrics
        
    except Exception as e:
        print(f"❌ Помилка в simulate_mpc_with_config: {e}")
        import traceback
        traceback.print_exc()
        raise

# Також потрібно додати відсутню функцію validate_config:

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Валідує конфігурацію на наявність обов'язкових параметрів з підтримкою L-MPC та K-MPC.
    
    Args:
        config: Конфігурація для перевірки
        
    Returns:
        Кортеж (валідна, список_помилок)
    """
    required_params = ['model_type', 'Np', 'Nc', 'N_data', 'control_pts']
    errors = []
    
    for param in required_params:
        if param not in config:
            errors.append(f"Відсутній обов'язковий параметр: {param}")
    
    # Перевірка типів та діапазонів
    if 'Np' in config and (not isinstance(config['Np'], int) or config['Np'] < 1):
        errors.append("Np повинен бути додатним цілим числом")
    
    if 'Nc' in config and (not isinstance(config['Nc'], int) or config['Nc'] < 1):
        errors.append("Nc повинен бути додатним цілим числом")
    
    # 🆕 РОЗШИРЕНА ВАЛІДАЦІЯ ДЛЯ РІЗНИХ ТИПІВ МОДЕЛЕЙ
    model_type = config.get('model_type', '').lower()
    
    if model_type not in ['krr', 'svr', 'linear', 'gpr']:
        errors.append("model_type повинен бути одним з: krr, svr, linear, gpr")
    
    # 🎯 ВАЛІДАЦІЯ L-MPC ПАРАМЕТРІВ
    if model_type == 'linear':
        linear_type = config.get('linear_type', 'ols')
        if linear_type not in ['ols', 'ridge', 'lasso']:
            errors.append("linear_type повинен бути одним з: ols, ridge, lasso")
            
        poly_degree = config.get('poly_degree', 1)
        if not isinstance(poly_degree, int) or not (1 <= poly_degree <= 3):
            errors.append("poly_degree повинен бути цілим числом від 1 до 3")
            
        if linear_type in ['ridge', 'lasso']:
            alpha = config.get('alpha', 1.0)
            if not isinstance(alpha, (int, float)) or alpha <= 0:
                errors.append("alpha повинен бути додатним числом для Ridge/Lasso")
    
    # 🎯 ВАЛІДАЦІЯ K-MPC ПАРАМЕТРІВ
    elif model_type in ['krr', 'svr']:
        kernel = config.get('kernel', 'rbf')
        valid_kernels = ['rbf', 'linear', 'poly']
        if kernel not in valid_kernels:
            errors.append(f"kernel повинен бути одним з: {', '.join(valid_kernels)}")
    
    return len(errors) == 0, errors

# =============================================================================
# === УТИЛІТАРНІ ФУНКЦІЇ ===
# =============================================================================

def get_config_info(config_name: str) -> Optional[Dict[str, Any]]:
    """
    Отримує інформацію про конфігурацію без повного завантаження.
    
    Args:
        config_name: Назва конфігурації
        
    Returns:
        Словник з базовою інформацією або None якщо конфігурація не знайдена
    """
    try:
        config = load_config(config_name)
        return {
            'name': config.get('name', config_name),
            'description': config.get('description', 'Опис відсутній'),
            'model_type': config.get('model_type', 'не вказано'),
            'N_data': config.get('N_data', 'не вказано'),
            'Np': config.get('Np', 'не вказано'),
            'Nc': config.get('Nc', 'не вказано')
        }
    except (FileNotFoundError, ValueError):
        return None

def generate_results_filename(config_name: str, file_format: str = "parquet") -> Path:
    """
    Генерує шлях до файлу результатів з timestamp.
    
    Args:
        config_name: Назва конфігурації
        file_format: Формат файлу ('parquet', 'csv', 'json')
        
    Returns:
        Path до файлу результатів
    """
    import pandas as pd
    
    _ensure_results_dir_exists()
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mpc_results_{config_name}_{timestamp}.{file_format}"
    
    return _get_results_dir() / filename

def save_simulation_results(results_df: pd.DataFrame, config_name: str, 
                          metrics: Optional[Dict] = None) -> str:
    """
    Зберігає результати симуляції в папку mpc_results.
    
    Args:
        results_df: DataFrame з результатами симуляції
        config_name: Назва використаної конфігурації
        metrics: Додаткові метрики для збереження
        
    Returns:
        Шлях до збереженого файлу
    """
    # Зберігаємо основні результати
    results_file = generate_results_filename(config_name, "parquet")
    results_df.to_parquet(results_file, index=False)
    
    # Додатково зберігаємо метрики якщо є
    if metrics:
        metrics_file = generate_results_filename(config_name, "json")
        metrics_file = metrics_file.with_name(metrics_file.name.replace("mpc_results_", "mpc_metrics_"))
        
        # Очищаємо метрики для JSON серіалізації
        clean_metrics = {}
        for key, value in metrics.items():
            try:
                import numpy as np
                if isinstance(value, (np.integer, np.floating)):
                    clean_metrics[key] = float(value)
                elif isinstance(value, np.ndarray):
                    clean_metrics[key] = value.tolist()
                elif pd.isna(value):
                    clean_metrics[key] = None
                else:
                    clean_metrics[key] = value
            except:
                clean_metrics[key] = str(value)
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(clean_metrics, f, indent=4, ensure_ascii=False)
    
    return str(results_file)

def list_saved_results() -> List[Dict[str, str]]:
    """
    Повертає список збережених результатів.
    
    Returns:
        Список словників з інформацією про файли результатів
    """
    results_dir = _get_results_dir()
    if not results_dir.exists():
        return []
    
    results = []
    for file_path in results_dir.glob("mpc_results_*.parquet"):
        # Парсимо назву файлу: mpc_results_CONFIG_TIMESTAMP.parquet
        name_parts = file_path.stem.split('_')
        if len(name_parts) >= 4:
            config_name = '_'.join(name_parts[2:-2]) if len(name_parts) > 4 else name_parts[2]
            timestamp = '_'.join(name_parts[-2:])
            
            results.append({
                'file': file_path.name,
                'config': config_name,
                'timestamp': timestamp,
                'path': str(file_path),
                'size_mb': file_path.stat().st_size / (1024 * 1024)
            })
    
    return sorted(results, key=lambda x: x['timestamp'], reverse=True)
    """
    Валідує конфігурацію на наявність обов'язкових параметрів.
    
    Args:
        config: Конфігурація для перевірки
        
    Returns:
        Кортеж (валідна, список_помилок)
    """
    required_params = ['model_type', 'Np', 'Nc', 'N_data', 'control_pts']
    errors = []
    
    for param in required_params:
        if param not in config:
            errors.append(f"Відсутній обов'язковий параметр: {param}")
    
    # Перевірка типів та діапазонів
    if 'Np' in config and not isinstance(config['Np'], int) or config['Np'] < 1:
        errors.append("Np повинен бути додатним цілим числом")
    
    if 'Nc' in config and not isinstance(config['Nc'], int) or config['Nc'] < 1:
        errors.append("Nc повинен бути додатним цілим числом")
    
    if 'model_type' in config and config['model_type'] not in ['krr', 'svr', 'linear', 'gpr']:
        errors.append("model_type повинен бути одним з: krr, svr, linear, gpr")
    
    return len(errors) == 0, errors

if __name__ == '__main__':
    create_default_configs_ext()
    