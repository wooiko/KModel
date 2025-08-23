# model_comparison_service.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path
from datetime import datetime

from data_gen import StatefulDataGenerator
from model import KernelModel


class ModelComparisonService:
    """
    Службовий клас для порівняння лінійних (ARX) та ядерних моделей у рамках дисертаційного дослідження.
    
    Основні можливості:
    - Створення симуляційних даних з керованими аномаліями
    - Навчання та порівняння лінійних і ядерних моделей
    - Аналіз нелінійності процесу
    - Генерація комплексних візуалізацій
    - Збереження результатів у JSON форматі
    """
    
    def __init__(self, reference_df: Optional[pd.DataFrame] = None, 
                 output_dir: Optional[str] = None):
        """
        Ініціалізація сервісу порівняння моделей.
        
        Args:
            reference_df: Референтні дані для симуляції
            output_dir: Базова директорія для збереження результатів. 
                       Якщо None, створюється 'results/YYYY-MM-DD_HH-MM-SS'
        """
        self.reference_df = self._load_reference_data(reference_df)
        self.results = {}
        self.figures = {}
        self._setup_output_directories(output_dir)
        
    def _load_reference_data(self, reference_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Завантаження референтних даних"""
        if reference_df is None:
            try:
                reference_df = pd.read_parquet('processed.parquet')
                print(f"✅ Завантажено {len(reference_df)} записів з processed.parquet")
            except FileNotFoundError:
                print("❌ Файл 'processed.parquet' не знайдено")
                raise
        return reference_df
    
    def _setup_output_directories(self, output_dir: Optional[str]) -> None:
        """Створення структури директорій для збереження результатів"""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"evaluation_results/models/{timestamp}"
        
        self.output_dir = Path(output_dir)
        
        # Створення підпапок
        self.dirs = {
            'data': self.output_dir / 'data',
            'visualizations': self.output_dir / 'visualizations', 
            'reports': self.output_dir / 'reports',
            'latex': self.output_dir / 'latex',
            'comparisons': self.output_dir / 'comparisons'
        }
        
        # Створення всіх директорій
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"📁 Результати зберігатимуться у: {self.output_dir.absolute()}")
        
    def create_anomaly_config(self, N_data: int, train_frac: float = 0.7, 
                             val_frac: float = 0.15, test_frac: float = 0.15,
                             seed: int = 42, severity: str = "mild", 
                             include_train: bool = False) -> dict:
        """
        Генерує reproducible anomaly_config для DataGenerator.generate_anomalies().
        
        Args:
            N_data: Загальна кількість точок даних
            train_frac: Частка тренувальних даних
            val_frac: Частка валідаційних даних  
            test_frac: Частка тестових даних
            seed: Насіння для відтворюваності
            severity: Рівень серйозності аномалій ('mild' | 'medium' | 'strong')
            include_train: Чи включати аномалії в тренувальні дані
            
        Returns:
            dict: Конфігурація аномалій для генератора даних
        """
        total = train_frac + val_frac + test_frac
        train_frac, val_frac, test_frac = train_frac/total, val_frac/total, test_frac/total

        train_end = int(train_frac * N_data)
        val_end   = train_end + int(val_frac * N_data)
        segments = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, N_data)}

        base_durations = {"spike": 1, "drift": 25, "drop": 20, "freeze": 15}
        sev_map = {"mild": (0.08, 0.15), "medium": (0.12, 0.22), "strong": (0.18, 0.30)}
        mag_lo, mag_hi = sev_map.get(severity, sev_map["mild"])

        rng = np.random.default_rng(seed)

        def seg_bounds(name):
            s, e = segments[name]
            return s, max(s, e-1), max(0, e-s)

        def pick_start(seg_name, dur):
            s, e_minus1, length = seg_bounds(seg_name)
            if length <= 0:
                return None
            dur = min(dur, length)
            hi = max(s, e_minus1 - (dur-1))
            return int(rng.integers(low=s, high=hi+1)), dur

        params = [
            "ore_mass_flow", "feed_fe_percent", "solid_feed_percent",
            "concentrate_fe_percent", "tailings_fe_percent",
            "concentrate_mass_flow", "tailings_mass_flow"
        ]
        cfg = {p: [] for p in params}

        def add_anom(param, seg, typ, mag=None, force_positive=False):
            dur = base_durations[typ]
            sd = pick_start(seg, dur)
            if sd is None:
                return
            start, dur = sd
            if typ == "freeze":
                cfg[param].append({"start": start, "duration": dur, "type": typ})
            else:
                m = float(abs(mag) if mag is not None else rng.uniform(mag_lo, mag_hi))
                if typ != "drop" and not force_positive and rng.random() < 0.5:
                    m = -m
                if typ == "drop":
                    m = abs(m)
                cfg[param].append({"start": start, "duration": dur, "magnitude": m, "type": typ})

        # План аномалій
        if include_train: 
            add_anom("ore_mass_flow", "train", "drift")
            add_anom("solid_feed_percent", "train", "freeze")
            
        add_anom("ore_mass_flow", "val",  "drift")
        add_anom("ore_mass_flow", "test", "spike")
        add_anom("solid_feed_percent", "val", "freeze")
        add_anom("feed_fe_percent", "test", "spike")
        add_anom("concentrate_mass_flow", "val", "drop", force_positive=True)
        add_anom("tailings_mass_flow", "test", "drift")
        add_anom("concentrate_fe_percent", "val", "spike")
        add_anom("tailings_fe_percent", "test", "freeze")

        return cfg
    
    def create_simulation_data(self, params: dict) -> Tuple[StatefulDataGenerator, pd.DataFrame]:
        """
        Створення симуляційних даних через StatefulDataGenerator.
        
        Args:
            params: Параметри симуляції
            
        Returns:
            Tuple[StatefulDataGenerator, pd.DataFrame]: Генератор та симуляційні дані
        """
        true_gen = StatefulDataGenerator(
            self.reference_df,
            ore_flow_var_pct=3.0,
            time_step_s=params['time_step_s'],
            time_constants_s=params['time_constants_s'],
            dead_times_s=params['dead_times_s'],
            true_model_type=params['plant_model_type'],
            seed=params['seed']
        )

        # Аномалії
        anomaly_cfg = None
        if params.get('use_anomalies', True):
            anomaly_cfg = self.create_anomaly_config(
                N_data=params['N_data'],
                train_frac=params.get('train_size', 0.8),
                val_frac=params.get('val_size', 0.1),
                test_frac=params.get('test_size', 0.1),
                seed=params['seed'],
                severity=params.get('anomaly_severity', 'mild'),
                include_train=params.get('anomaly_in_train', False),
            )

        # Базові дані
        df_true_orig = true_gen.generate(
            T=params['N_data'],
            control_pts=params['control_pts'],
            n_neighbors=params['n_neighbors'],
            noise_level=params.get('noise_level', 'none'),
            anomaly_config=anomaly_cfg
        )

        # Нелінійний варіант
        if params.get('enable_nonlinear', False):
            df_true = true_gen.generate_nonlinear_variant(
                base_df=df_true_orig,
                non_linear_factors=params['nonlinear_config'],
                noise_level='none',
                anomaly_config=anomaly_cfg
            )
        else:
            df_true = df_true_orig

        return true_gen, df_true
    
    def create_lagged_matrices(self, df: pd.DataFrame, lag: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Створення лагових матриць для порівняння моделей.
        
        Args:
            df: DataFrame з даними
            lag: Кількість лагів
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Матриці X (входи) та Y (виходи)
        """
        input_vars = ['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']
        output_vars = ['concentrate_fe', 'concentrate_mass']
        
        # Перевірка альтернативних назв
        if 'concentrate_fe' not in df.columns and 'concentrate_fe_percent' in df.columns:
            output_vars = ['concentrate_fe_percent', 'concentrate_mass_flow']
        
        # Перевірка наявності колонок
        missing_vars = [var for var in input_vars + output_vars if var not in df.columns]
        if missing_vars:
            print(f"⚠️ Відсутні колонки: {missing_vars}")
            print(f"📋 Доступні колонки: {list(df.columns)}")
            return StatefulDataGenerator.create_lagged_dataset(df, lags=lag)
        
        n = len(df)
        X, Y = [], []
        
        for i in range(lag, n):
            row = []
            for var in input_vars:
                for j in range(lag + 1):
                    row.append(df[var].iloc[i - j])
            X.append(row)
            Y.append([df[var].iloc[i] for var in output_vars])
        
        return np.array(X), np.array(Y)
    
    def analyze_nonlinearity(self, df_sim: pd.DataFrame, true_gen: StatefulDataGenerator) -> dict:
        """
        Аналіз нелінійності симуляційних даних.
        
        Args:
            df_sim: Симуляційні дані
            true_gen: Генератор даних
            
        Returns:
            dict: Метрики нелінійності
        """
        metrics = {}
        
        # Оцінка S-подібності через варіації градієнтів
        if 'concentrate_fe' in df_sim.columns:
            fe_values = df_sim['concentrate_fe'].values
            fe_gradients = np.diff(fe_values)
            metrics['fe_gradient_variance'] = np.var(fe_gradients)
            metrics['fe_gradient_skewness'] = pd.Series(fe_gradients).skew()
        
        # Нелінійні взаємодії через кореляційний аналіз
        numeric_cols = df_sim.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            pearson_corr = df_sim[numeric_cols].corr(method='pearson')
            spearman_corr = df_sim[numeric_cols].corr(method='spearman')
            nonlinearity_indicator = abs(spearman_corr - pearson_corr).mean().mean()
            metrics['correlation_nonlinearity'] = nonlinearity_indicator
        
        # Ентропійна оцінка складності
        if 'solid_feed_percent' in df_sim.columns:
            control_changes = np.abs(np.diff(df_sim['solid_feed_percent']))
            control_entropy = -np.sum((control_changes + 1e-10) * np.log(control_changes + 1e-10))
            metrics['control_complexity'] = control_entropy
        
        # Характеристики розподілу
        if 'concentrate_mass' in df_sim.columns:
            mass_values = df_sim['concentrate_mass'].values
            metrics['mass_distribution_kurtosis'] = pd.Series(mass_values).kurtosis()
            metrics['mass_distribution_skewness'] = pd.Series(mass_values).skew()
        
        return metrics
    
    def train_models(self, X_train_scaled: np.ndarray, Y_train_scaled: np.ndarray,
                    X_val_scaled: Optional[np.ndarray] = None, 
                    Y_val_scaled: Optional[np.ndarray] = None,
                    **kwargs) -> Tuple[KernelModel, KernelModel, dict]:
        """
        Навчання лінійної та ядерної моделей.
        
        Args:
            X_train_scaled: Масштабовані тренувальні входи
            Y_train_scaled: Масштабовані тренувальні виходи
            X_val_scaled: Масштабовані валідаційні входи (опційно)
            Y_val_scaled: Масштабовані валідаційні виходи (опційно)
            **kwargs: Додаткові параметри навчання
            
        Returns:
            Tuple: (лінійна_модель, ядерна_модель, метрики_навчання)
        """
        training_metrics = {}
        
        # Лінійна модель (ARX)
        print("\n🔴 НАВЧАННЯ ЛІНІЙНОЇ МОДЕЛІ (ARX)")
        print("-" * 40)
        linear_model = KernelModel(
            model_type='linear', 
            linear_type='ols', 
            poly_degree=1, 
            include_bias=True
        )
        
        start_time = time.time()
        try:
            if X_val_scaled is not None and Y_val_scaled is not None:
                linear_model.fit(X_train_scaled, Y_train_scaled, 
                               X_val=X_val_scaled, Y_val=Y_val_scaled)
            else:
                linear_model.fit(X_train_scaled, Y_train_scaled)
        except TypeError:
            linear_model.fit(X_train_scaled, Y_train_scaled)
        linear_train_time = time.time() - start_time
        
        print(f"   ⏱️ Час навчання: {linear_train_time:.3f} сек")
        
        # Ядерна модель (KRR)  
        print("\n🟢 НАВЧАННЯ ЯДЕРНОЇ МОДЕЛІ (KRR)")
        print("-" * 40)
        kernel_model = KernelModel(
            model_type='krr',
            kernel='rbf',
            find_optimal_params=kwargs.get('find_optimal_params', True),
            n_iter_random_search=kwargs.get('n_iter_search', 20)
        )
        
        start_time = time.time()
        try:
            if X_val_scaled is not None and Y_val_scaled is not None:
                kernel_model.fit(X_train_scaled, Y_train_scaled,
                               X_val=X_val_scaled, Y_val=Y_val_scaled)
            else:
                kernel_model.fit(X_train_scaled, Y_train_scaled)
        except TypeError:
            kernel_model.fit(X_train_scaled, Y_train_scaled)
        kernel_train_time = time.time() - start_time
        
        print(f"   ⏱️ Час навчання: {kernel_train_time:.3f} сек")
        
        training_metrics = {
            'linear_train_time': linear_train_time,
            'kernel_train_time': kernel_train_time
        }
        
        return linear_model, kernel_model, training_metrics
    
    def evaluate_models(self, linear_model: KernelModel, kernel_model: KernelModel,
                       X_test_scaled: np.ndarray, Y_test: np.ndarray, 
                       y_scaler: StandardScaler) -> dict:
        """
        Оцінка продуктивності моделей.
        
        Args:
            linear_model: Навчена лінійна модель
            kernel_model: Навчена ядерна модель
            X_test_scaled: Масштабовані тестові входи
            Y_test: Немасштабовані тестові виходи
            y_scaler: Скейлер для виходів
            
        Returns:
            dict: Метрики оцінки
        """
        # Прогнозування
        Y_pred_linear_scaled = linear_model.predict(X_test_scaled)
        Y_pred_linear = y_scaler.inverse_transform(Y_pred_linear_scaled)
        
        Y_pred_kernel_scaled = kernel_model.predict(X_test_scaled)
        Y_pred_kernel = y_scaler.inverse_transform(Y_pred_kernel_scaled)
        
        # Метрики
        linear_mse = mean_squared_error(Y_test, Y_pred_linear)
        linear_rmse_fe = np.sqrt(mean_squared_error(Y_test[:, 0], Y_pred_linear[:, 0]))
        linear_rmse_mass = np.sqrt(mean_squared_error(Y_test[:, 1], Y_pred_linear[:, 1]))
        
        kernel_mse = mean_squared_error(Y_test, Y_pred_kernel)
        kernel_rmse_fe = np.sqrt(mean_squared_error(Y_test[:, 0], Y_pred_kernel[:, 0]))
        kernel_rmse_mass = np.sqrt(mean_squared_error(Y_test[:, 1], Y_pred_kernel[:, 1]))
        
        # Покращення
        improvement_mse = ((linear_mse - kernel_mse) / (linear_mse + 1e-12)) * 100
        improvement_fe = ((linear_rmse_fe - kernel_rmse_fe) / (linear_rmse_fe + 1e-12)) * 100
        improvement_mass = ((linear_rmse_mass - kernel_rmse_mass) / (linear_rmse_mass + 1e-12)) * 100
        
        print(f"   📊 Лінійна MSE: {linear_mse:.6f}")
        print(f"   📊 Лінійна RMSE Fe: {linear_rmse_fe:.3f}")
        print(f"   📊 Лінійна RMSE Mass: {linear_rmse_mass:.3f}")
        print(f"   📊 Ядерна MSE: {kernel_mse:.6f}")
        print(f"   📊 Ядерна RMSE Fe: {kernel_rmse_fe:.3f}")
        print(f"   📊 Ядерна RMSE Mass: {kernel_rmse_mass:.3f}")
        
        return {
            'Y_pred_linear': Y_pred_linear,
            'Y_pred_kernel': Y_pred_kernel,
            'linear_mse': linear_mse,
            'linear_rmse_fe': linear_rmse_fe,
            'linear_rmse_mass': linear_rmse_mass,
            'kernel_mse': kernel_mse,
            'kernel_rmse_fe': kernel_rmse_fe,
            'kernel_rmse_mass': kernel_rmse_mass,
            'improvement_mse': improvement_mse,
            'improvement_fe': improvement_fe,
            'improvement_mass': improvement_mass
        }

    def run_comparison(self, **kwargs) -> dict:
        """
        Виконання повного порівняння лінійних та ядерних моделей.
        
        Args:
            **kwargs: Параметри симуляції та навчання
            
        Returns:
            dict: Повні результати порівняння
        """
        print("🎓 ПОРІВНЯННЯ МОДЕЛЕЙ ДЛЯ ДИСЕРТАЦІЇ")
        print("=" * 60)
        print("Розділ 2.1.1: Логічний перехід до ядерних моделей")
        print("=" * 60)

        # Параметри симуляції
        simulation_params = {
            'N_data': kwargs.get('N_data', 7000),
            'control_pts': 700,
            'lag': kwargs.get('lag', 2),
            'train_size': kwargs.get('train_size', 0.8),
            'val_size': kwargs.get('val_size', 0.1),
            'test_size': kwargs.get('test_size', 0.1),
            'time_step_s': 5,
            'time_constants_s': {
                'concentrate_fe_percent': 8.0,
                'tailings_fe_percent': 10.0,
                'concentrate_mass_flow': 5.0,
                'tailings_mass_flow': 7.0
            },
            'dead_times_s': {
                'concentrate_fe_percent': 20.0,
                'tailings_fe_percent': 25.0,
                'concentrate_mass_flow': 20.0,
                'tailings_mass_flow': 25.0
            },
            'plant_model_type': 'rf',
            'seed': kwargs.get('seed', 42),
            'n_neighbors': 5,
            'noise_level': kwargs.get('noise_level', 'none'),
            'enable_nonlinear': True,
            'nonlinear_config': {
                'concentrate_fe_percent': ('pow', 2.0),
                'concentrate_mass_flow': ('pow', 1.5)
            },
            'use_anomalies': kwargs.get('use_anomalies', True),
            'anomaly_severity': kwargs.get('anomaly_severity', 'mild'),
            'anomaly_in_train': kwargs.get('anomaly_in_train', False),
        }

        # Створення симуляційних даних
        print(f"📈 Створення симуляційних даних (N={simulation_params['N_data']}, L={simulation_params['lag']})...")
        true_gen, df_sim = self.create_simulation_data(simulation_params)

        # Лаговані матриці
        X, Y = self.create_lagged_matrices(df_sim, simulation_params['lag'])
        print(f"   Розмірність X: {X.shape}, Y: {Y.shape}")

        # Спліт на train/val/test
        n = X.shape[0]
        n_train = int(simulation_params['train_size'] * n)
        n_val = int(simulation_params['val_size'] * n)

        X_train, Y_train = X[:n_train], Y[:n_train]
        X_val, Y_val = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
        X_test, Y_test = X[n_train + n_val:], Y[n_train + n_val:]

        # Нормалізація
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train_scaled = x_scaler.fit_transform(X_train)
        Y_train_scaled = y_scaler.fit_transform(Y_train)
        X_val_scaled = x_scaler.transform(X_val)
        Y_val_scaled = y_scaler.transform(Y_val)
        X_test_scaled = x_scaler.transform(X_test)

        print(f"   Тренувальний набір: {X_train_scaled.shape[0]} зразків")
        print(f"   Валідаційний набір: {X_val_scaled.shape[0]} зразків")
        print(f"   Тестовий набір: {X_test_scaled.shape[0]} зразків")

        # Навчання моделей
        linear_model, kernel_model, training_metrics = self.train_models(
            X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, **kwargs
        )

        # Оцінка моделей
        evaluation_metrics = self.evaluate_models(
            linear_model, kernel_model, X_test_scaled, Y_test, y_scaler
        )

        # Аналіз нелінійності
        print("\n🔍 АНАЛІЗ НЕЛІНІЙНОСТІ ПРОЦЕСУ")
        print("-" * 40)
        nonlinearity_metrics = self.analyze_nonlinearity(df_sim, true_gen)
        for metric_name, value in nonlinearity_metrics.items():
            print(f"   📈 {metric_name}: {value:.3f}")

        # Збереження даних для візуалізації
        self._last_simulation_data = (
            Y_test, 
            evaluation_metrics['Y_pred_linear'], 
            evaluation_metrics['Y_pred_kernel'],
            evaluation_metrics, 
            nonlinearity_metrics, 
            df_sim
        )

        # Підсумкові результати
        improvement_mse = evaluation_metrics['improvement_mse']
        target_achieved = improvement_mse >= 15
        
        print("\n📊 ПОРІВНЯННЯ ТА АНАЛІЗ НЕЛІНІЙНОСТІ")
        print("-" * 50)
        print("🎯 КЛЮЧОВІ РЕЗУЛЬТАТИ ДЛЯ ДИСЕРТАЦІЇ:")
        print(f"   💡 Покращення MSE: {improvement_mse:.1f}%")
        print(f"   💡 Покращення RMSE Fe: {evaluation_metrics['improvement_fe']:.1f}%")
        print(f"   💡 Покращення RMSE Mass: {evaluation_metrics['improvement_mass']:.1f}%")
        print(f"   {'✅' if target_achieved else '❌'} Цільовий діапазон "
              f"{'ДОСЯГНУТО' if target_achieved else 'НЕ досягнуто'}")

        # Збереження результатів
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'simulation_params': simulation_params,
            'data_info': {
                'samples_total': len(df_sim),
                'samples_train': X_train_scaled.shape[0],
                'samples_val': X_val_scaled.shape[0],
                'samples_test': X_test_scaled.shape[0],
                'lag_used': simulation_params['lag'],
                'features': X_train_scaled.shape[1]
            },
            'linear_model': {
                'type': 'Linear (ARX)',
                'mse': evaluation_metrics['linear_mse'],
                'rmse_fe': evaluation_metrics['linear_rmse_fe'],
                'rmse_mass': evaluation_metrics['linear_rmse_mass'],
                'train_time': training_metrics['linear_train_time']
            },
            'kernel_model': {
                'type': 'Kernel Ridge Regression (RBF)',
                'mse': evaluation_metrics['kernel_mse'],
                'rmse_fe': evaluation_metrics['kernel_rmse_fe'],
                'rmse_mass': evaluation_metrics['kernel_rmse_mass'],
                'train_time': training_metrics['kernel_train_time']
            },
            'performance_comparison': {
                'mse_improvement_percent': improvement_mse,
                'rmse_fe_improvement_percent': evaluation_metrics['improvement_fe'],
                'rmse_mass_improvement_percent': evaluation_metrics['improvement_mass'],
                'target_achieved': target_achieved,
                'target_range': (15, 20)
            },
            'nonlinearity_analysis': nonlinearity_metrics
        }

        self.results = results
        return results
       
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Збереження результатів у JSON файл.
        
        Args:
            filename: Ім'я файлу. Якщо None, генерується автоматично
            
        Returns:
            str: Шлях до збереженого файлу
        """
        if not self.results:
            raise ValueError("Немає результатів для збереження. Спочатку виконайте run_comparison()")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'comparison_results_{timestamp}.json'
        
        filepath = self.dirs['data'] / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
            
        print(f"📁 Результати збережено у {filepath}")
        return str(filepath)

    def create_comparison_visualizations(self, Y_test: np.ndarray, Y_pred_linear: np.ndarray, 
                                       Y_pred_kernel: np.ndarray, evaluation_metrics: dict,
                                       nonlinearity_metrics: dict, df_sim: pd.DataFrame) -> dict:
        """
        Створення комплексних візуалізацій для порівняння моделей.
        
        Args:
            Y_test: Реальні тестові значення
            Y_pred_linear: Прогнози лінійної моделі
            Y_pred_kernel: Прогнози ядерної моделі
            evaluation_metrics: Метрики оцінки моделей
            nonlinearity_metrics: Метрики нелінійності
            df_sim: Повні симуляційні дані
            
        Returns:
            dict: Словник створених фігур
        """
        import matplotlib.pyplot as plt
        
        # Налаштування стилю
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 10
        
        figures = {}
        
        # ОСНОВНА ФІГУРА: ПОРІВНЯННЯ МОДЕЛЕЙ
        fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig1.suptitle('Порівняння лінійної та ядерної моделей для дисертації', fontsize=16, fontweight='bold')
        
        # 1.1 Scatter plot для Fe концентрації
        ax = axes[0, 0]
        ax.scatter(Y_test[:, 0], Y_pred_linear[:, 0], alpha=0.6, s=20, color='red', label='Лінійна модель')
        ax.scatter(Y_test[:, 0], Y_pred_kernel[:, 0], alpha=0.6, s=20, color='green', label='Ядерна модель')
        
        # Ідеальна лінія
        min_val, max_val = Y_test[:, 0].min(), Y_test[:, 0].max()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Ідеальна лінія')
        
        ax.set_xlabel('Реальна концентрація Fe (%)')
        ax.set_ylabel('Прогнозована концентрація Fe (%)')
        ax.set_title('Прогнозування концентрації Fe')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Додавання R² на графік
        r2_linear = 1 - np.sum((Y_test[:, 0] - Y_pred_linear[:, 0])**2) / np.sum((Y_test[:, 0] - np.mean(Y_test[:, 0]))**2)
        r2_kernel = 1 - np.sum((Y_test[:, 0] - Y_pred_kernel[:, 0])**2) / np.sum((Y_test[:, 0] - np.mean(Y_test[:, 0]))**2)
        ax.text(0.05, 0.95, f'R² лінійна: {r2_linear:.3f}\nR² ядерна: {r2_kernel:.3f}', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 1.2 Scatter plot для масового потоку
        ax = axes[0, 1]
        ax.scatter(Y_test[:, 1], Y_pred_linear[:, 1], alpha=0.6, s=20, color='red', label='Лінійна модель')
        ax.scatter(Y_test[:, 1], Y_pred_kernel[:, 1], alpha=0.6, s=20, color='green', label='Ядерна модель')
        
        min_val, max_val = Y_test[:, 1].min(), Y_test[:, 1].max()
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Ідеальна лінія')
        
        ax.set_xlabel('Реальний масовий потік (т/год)')
        ax.set_ylabel('Прогнозований масовий потік (т/год)')
        ax.set_title('Прогнозування масового потоку')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 1.3 Порівняння MSE
        ax = axes[0, 2]
        models = ['Лінійна\n(ARX)', 'Ядерна\n(KRR)']
        mse_values = [evaluation_metrics['linear_mse'], evaluation_metrics['kernel_mse']]
        colors = ['red', 'green']
        
        bars = ax.bar(models, mse_values, color=colors, alpha=0.7, width=0.6)
        ax.set_ylabel('MSE')
        ax.set_title(f'Порівняння MSE\n(покращення: {evaluation_metrics["improvement_mse"]:.1f}%)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Додавання значень на стовпці
        for bar, value in zip(bars, mse_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Додавання стрілки покращення
        improvement = evaluation_metrics['improvement_mse']
        if improvement > 0:
            ax.annotate('', xy=(1, evaluation_metrics['kernel_mse']), xytext=(0, evaluation_metrics['linear_mse']),
                       arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
            ax.text(0.5, (evaluation_metrics['linear_mse'] + evaluation_metrics['kernel_mse'])/2, 
                   f'-{improvement:.1f}%', ha='center', va='center', color='blue', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 1.4 Часовий ряд помилок для Fe
        ax = axes[1, 0]
        time_steps = range(len(Y_test))
        error_linear_fe = Y_test[:, 0] - Y_pred_linear[:, 0]
        error_kernel_fe = Y_test[:, 0] - Y_pred_kernel[:, 0]
        
        ax.plot(time_steps, error_linear_fe, color='red', alpha=0.7, linewidth=1, label='Помилка лінійної')
        ax.plot(time_steps, error_kernel_fe, color='green', alpha=0.7, linewidth=1, label='Помилка ядерної')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Крок тестування')
        ax.set_ylabel('Помилка прогнозу Fe (%)')
        ax.set_title('Динаміка помилок прогнозування Fe')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 1.5 Розподіл помилок
        ax = axes[1, 1]
        ax.hist(error_linear_fe, bins=30, alpha=0.6, color='red', label='Лінійна модель', density=True)
        ax.hist(error_kernel_fe, bins=30, alpha=0.6, color='green', label='Ядерна модель', density=True)
        
        ax.set_xlabel('Помилка прогнозу Fe (%)')
        ax.set_ylabel('Щільність розподілу')
        ax.set_title('Розподіл помилок прогнозування')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Статистики помилок
        ax.text(0.02, 0.98, 
               f'Лінійна:\nСТД: {np.std(error_linear_fe):.3f}\nСер.: {np.mean(error_linear_fe):.3f}\n\n'
               f'Ядерна:\nСТД: {np.std(error_kernel_fe):.3f}\nСер.: {np.mean(error_kernel_fe):.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 1.6 Метрики нелінійності
        ax = axes[1, 2]
        if nonlinearity_metrics:
            metric_names = list(nonlinearity_metrics.keys())
            metric_values = list(nonlinearity_metrics.values())
            
            # Скорочення назв для кращого відображення
            short_names = []
            for name in metric_names:
                if 'gradient' in name:
                    short_names.append('Градієнт\nваріації')
                elif 'correlation' in name:
                    short_names.append('Кореляційна\nнелінійність')
                elif 'complexity' in name:
                    short_names.append('Складність\nкерування')
                elif 'kurtosis' in name:
                    short_names.append('Куртозис\nрозподілу')
                elif 'skewness' in name:
                    short_names.append('Асиметрія\nрозподілу')
                else:
                    short_names.append(name[:10] + '...' if len(name) > 10 else name)
            
            bars = ax.bar(range(len(metric_values)), metric_values, color='orange', alpha=0.7)
            ax.set_xticks(range(len(short_names)))
            ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Значення метрики')
            ax.set_title('Характеристики нелінійності процесу')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Додавання значень на стовпці
            for i, (bar, value) in enumerate(zip(bars, metric_values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.02,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Метрики нелінійності\nне доступні', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Характеристики нелінійності процесу')
    
        main_plot_path = self.dirs['visualizations'] / 'model_comparison_main.png'
        plt.tight_layout()
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        figures['main_comparison'] = fig1
        
        # ДОДАТКОВА ФІГУРА: ДЕТАЛЬНИЙ АНАЛІЗ
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig2.suptitle('Детальний аналіз продуктивності моделей', fontsize=14, fontweight='bold')
        
        # 2.1 Box plot помилок
        ax = axes[0, 0]
        error_data = [error_linear_fe, error_kernel_fe]
        bp = ax.boxplot(error_data, labels=['Лінійна', 'Ядерна'], patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][1].set_facecolor('green')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_alpha(0.6)
        
        ax.set_ylabel('Помилка прогнозу Fe (%)')
        ax.set_title('Розподіл помилок (квартилі)')
        ax.grid(True, alpha=0.3)
        
        # 2.2 Кумулятивний розподіл помилок
        ax = axes[0, 1]
        sorted_linear = np.sort(np.abs(error_linear_fe))
        sorted_kernel = np.sort(np.abs(error_kernel_fe))
        
        y_linear = np.arange(1, len(sorted_linear) + 1) / len(sorted_linear)
        y_kernel = np.arange(1, len(sorted_kernel) + 1) / len(sorted_kernel)
        
        ax.plot(sorted_linear, y_linear, color='red', linewidth=2, label='Лінійна модель')
        ax.plot(sorted_kernel, y_kernel, color='green', linewidth=2, label='Ядерна модель')
        
        ax.set_xlabel('Абсолютна помилка Fe (%)')
        ax.set_ylabel('Кумулятивна ймовірність')
        ax.set_title('Кумулятивний розподіл помилок')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2.3 Кореляція залишків
        ax = axes[1, 0]
        ax.scatter(error_linear_fe, error_kernel_fe, alpha=0.6, s=20, color='purple')
        
        # Лінія кореляції
        correlation = np.corrcoef(error_linear_fe, error_kernel_fe)[0, 1]
        ax.plot([error_linear_fe.min(), error_linear_fe.max()], 
               [error_kernel_fe.min(), error_kernel_fe.max()], 'r--', alpha=0.8)
        
        ax.set_xlabel('Помилка лінійної моделі (%)')
        ax.set_ylabel('Помилка ядерної моделі (%)')
        ax.set_title(f'Кореляція помилок (r = {correlation:.3f})')
        ax.grid(True, alpha=0.3)
        
        # 2.4 Покращення за квартилями
        ax = axes[1, 1]
        quartiles = [25, 50, 75, 90, 95]
        linear_percentiles = np.percentile(np.abs(error_linear_fe), quartiles)
        kernel_percentiles = np.percentile(np.abs(error_kernel_fe), quartiles)
        improvements = ((linear_percentiles - kernel_percentiles) / linear_percentiles) * 100
        
        bars = ax.bar(range(len(quartiles)), improvements, color='blue', alpha=0.7)
        ax.set_xticks(range(len(quartiles)))
        ax.set_xticklabels([f'{q}%' for q in quartiles])
        ax.set_ylabel('Покращення (%)')
        ax.set_xlabel('Квартиль помилок')
        ax.set_title('Покращення за квартилями помилок')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Додавання значень на стовпці
        for bar, value in zip(bars, improvements):
            color = 'green' if value > 0 else 'red'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(improvements)*0.02,
                   f'{value:.1f}%', ha='center', va='bottom', color=color, fontweight='bold')
        
        detailed_plot_path = self.dirs['visualizations'] / 'model_comparison_detailed.png'
        plt.tight_layout()
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        figures['detailed_analysis'] = fig2
        
        print("📊 Створено комплексні візуалізації:")
        print("   📈 dissertation_model_comparison.png - основне порівняння")
        print("   📊 dissertation_detailed_analysis.png - детальний аналіз")
        
        self.figures = figures
        return figures

    def create_dissertation_summary_visualization(self, Y_test: np.ndarray, Y_pred_linear: np.ndarray, 
                                                 Y_pred_kernel: np.ndarray, evaluation_metrics: dict) -> str:
        """
        Створення компактної візуалізації для дисертації.
        Два графіки: scatter plot порівняння та bar chart покращення.
        
        Args:
            Y_test: Реальні тестові значення
            Y_pred_linear: Прогнози лінійної моделі
            Y_pred_kernel: Прогнози ядерної моделі
            evaluation_metrics: Метрики оцінки моделей
            
        Returns:
            str: Шлях до збереженого файлу
        """
        import matplotlib.pyplot as plt
        
        # Налаштування для дисертації
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Порівняння лінійної та ядерної моделей', fontsize=14, fontweight='bold')
        
        # ГРАФІК 1: Scatter plot для концентрації Fe (найважливіший показник)
        ax1.scatter(Y_test[:, 0], Y_pred_linear[:, 0], alpha=0.7, s=30, 
                   color='red', label='Лінійна модель', marker='o')
        ax1.scatter(Y_test[:, 0], Y_pred_kernel[:, 0], alpha=0.7, s=30, 
                   color='green', label='Ядерна модель', marker='s')
        
        # Ідеальна лінія
        min_val, max_val = Y_test[:, 0].min(), Y_test[:, 0].max()
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', 
                 alpha=0.8, linewidth=2, label='Ідеальна лінія')
        
        ax1.set_xlabel('Реальна концентрація Fe (%)')
        ax1.set_ylabel('Прогнозована концентрація Fe (%)')
        ax1.set_title('Точність прогнозування')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Додавання R² в нижній правий кут
        r2_linear = 1 - np.sum((Y_test[:, 0] - Y_pred_linear[:, 0])**2) / np.sum((Y_test[:, 0] - np.mean(Y_test[:, 0]))**2)
        r2_kernel = 1 - np.sum((Y_test[:, 0] - Y_pred_kernel[:, 0])**2) / np.sum((Y_test[:, 0] - np.mean(Y_test[:, 0]))**2)
        
        ax1.text(0.98, 0.02, f'R² лінійна: {r2_linear:.3f}\nR² ядерна: {r2_kernel:.3f}', 
                 transform=ax1.transAxes, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # ГРАФІК 2: Bar chart покращення метрик
        metrics = ['MSE', 'RMSE Fe', 'RMSE Mass']
        improvements = [
            evaluation_metrics['improvement_mse'],
            evaluation_metrics['improvement_fe'], 
            evaluation_metrics['improvement_mass']
        ]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']  # Професійна палітра
        bars = ax2.bar(metrics, improvements, color=colors, alpha=0.8, width=0.6)
        
        ax2.set_ylabel('Покращення (%)')
        ax2.set_title('Переваги ядерної моделі')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, max(improvements) * 1.2)
        
        # Додавання значень на стовпці
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + max(improvements)*0.02,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Лінія цільового рівня покращення (15%)
        ax2.axhline(y=15, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax2.text(len(metrics)-1, 15.5, 'Цільовий рівень (15%)', 
                 ha='right', va='bottom', color='red', fontsize=10)
        
        plt.tight_layout()
        
        # Збереження
        summary_plot_path = self.dirs['visualizations'] / 'dissertation_summary.png'
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Компактну візуалізацію збережено у {summary_plot_path}")
        return str(summary_plot_path)
    
    def create_performance_table_visualization(self, evaluation_metrics: dict) -> str:
        """
        Створення табличної візуалізації метрик у вигляді графіку.
        Альтернатива для презентацій.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Дані для таблиці
        metrics_data = [
            ['Метрика', 'Лінійна модель', 'Ядерна модель', 'Покращення'],
            ['MSE', f"{evaluation_metrics['linear_mse']:.4f}", 
             f"{evaluation_metrics['kernel_mse']:.4f}", 
             f"{evaluation_metrics['improvement_mse']:.1f}%"],
            ['RMSE Fe (%)', f"{evaluation_metrics['linear_rmse_fe']:.3f}", 
             f"{evaluation_metrics['kernel_rmse_fe']:.3f}", 
             f"{evaluation_metrics['improvement_fe']:.1f}%"],
            ['RMSE Mass (т/год)', f"{evaluation_metrics['linear_rmse_mass']:.3f}", 
             f"{evaluation_metrics['kernel_rmse_mass']:.3f}", 
             f"{evaluation_metrics['improvement_mass']:.1f}%"]
        ]
        
        # Створення таблиці
        table = ax.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                        cellLoc='center', loc='center')
        
        # Стилізація таблиці
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Кольорове кодування заголовків
        for i in range(len(metrics_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Кольорове кодування стовпця покращення
        for i in range(1, len(metrics_data)):
            improvement_val = float(metrics_data[i][3].replace('%', ''))
            if improvement_val >= 15:
                table[(i, 3)].set_facecolor('#D5E8D4')  # Зелений для досягнення цілі
            else:
                table[(i, 3)].set_facecolor('#FFF2CC')  # Жовтий для недосягнення
            table[(i, 3)].set_text_props(weight='bold')
        
        ax.set_title('Порівняння продуктивності моделей', fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Збереження
        table_plot_path = self.dirs['visualizations'] / 'performance_table.png'
        plt.savefig(table_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Табличну візуалізацію збережено у {table_plot_path}")
        return str(table_plot_path)
    
    def generate_report_summary(self) -> str:
        """
        Генерація текстового звіту для дисертації.
        ВИПРАВЛЕНО: правильне збереження у папку reports
        """
        if not self.results:
            raise ValueError("Немає результатів для звіту. Спочатку виконайте run_comparison()")
        
        perf = self.results['performance_comparison']
        linear = self.results['linear_model']
        kernel = self.results['kernel_model']
        data_info = self.results['data_info']
        
        report = f"""
    ЗВІТ ПРО ПОРІВНЯННЯ МОДЕЛЕЙ ДЛЯ ДИСЕРТАЦІЇ
    {'='*55}
    
    ДАНІ ПРО ЕКСПЕРИМЕНТ:
        Загальна кількість зразків: {data_info['samples_total']:,}
        Тренувальні зразки: {data_info['samples_train']:,}
        Валідаційні зразки: {data_info['samples_val']:,}
        Тестові зразки: {data_info['samples_test']:,}
        Кількість лагів: {data_info['lag_used']}
        Кількість ознак: {data_info['features']}
    
    РЕЗУЛЬТАТИ ЛІНІЙНОЇ МОДЕЛІ (ARX):
        MSE: {linear['mse']:.6f}
        RMSE Fe: {linear['rmse_fe']:.3f}%
        RMSE Mass: {linear['rmse_mass']:.3f} т/год
        Час навчання: {linear['train_time']:.3f} сек
    
    РЕЗУЛЬТАТИ ЯДЕРНОЇ МОДЕЛІ (KRR):
        MSE: {kernel['mse']:.6f}
        RMSE Fe: {kernel['rmse_fe']:.3f}%
        RMSE Mass: {kernel['rmse_mass']:.3f} т/год
        Час навчання: {kernel['train_time']:.3f} сек
    
    ПОРІВНЯННЯ ПРОДУКТИВНОСТІ:
        Покращення MSE: {perf['mse_improvement_percent']:.1f}%
        Покращення RMSE Fe: {perf['rmse_fe_improvement_percent']:.1f}%
        Покращення RMSE Mass: {perf['rmse_mass_improvement_percent']:.1f}%
        
    ВИСНОВОК:
        Цільовий діапазон (15-20%): {'✅ ДОСЯГНУТО' if perf['target_achieved'] else '❌ НЕ ДОСЯГНУТО'}
        
    НЕЛІНІЙНІСТЬ ПРОЦЕСУ:"""
        
        if 'nonlinearity_analysis' in self.results:
            for metric, value in self.results['nonlinearity_analysis'].items():
                report += f"\n    {metric}: {value:.3f}"
        
        report += f"""
    
    ДАТА ЕКСПЕРИМЕНТУ: {self.results['timestamp']}
    {'='*55}
    """
        
        # ВИПРАВЛЕННЯ: збереження у відповідну папку
        report_file_path = self.dirs['reports'] / 'analysis_report.txt'
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📝 Звіт збережено у {report_file_path}")
        return str(report_file_path)  
    
    def export_metrics_for_latex(self) -> str:
        """
        Експорт метрик у форматі LaTeX таблиці.
        ВИПРАВЛЕНО: правильне збереження у папку latex
        """
        if not self.results:
            raise ValueError("Немає результатів для експорту. Спочатку виконайте run_comparison()")
        
        linear = self.results['linear_model']
        kernel = self.results['kernel_model']
        perf = self.results['performance_comparison']
        
        latex_table = f"""
    \\begin{{table}}[h]
    \\centering
    \\caption{{Порівняння продуктивності лінійної та ядерної моделей}}
    \\label{{tab:model_comparison}}
    \\begin{{tabular}}{{|l|c|c|c|}}
    \\hline
    \\textbf{{Метрика}} & \\textbf{{Лінійна (ARX)}} & \\textbf{{Ядерна (KRR)}} & \\textbf{{Покращення, \\%}} \\\\
    \\hline
    MSE & {linear['mse']:.6f} & {kernel['mse']:.6f} & {perf['mse_improvement_percent']:.1f} \\\\
    \\hline
    RMSE Fe, \\% & {linear['rmse_fe']:.3f} & {kernel['rmse_fe']:.3f} & {perf['rmse_fe_improvement_percent']:.1f} \\\\
    \\hline
    RMSE Mass, т/год & {linear['rmse_mass']:.3f} & {kernel['rmse_mass']:.3f} & {perf['rmse_mass_improvement_percent']:.1f} \\\\
    \\hline
    Час навчання, сек & {linear['train_time']:.3f} & {kernel['train_time']:.3f} & - \\\\
    \\hline
    \\end{{tabular}}
    \\end{{table}}
    """
        
        # ВИПРАВЛЕННЯ: збереження у відповідну папку
        latex_file_path = self.dirs['latex'] / 'model_comparison_table.tex'
        with open(latex_file_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        print(f"📄 LaTeX таблицю збережено у {latex_file_path}")
        return str(latex_file_path)  
    
    def run_full_analysis_with_visualizations(self, **kwargs) -> dict:
        """
        Повний аналіз з візуалізаціями для дисертації.
        ОНОВЛЕНО: додано компактну візуалізацію
        """
        # Виконання основного порівняння
        results = self.run_comparison(**kwargs)
        
        # Отримання даних для візуалізації з останнього запуску
        if hasattr(self, '_last_simulation_data'):
            Y_test, Y_pred_linear, Y_pred_kernel, evaluation_metrics, nonlinearity_metrics, df_sim = self._last_simulation_data
            
            # Створення візуалізацій
            print("\nГенерація візуалізацій...")
            
            # Детальні візуалізації
            figures = self.create_comparison_visualizations(
                Y_test, Y_pred_linear, Y_pred_kernel, 
                evaluation_metrics, nonlinearity_metrics, df_sim
            )
            
            # НОВА: Компактна візуалізація для дисертації
            summary_viz_path = self.create_dissertation_summary_visualization(
                Y_test, Y_pred_linear, Y_pred_kernel, evaluation_metrics
            )
            
            # НОВА: Табличний графік (опціонально)
            table_viz_path = self.create_performance_table_visualization(evaluation_metrics)
            
            # Генерація звіту
            print("\nГенерація звіту...")
            report_path = self.generate_report_summary()
            
            # Експорт LaTeX таблиці
            latex_path = self.export_metrics_for_latex()
            
            # Збереження результатів
            results_path = self.save_results()
            
            print("\nПовний аналіз завершено")
            print(f"Всі файли збережено у директорії: {self.output_dir}")
            print("Структура файлів:")
            print(f"   Візуалізації:")
            print(f"     - {summary_viz_path} (для дисертації)")
            print(f"     - {table_viz_path} (альтернативна)")
            print(f"   Детальні графіки та звіти у відповідних папках")
            
            return results
        else:
            print("Дані симуляції недоступні для візуалізації")
            return results
        
    def add_custom_metrics(self, custom_metrics: dict) -> None:
        """
        Додавання користувацьких метрик до результатів.
        
        Args:
            custom_metrics: Словник з додатковими метриками
        """
        if not self.results:
            raise ValueError("Немає базових результатів. Спочатку виконайте run_comparison()")
        
        if 'custom_metrics' not in self.results:
            self.results['custom_metrics'] = {}
        
        self.results['custom_metrics'].update(custom_metrics)
        print(f"✅ Додано {len(custom_metrics)} користувацьких метрик")
    
    def get_performance_summary(self) -> dict:
        """
        Отримання короткого підсумку продуктивності.
        
        Returns:
            dict: Основні показники продуктивності
        """
        if not self.results:
            raise ValueError("Немає результатів для підсумку. Спочатку виконайте run_comparison()")
        
        perf = self.results['performance_comparison']
        
        summary = {
            'mse_improvement': perf['mse_improvement_percent'],
            'target_achieved': perf['target_achieved'],
            'best_metric': max(perf['mse_improvement_percent'], 
                              perf['rmse_fe_improvement_percent'],
                              perf['rmse_mass_improvement_percent']),
            'overall_grade': 'Excellent' if perf['mse_improvement_percent'] > 20 else
                            'Good' if perf['mse_improvement_percent'] > 15 else
                            'Moderate' if perf['mse_improvement_percent'] > 10 else
                            'Poor'
        }
        
        return summary
    
    def compare_multiple_configurations(self, config_list: list) -> pd.DataFrame:
        """
        Порівняння кількох конфігурацій параметрів.
        ВИПРАВЛЕНО: правильне збереження у папку comparisons
        """
        comparison_results = []
        
        for i, config in enumerate(config_list):
            print(f"\n🔄 Тестування конфігурації {i+1}/{len(config_list)}")
            print("-" * 50)
            
            try:
                results = self.run_comparison(**config)
                
                row = {
                    'config_id': i+1,
                    'linear_mse': results['linear_model']['mse'],
                    'kernel_mse': results['kernel_model']['mse'],
                    'mse_improvement': results['performance_comparison']['mse_improvement_percent'],
                    'linear_rmse_fe': results['linear_model']['rmse_fe'],
                    'kernel_rmse_fe': results['kernel_model']['rmse_fe'],
                    'fe_improvement': results['performance_comparison']['rmse_fe_improvement_percent'],
                    'target_achieved': results['performance_comparison']['target_achieved'],
                    'linear_train_time': results['linear_model']['train_time'],
                    'kernel_train_time': results['kernel_model']['train_time']
                }
                
                # Додавання параметрів конфігурації
                for key, value in config.items():
                    if key not in ['time_constants_s', 'dead_times_s', 'nonlinear_config']:
                        row[f'param_{key}'] = value
                        
                comparison_results.append(row)
                
            except Exception as e:
                print(f"❌ Помилка в конфігурації {i+1}: {e}")
                continue
        
        df_comparison = pd.DataFrame(comparison_results)
        
        # ВИПРАВЛЕННЯ: збереження у відповідну папку
        comparison_file_path = self.dirs['comparisons'] / 'configurations_comparison.csv'
        df_comparison.to_csv(comparison_file_path, index=False)
        print(f"\n📊 Порівняння {len(comparison_results)} конфігурацій збережено у {comparison_file_path}")
        
        return df_comparison
    
def basic_comparison_example():
    """Базовий приклад порівняння моделей"""
    print("=== БАЗОВИЙ ПРИКЛАД ПОРІВНЯННЯ МОДЕЛЕЙ ===\n")
    
    # Ініціалізація сервісу
    service = ModelComparisonService()
    
    # Виконання базового порівняння з стандартними параметрами
    results = service.run_comparison(
        N_data=5000,           # Кількість точок даних
        lag=2,                 # Кількість лагів  
        anomaly_severity='mild', # Рівень аномалій
        use_anomalies=True,     # Включити аномалії
        seed=42                 # Для відтворюваності
    )
    
    # Отримання короткого підсумку
    summary = service.get_performance_summary()
    print(f"Покращення MSE: {summary['mse_improvement']:.1f}%")
    print(f"Цільовий рівень досягнуто: {summary['target_achieved']}")
    print(f"Загальна оцінка: {summary['overall_grade']}")
    
    # Збереження результатів
    filename = service.save_results()
    print(f"Результати збережено у {filename}")
    
    return results

def full_analysis_example():
    """Повний приклад аналізу з візуалізаціями"""
    print("\n=== ПОВНИЙ АНАЛІЗ З ВІЗУАЛІЗАЦІЯМИ ===\n")
    
    # Ініціалізація сервісу
    service = ModelComparisonService()
    
    # Повний аналіз з візуалізаціями
    results = service.run_full_analysis_with_visualizations(
        N_data=7000,
        lag=3,
        anomaly_severity='medium',
        anomaly_in_train=False,  # Аномалії тільки в val/test
        find_optimal_params=True, # Оптимізація гіперпараметрів
        n_iter_search=30,        # Більше ітерацій пошуку
        seed=123
    )
    
    # Додавання користувацьких метрик
    custom_metrics = {
        'data_quality_score': 0.85,
        'simulation_realism': 0.92,
        'computational_efficiency': 0.78
    }
    service.add_custom_metrics(custom_metrics)
    
    # Оновлене збереження з користувацькими метриками
    service.save_results('full_analysis_results.json')
    
    return results

def multiple_configurations_example():
    """Приклад порівняння кількох конфігурацій"""
    print("\n=== ПОРІВНЯННЯ КІЛЬКОХ КОНФІГУРАЦІЙ ===\n")
    
    service = ModelComparisonService()
    
    # Визначення конфігурацій для тестування
    configurations = [
        {
            'N_data': 5000,
            'lag': 2,
            'anomaly_severity': 'mild',
            'use_anomalies': True,
            'seed': 42
        },
        {
            'N_data': 5000,
            'lag': 3,
            'anomaly_severity': 'mild',
            'use_anomalies': True,
            'seed': 42
        },
        {
            'N_data': 5000,
            'lag': 2,
            'anomaly_severity': 'medium',
            'use_anomalies': True,
            'seed': 42
        },
        {
            'N_data': 7000,
            'lag': 2,
            'anomaly_severity': 'mild',
            'use_anomalies': True,
            'seed': 42
        },
        {
            'N_data': 5000,
            'lag': 2,
            'anomaly_severity': 'mild',
            'use_anomalies': False,  # Без аномалій
            'seed': 42
        }
    ]
    
    # Виконання порівняння
    comparison_df = service.compare_multiple_configurations(configurations)
    
    # Аналіз результатів
    print("\nТОП-3 конфігурації за покращенням MSE:")
    top_configs = comparison_df.nlargest(3, 'mse_improvement')
    for idx, row in top_configs.iterrows():
        print(f"Конфігурація {row['config_id']}: "
              f"MSE покращення = {row['mse_improvement']:.1f}%, "
              f"Lag = {row.get('param_lag', 'N/A')}, "
              f"Аномалії = {row.get('param_anomaly_severity', 'N/A')}")
    
    # Статистичний аналіз
    print(f"\nСтатистика покращень MSE:")
    print(f"Середнє: {comparison_df['mse_improvement'].mean():.1f}%")
    print(f"Медіана: {comparison_df['mse_improvement'].median():.1f}%")
    print(f"Мін: {comparison_df['mse_improvement'].min():.1f}%")
    print(f"Макс: {comparison_df['mse_improvement'].max():.1f}%")
    
    return comparison_df

def custom_parameters_example():
    """Приклад з користувацькими параметрами симуляції"""
    print("\n=== КОРИСТУВАЦЬКІ ПАРАМЕТРИ СИМУЛЯЦІЇ ===\n")
    
    service = ModelComparisonService(output_dir)
    
    # Спеціальні параметри для специфічного дослідження
    results = service.run_comparison(
        N_data=8000,
        lag=4,                    # Більша історія
        train_size=0.7,          # Менше тренувальних даних
        val_size=0.2,            # Більше валідаційних
        test_size=0.1,           # Менше тестових
        anomaly_severity='strong', # Сильні аномалії
        anomaly_in_train=True,    # Аномалії і в тренувальних даних
        noise_level='low',        # Додатковий шум
        find_optimal_params=True,
        n_iter_search=50,         # Ретельний пошук параметрів
        seed=999
    )
    
    # Генерація LaTeX таблиці
    latex_table = service.export_metrics_for_latex()
    print("LaTeX таблиця:")
    print(latex_table)
    
    return results

def dissertation_ready_example():
    """Приклад, готовий для використання в дисертації"""
    print("\n=== ПРИКЛАД ДЛЯ ДИСЕРТАЦІЇ ===\n")
    
    # Налаштування для максимальної якості результатів
    service = ModelComparisonService()
    
    # Оптимальні параметри для дисертаційного дослідження
    results = service.run_full_analysis_with_visualizations(
        N_data=10000,            # Великий набір даних
        lag=3,                   # Оптимальна глибина історії
        train_size=0.8,          # Стандартний розподіл
        val_size=0.1,
        test_size=0.1,
        anomaly_severity='mild',  # Реалістичні аномалії
        anomaly_in_train=False,   # Чисті тренувальні дані
        use_anomalies=True,
        find_optimal_params=True,
        n_iter_search=50,        # Максимальна якість оптимізації
        seed=42                   # Відтворюваність
    )
    
    # Детальний звіт
    report = service.generate_report_summary()
    
    # Збереження всіх результатів
    with open('dissertation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    service.save_results('dissertation_results.json')
    
    print("✅ ГОТОВО ДЛЯ ДИСЕРТАЦІЇ:")
    print("📊 dissertation_model_comparison.png - основні графіки")
    print("📈 dissertation_detailed_analysis.png - детальний аналіз") 
    print("📄 model_comparison_table.tex - таблиця для LaTeX")
    print("📝 dissertation_report.txt - текстовий звіт")
    print("📁 dissertation_results.json - повні результати")
    
    return results

def main():
    """Головна функція з демонстрацією всіх можливостей"""
    print("🎓 ДЕМОНСТРАЦІЯ ModelComparisonService ДЛЯ ДИСЕРТАЦІЇ")
    print("=" * 60)
    
    try:
        # Базовий приклад
        # basic_results = basic_comparison_example()
        
        # Повний аналіз
        full_results = full_analysis_example()
        
        # Порівняння конфігурацій
        # comparison_df = multiple_configurations_example()
        
        # Користувацькі параметри
        # custom_results = custom_parameters_example()
        
        # Фінальний приклад для дисертації
        # dissertation_results = dissertation_ready_example()
        # 
        print(f"\n🎉 ВСІ ПРИКЛАДИ ВИКОНАНО УСПІШНО!")
        print(f"📊 Кращий результат покращення MSE: "
              f"{full_results['performance_comparison']['mse_improvement_percent']:.1f}%")
        
    except Exception as e:
        print(f"❌ Помилка під час виконання: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     # Ініціалізація сервісу
#     service = ModelComparisonService()
    
#     results = service.run_full_analysis_with_visualizations(
#         N_data=7000,
#         lag=2,
#         use_anomalies=True,
#         anomaly_severity='mild',
#         find_optimal_params=True
#     )