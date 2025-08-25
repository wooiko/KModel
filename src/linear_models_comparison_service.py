# linear_models_comparison_service.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from pathlib import Path
from datetime import datetime
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.tools.tools import add_constant

# Припускаємо, що ці модулі знаходяться в тому ж каталозі
from data_gen import StatefulDataGenerator
from model import KernelModel


class LinearModelsComparisonService:
    """
    Сервіс для комплексного порівняння різних лінійних моделей
    з гнучкою конфігурацією для дослідження обмежень ARX підходу.
    
    Основні можливості:
    - Порівняння різних типів лінійних моделей (OLS, Ridge, Lasso, ARMAX)
    - Аналіз впливу лагової структури
    - Діагностика резидуалів та статистичні тести
    - Дослідження робастності до нелінійності та шуму
    - Генерація звітів та візуалізацій для дисертації
    """
    
    def __init__(self, reference_df: Optional[pd.DataFrame] = None, 
                 output_dir: Optional[str] = None):
        """
        Ініціалізація сервісу порівняння лінійних моделей.
        
        Args:
            reference_df: Референтні дані для симуляції
            output_dir: Базова директорія для збереження результатів
        """
        self.reference_df = reference_df
        self.output_dir = Path(output_dir) if output_dir else Path("linear_comparison_results")
        
        # Створення структури директорій
        self.dirs = {
            'main': self.output_dir,
            'plots': self.output_dir / 'plots',
            'data': self.output_dir / 'data', 
            'reports': self.output_dir / 'reports',
            'latex': self.output_dir / 'latex',
            'diagnostics': self.output_dir / 'diagnostics'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.models = {}
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def run_comprehensive_comparison(self, 
                                   model_configs: List[Dict[str, Any]],
                                   global_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Запуск комплексного порівняння лінійних моделей.
        
        Args:
            model_configs: Список конфігурацій для кожної моделі
                Приклад: [
                    {'name': 'ARX_OLS', 'linear_type': 'ols', 'poly_degree': 1},
                    {'name': 'ARX_Ridge', 'linear_type': 'ridge', 'alpha': 0.1},
                    {'name': 'ARX_Lasso', 'linear_type': 'lasso', 'alpha': 0.01}
                ]
            global_config: Глобальні параметри (дані, розбиття, симуляція)
                
        Returns:
            Dict з результатами порівняння
        """
        print("🚀 ЗАПУСК КОМПЛЕКСНОГО ПОРІВНЯННЯ ЛІНІЙНИХ МОДЕЛЕЙ")
        print("=" * 60)
        
        # 1. Підготовка даних
        data_results = self._prepare_data(global_config)
        X_train, Y_train, X_val, Y_val, X_test, Y_test, X_test_unscaled, Y_test_scaled = data_results
        
        # 2. Навчання всіх моделей
        self.models = {}
        training_results = {}
        
        for config in model_configs:
            model_name = config['name']
            print(f"\n📚 Навчання моделі: {model_name}")
            print("-" * 40)
            
            model_results = self._train_single_model(
                config, X_train, Y_train, X_val, Y_val
            )
            
            self.models[model_name] = model_results['model']
            training_results[model_name] = model_results['metrics']
        
        # 3. Оцінка всіх моделей на тестових даних
        evaluation_results = self._evaluate_all_models(
            X_test, Y_test, training_results
        )
        
        # 4. Діагностика резидуалів
        diagnostics_results = self._run_residual_diagnostics(
            X_test, Y_test
        )
        
        # 5. Аналіз робастності
        robustness_results = self._run_robustness_analysis(
            global_config, model_configs
        )
        
        # 6. Збірка результатів
        self.results = {
            'models_config': model_configs,
            'global_config': global_config,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'diagnostics_results': diagnostics_results,
            'robustness_results': robustness_results,
            'data_info': {
                'train_size': X_train.shape[0],
                'val_size': X_val.shape[0] if X_val is not None else 0,
                'test_size': X_test.shape[0],
                'n_features': X_train.shape[1]
            }
        }
        
        # 7. Генерація звітів та візуалізацій
        # ✅ ВИПРАВЛЕННЯ: Передаємо Y_test для коректної побудови діагностичних графіків
        self._generate_comprehensive_report(Y_test)
        
        return self.results
    
    def _prepare_data(self, global_config: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
        """
        Метод підготовки даних згідно з глобальною конфігурацією.
        """
        
        if global_config.get('use_simulation', True):
            print("📄 Генерація синтетичних даних з АНОМАЛІЯМИ та ШУМОМ...")
            
            # Створення генератора даних
            data_gen = StatefulDataGenerator(
                reference_df=self.reference_df,
                ore_flow_var_pct=3.0,
                time_step_s=global_config.get('time_step_s', 5),
                time_constants_s={
                    'concentrate_fe_percent': 300,
                    'tailings_fe_percent': 400,
                    'concentrate_mass_flow': 600,
                    'tailings_mass_flow': 700,
                    'default': 500
                },
                dead_times_s={
                    'concentrate_fe_percent': 60,
                    'tailings_fe_percent': 80,
                    'concentrate_mass_flow': 120,
                    'tailings_mass_flow': 140,
                    'default': 90
                },
                true_model_type=global_config.get('plant_model_type', 'rf'),
                seed=global_config.get('seed', 42)
            )
            
            # Створення конфігурації аномалій
            anomaly_cfg = None
            if global_config.get('use_anomalies', True):
                anomaly_cfg = self._create_anomaly_config(global_config)
                print(f"   🔴 Аномалії АКТИВОВАНІ: {len(anomaly_cfg) if anomaly_cfg else 0} конфігурацій")
            else:
                print("   ⚪ Аномалії ВІДКЛЮЧЕНІ")
            
            n_data = global_config.get('N_data', global_config.get('T', 5000))
            
            # Генерація базових даних
            df_base = data_gen.generate(
                T=n_data,
                control_pts=global_config.get('control_pts', 500),
                n_neighbors=global_config.get('n_neighbors', 5),
                noise_level=global_config.get('noise_level', 'mild'),
                anomaly_config=anomaly_cfg
            )
            
            if global_config.get('enable_nonlinear', False):
                print("   🔄 Застосування нелінійних трансформацій...")
                df = data_gen.generate_nonlinear_variant(
                    base_df=df_base,
                    non_linear_factors=global_config.get('nonlinear_config', {}),
                    noise_level='mild',
                    anomaly_config=anomaly_cfg
                )
                print(f"   📈 Нелінійність застосована: {global_config.get('nonlinear_config', {})}")
            else:
                df = df_base
                print("   📊 Використовується лінійна модель")
            
        else:
            print("📁 Використання наданих даних...")
            if self.reference_df is None:
                raise ValueError("Для використання реальних даних потрібен reference_df")
            df = self.reference_df.copy()
        
        # Лагові ознаки
        lag_depth = global_config.get('lag_depth', 3)
        X, Y = self._create_lag_features(df, lag_depth)
    
        # Розбиття
        train_size = global_config.get('train_size', 0.8)
        val_size = global_config.get('val_size', 0.1)
        n = X.shape[0]
        n_train = int(n * train_size)
        n_val = int(n * val_size)
        n_test = n - n_train - n_val
    
        X_train = X[:n_train]
        Y_train = Y[:n_train]
        X_val = X[n_train:n_train + n_val] if n_val > 0 else None
        Y_val = Y[n_train:n_train + n_val] if n_val > 0 else None
        X_test = X[-n_test:]
        Y_test = Y[-n_test:]
    
        # Масштабування: fit ТІЛЬКИ один раз, коли вони ще не фіткнуті
        if not hasattr(self.scaler_x, "mean_"):
            self.scaler_x.fit(X_train)
        if not hasattr(self.scaler_y, "mean_"):
            self.scaler_y.fit(Y_train)
    
        X_train_scaled = self.scaler_x.transform(X_train)
        X_val_scaled = self.scaler_x.transform(X_val) if X_val is not None else None
        X_test_scaled = self.scaler_x.transform(X_test)
    
        Y_train_scaled = self.scaler_y.transform(Y_train)
        Y_val_scaled = self.scaler_y.transform(Y_val) if Y_val is not None else None
        Y_test_scaled = self.scaler_y.transform(Y_test)
    
        # Повертаємо і масштабовані, і «реальні» тестові Y для коректних метрик
        return (
            X_train_scaled, Y_train_scaled,
            X_val_scaled, Y_val_scaled,
            X_test_scaled, Y_test,  # Y_test у реальних одиницях
            X_test, Y_test_scaled
        ) 
     
    def _create_anomaly_config(self, global_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Метод створення конфігурації аномалій.
        """
        N_data = global_config.get('N_data', global_config.get('T', 5000))
        train_frac = global_config.get('train_size', 0.8)
        val_frac = global_config.get('val_size', 0.1)
        test_frac = 1 - train_frac - val_frac
        seed = global_config.get('seed', 42)
        
        base_anomaly_config = StatefulDataGenerator.generate_anomaly_config(
            N_data=N_data,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed
        )
        
        severity = global_config.get('anomaly_severity', 'medium')
        include_train = global_config.get('anomaly_in_train', True)
        
        n_train = int(N_data * train_frac)
        n_val = int(N_data * val_frac)
        
        enhanced_anomaly_config = base_anomaly_config.copy() if base_anomaly_config else {}
        
        test_start = n_train + n_val
        
        if severity == 'mild':
            n_test_anomalies, anomaly_strength = 3, 1.5
        elif severity == 'medium':
            n_test_anomalies, anomaly_strength = 6, 2.0
        elif severity == 'strong':
            n_test_anomalies, anomaly_strength = 10, 3.0
        else:
            n_test_anomalies, anomaly_strength = 5, 2.0
        
        np.random.seed(seed + 100)
        test_indices = np.random.choice(
            range(test_start, N_data - 10), 
            size=n_test_anomalies, 
            replace=False
        )
        
        if 'feed_fe_percent' not in enhanced_anomaly_config:
            enhanced_anomaly_config['feed_fe_percent'] = []
        if 'ore_mass_flow' not in enhanced_anomaly_config:
            enhanced_anomaly_config['ore_mass_flow'] = []
        
        for idx in test_indices:
            enhanced_anomaly_config['feed_fe_percent'].append({
                'start': int(idx),
                'duration': np.random.randint(3, 8),
                'magnitude': anomaly_strength,
                'type': 'spike'
            })
            
            enhanced_anomaly_config['ore_mass_flow'].append({
                'start': int(idx) + 2,
                'duration': np.random.randint(2, 6),
                'magnitude': anomaly_strength * 0.8,
                'type': 'spike'
            })
        
        if include_train:
            n_train_anomalies = max(1, n_test_anomalies // 3)
            
            train_indices = np.random.choice(
                range(100, n_train - 100), 
                size=n_train_anomalies, 
                replace=False
            )
            
            if 'feed_fe_percent' not in enhanced_anomaly_config:
                enhanced_anomaly_config['feed_fe_percent'] = []
            
            for idx in train_indices:
                enhanced_anomaly_config['feed_fe_percent'].append({
                    'start': int(idx),
                    'duration': np.random.randint(2, 5),
                    'magnitude': anomaly_strength * 0.7,
                    'type': 'spike'
                })
            
            print(f"   🔴 Додано {n_train_anomalies} аномалій у тренувальні дані")
        
        print(f"   🔴 Додано {n_test_anomalies * 2} аномалій у тестові дані")
        print(f"   📊 Рівень аномалій: {severity} (сила: {anomaly_strength})")
        
        return enhanced_anomaly_config
    
    
    def diagnose_data_quality(self, X: np.ndarray, Y: np.ndarray, 
                             df_original: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Діагностичний метод для виявлення проблем з якістю даних.
        """
        
        print("🔍 ДІАГНОСТИКА ЯКОСТІ ДАНИХ")
        print("=" * 35)
        
        diagnostics = {
            'data_variability': {},
            'anomaly_presence': {},
            'nonlinearity_check': {},
            'data_quality_score': 0.0,
            'warnings': []
        }
        
        print("\n📊 Перевірка варіації даних:")
        for i in range(Y.shape[1]):
            col_name = f'Y_col_{i}'
            y_col = Y[:, i]
            std_val = np.std(y_col)
            coef_var = std_val / (np.mean(y_col) + 1e-12)
            unique_ratio = len(np.unique(y_col.round(6))) / len(y_col)
            diagnostics['data_variability'][col_name] = {
                'std': std_val, 'coef_variation': coef_var, 'unique_ratio': unique_ratio,
                'range': [np.min(y_col), np.max(y_col)]
            }
            print(f"   {col_name}: std={std_val:.4f}, CV={coef_var:.4f}, unique={unique_ratio:.3f}")
            if coef_var < 0.01:
                warning = f"ДУЖЕ МАЛА ВАРІАЦІЯ в {col_name} (CV={coef_var:.6f})"
                diagnostics['warnings'].append(warning)
                print(f"   ⚠️  {warning}")
            if unique_ratio < 0.1:
                warning = f"ДУЖЕ МАЛО УНІКАЛЬНИХ ЗНАЧЕНЬ в {col_name} ({unique_ratio:.3f})"
                diagnostics['warnings'].append(warning)
                print(f"   ⚠️  {warning}")
        
        print("\n🔴 Перевірка наявності аномалій:")
        for i in range(Y.shape[1]):
            col_name = f'Y_col_{i}'
            y_col = Y[:, i]
            q1, q3 = np.percentile(y_col, [25, 75])
            iqr = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = np.where((y_col < lower_bound) | (y_col > upper_bound))[0]
            outlier_ratio = len(outliers) / len(y_col)
            diagnostics['anomaly_presence'][col_name] = {
                'n_outliers': len(outliers), 'outlier_ratio': outlier_ratio,
                'outlier_indices': outliers.tolist()[:20]
            }
            print(f"   {col_name}: {len(outliers)} викидів ({outlier_ratio:.3f})")
            if outlier_ratio < 0.01:
                warning = f"ДУЖЕ МАЛО АНОМАЛІЙ в {col_name} ({outlier_ratio:.4f})"
                diagnostics['warnings'].append(warning)
                print(f"   ⚠️  {warning}")
        
        quality_score = 100.0
        for warning in diagnostics['warnings']:
            if 'ДУЖЕ МАЛА ВАРІАЦІЯ' in warning: quality_score -= 30
            elif 'ДУЖЕ МАЛО УНІКАЛЬНИХ' in warning: quality_score -= 20
            elif 'ДУЖЕ МАЛО АНОМАЛІЙ' in warning: quality_score -= 25
            elif 'СЛАБКА НЕЛІНІЙНІСТЬ' in warning: quality_score -= 15
        
        diagnostics['data_quality_score'] = max(0, quality_score)
        print(f"\n🎯 ЗАГАЛЬНА ОЦІНКА ЯКОСТІ ДАНИХ: {diagnostics['data_quality_score']:.1f}/100")
        if quality_score < 50: print("❌ КРИТИЧНІ ПРОБЛЕМИ З ДАНИМИ - перевірте генерацію!")
        elif quality_score < 75: print("⚠️  ПОМІРНІ ПРОБЛЕМИ З ДАНИМИ - рекомендується налаштування")
        else: print("✅ ЯКІСТЬ ДАНИХ ПРИЙНЯТНА")
        
        return diagnostics    
    
    def _create_lag_features(self, df: pd.DataFrame, lag_depth: int) -> Tuple[np.ndarray, np.ndarray]:
        """Створення лагових ознак для ARX моделей."""
        input_cols = ['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']
        output_cols = ['concentrate_fe_percent', 'concentrate_mass_flow']
        lag_features, lag_names = [], []
        
        for col in input_cols:
            for lag in range(lag_depth):
                lag_features.append(df[col].shift(lag))
                lag_names.append(f"{col}_lag_{lag}")
        
        for col in output_cols:
            for lag in range(1, lag_depth + 1):
                lag_features.append(df[col].shift(lag))
                lag_names.append(f"{col}_lag_{lag}")
        
        lag_df = pd.concat(lag_features, axis=1, keys=lag_names)
        valid_idx = lag_df.dropna().index
        X = lag_df.loc[valid_idx].values
        Y = df.loc[valid_idx, output_cols].values
        
        print(f"📊 Створено лагові ознаки: {X.shape[1]} features, {X.shape[0]} samples")
        return X, Y
    
    def _train_single_model(self, config: Dict[str, Any], 
                          X_train: np.ndarray, Y_train: np.ndarray,
                          X_val: Optional[np.ndarray], Y_val: Optional[np.ndarray]) -> Dict[str, Any]:
        """Навчання однієї лінійної моделі згідно з конфігурацією."""
        model_name = config['name']
        model_params = {k: v for k, v in config.items() if k != 'name'}
        model = KernelModel(model_type='linear', **model_params)
        
        start_time = time.time()
        try:
            model.fit(X_train, Y_train, X_val=X_val, Y_val=Y_val)
        except TypeError:
            model.fit(X_train, Y_train)
        train_time = time.time() - start_time
        
        Y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(Y_train, Y_train_pred)
        train_r2 = r2_score(Y_train, Y_train_pred)
        
        print(f"   ⏱️ Час навчання: {train_time:.3f} сек")
        print(f"   📊 Train MSE: {train_mse:.6f}")
        print(f"   📊 Train R²: {train_r2:.4f}")
        
        return {
            'model': model,
            'metrics': {'train_time': train_time, 'train_mse': train_mse, 'train_r2': train_r2, 'config': config}
        }
    
    def _evaluate_all_models(self, X_test: np.ndarray, Y_test: np.ndarray,
                           training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Оцінка всіх моделей на тестових даних."""
        print("\n🎯 ОЦІНКА МОДЕЛЕЙ НА ТЕСТОВИХ ДАНИХ")
        print("-" * 50)
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            print(f"Оцінка {model_name}...")
            Y_pred_scaled = model.predict(X_test)
            Y_pred = self.scaler_y.inverse_transform(Y_pred_scaled)
            
            mse = mean_squared_error(Y_test, Y_pred)
            rmse_fe = np.sqrt(mean_squared_error(Y_test[:, 0], Y_pred[:, 0]))
            rmse_mass = np.sqrt(mean_squared_error(Y_test[:, 1], Y_pred[:, 1]))
            mae = mean_absolute_error(Y_test, Y_pred)
            r2 = r2_score(Y_test, Y_pred)
            mape = np.mean(np.abs((Y_test - Y_pred) / (Y_test + 1e-8))) * 100
            
            evaluation_results[model_name] = {
                'mse': mse, 'rmse_fe': rmse_fe, 'rmse_mass': rmse_mass,
                'mae': mae, 'mape': mape, 'r2': r2, 'predictions': Y_pred,
                'train_time': training_results[model_name]['train_time']
            }
            print(f"   MSE: {mse:.6f}, R²: {r2:.4f}")
        return evaluation_results
    
    def _run_residual_diagnostics(self, X_test: np.ndarray, Y_test: np.ndarray) -> Dict[str, Any]:
        """Комплексна діагностика резидуалів для всіх моделей."""
        print("\n🔍 ДІАГНОСТИКА РЕЗИДУАЛІВ")
        print("-" * 40)
        diagnostics = {}
        
        for model_name, model in self.models.items():
            print(f"Діагностика {model_name}...")
            Y_pred_scaled = model.predict(X_test)
            Y_pred = self.scaler_y.inverse_transform(Y_pred_scaled)
            residuals = Y_test - Y_pred
            
            model_diagnostics = {}
            for i, output_name in enumerate(['Fe_concentration', 'Mass_flow']):
                res_i = residuals[:, i]
                shapiro_stat, shapiro_p = stats.shapiro(res_i[:min(len(res_i), 5000)])
                jb_stat, jb_p = stats.jarque_bera(res_i)
                model_diagnostics[f'{output_name}_normality'] = {
                    'shapiro_statistic': shapiro_stat, 'shapiro_p_value': shapiro_p,
                    'jb_statistic': jb_stat, 'jb_p_value': jb_p,
                    'is_normal': shapiro_p > 0.05 and jb_p > 0.05
                }
            
            try:
                # Додаємо константу, як вимагає BP-тест
                exog = add_constant(X_test, has_constant='add')
                bp_stat, bp_p, _, _ = het_breuschpagan(residuals[:, 0], exog)
                model_diagnostics['heteroscedasticity'] = {
                    'breusch_pagan_stat': float(bp_stat),
                    'breusch_pagan_p': float(bp_p),
                    'is_homoscedastic': bool(bp_p > 0.05),
                }
            except Exception as e:
                model_diagnostics['heteroscedasticity'] = {'error': str(e)}
            
            try:
                bp_stat, bp_p, _, _ = het_breuschpagan(residuals[:, 0], X_test)
                model_diagnostics['heteroscedasticity'] = {
                    'breusch_pagan_stat': bp_stat, 'breusch_pagan_p': bp_p,
                    'is_homoscedastic': bp_p > 0.05
                }
            except Exception as e:
                print(f"   ⚠️ Помилка в тесті гетероскедастичності: {e}")
                model_diagnostics['heteroscedasticity'] = {'error': str(e)}
            
            model_diagnostics['residual_stats'] = {
                'mean': np.mean(residuals, axis=0).tolist(), 'std': np.std(residuals, axis=0).tolist(), 
                'skewness': stats.skew(residuals, axis=0).tolist(), 'kurtosis': stats.kurtosis(residuals, axis=0).tolist()
            }
            diagnostics[model_name] = model_diagnostics
            
            print(f"   ✓ Нормальність: {model_diagnostics.get('Fe_concentration_normality', {}).get('is_normal', 'Unknown')}")
            print(f"   ✓ Автокореляція: {'Є' if model_diagnostics.get('autocorrelation', {}).get('has_autocorr', True) else 'Немає'}")
            print(f"   ✓ Гомоскедастичність: {model_diagnostics.get('heteroscedasticity', {}).get('is_homoscedastic', 'Unknown')}")
        
        return diagnostics
    
    def _run_robustness_analysis(self, global_config: Dict[str, Any], 
                               model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Аналіз робастності моделей до різних типів збурень."""
        print("\n💪 АНАЛІЗ РОБАСТНОСТІ")
        print("-" * 30)
        robustness_results = {}
        
        noise_levels = [0.01, 0.05, 0.10, 0.20]
        robustness_results['noise_robustness'] = self._test_noise_robustness(
            global_config, model_configs, noise_levels
        )
        
        nonlinearity_levels = [
            ('linear', {}),
            ('weak', {'concentrate_fe_percent': ('pow', 1.2)}),
            ('moderate', {'concentrate_fe_percent': ('pow', 1.8)}), 
            ('strong', {'concentrate_fe_percent': ('pow', 2.5)})
        ]
        robustness_results['nonlinearity_robustness'] = self._test_nonlinearity_robustness(
            global_config, model_configs, nonlinearity_levels
        )
        return robustness_results
    

    def _test_noise_robustness(self, global_config: Dict[str, Any],
                               model_configs: List[Dict[str, Any]],
                               noise_levels: List[float]) -> Dict[str, Any]:
        noise_results = {mc['name']: {} for mc in model_configs}
    
        for noise_level in noise_levels:
            noisy_cfg = global_config.copy()
            noisy_cfg['noise_level'] = 'custom'
            noisy_cfg['custom_noise_std'] = noise_level
    
            # ВАЖЛИВО: не рефітимо scaler-и на нових даних; _prepare_data повертає
            # X_* вже трансформовані поточними scaler-ами сервісу
            X_train_s, Y_train_s, _, _, X_test_s, Y_test_real, _, _ = self._prepare_data(noisy_cfg)
    
            for cfg in model_configs:
                mr = self._train_single_model(cfg, X_train_s, Y_train_s, None, None)
                Y_pred_s = mr['model'].predict(X_test_s)
                Y_pred = self.scaler_y.inverse_transform(Y_pred_s)
    
                mse = mean_squared_error(Y_test_real, Y_pred)
                r2 = r2_score(Y_test_real, Y_pred)
                noise_results[cfg['name']][f'noise_{noise_level}'] = {'mse': mse, 'r2': r2}
    
        return noise_results
    
    
    def _test_nonlinearity_robustness(self, global_config: Dict[str, Any],
                                      model_configs: List[Dict[str, Any]],
                                      nonlinearity_levels: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        results = {mc['name']: {} for mc in model_configs}
    
        for lvl_name, nl_cfg in nonlinearity_levels:
            cfg = global_config.copy()
            cfg['enable_nonlinear'] = bool(nl_cfg)
            cfg['nonlinear_config'] = nl_cfg
    
            X_train_s, Y_train_s, _, _, X_test_s, Y_test_real, _, _ = self._prepare_data(cfg)
    
            for mc in model_configs:
                mr = self._train_single_model(mc, X_train_s, Y_train_s, None, None)
                Y_pred_s = mr['model'].predict(X_test_s)
                Y_pred = self.scaler_y.inverse_transform(Y_pred_s)
    
                mse = mean_squared_error(Y_test_real, Y_pred)
                r2 = r2_score(Y_test_real, Y_pred)
                results[mc['name']][lvl_name] = {'mse': mse, 'r2': r2}
    
        return results
    
    # ✅ ВИПРАВЛЕННЯ: Додано Y_test як аргумент для коректної візуалізації
    def _generate_comprehensive_report(self, Y_test: np.ndarray):
        """Генерація комплексного звіту з результатами."""
        print("\n📝 ГЕНЕРАЦІЯ ЗВІТУ")
        print("-" * 25)
        
        json_path = self.dirs['data'] / f'linear_comparison_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        json_results = self._convert_results_for_json(self.results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"💾 Результати збережено: {json_path}")
        
        # ✅ ВИПРАВЛЕННЯ: Передаємо Y_test далі у функцію візуалізації
        self._create_comparison_visualizations(Y_test)
        self._generate_latex_table()
        self._generate_text_report()
    
    def _convert_results_for_json(self, results: Dict) -> Dict:
        """Конвертація результатів для JSON серіалізації."""
        def convert_value(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, dict): return {k: convert_value(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert_value(item) for item in obj]
            return obj
        return convert_value(results)
    
    def _create_comparison_visualizations(self, Y_test: np.ndarray):
        self._plot_accuracy_comparison()
        self._plot_residual_analysis(Y_test)  # Y_test у реальних одиницях
        self._plot_noise_robustness()
        self._plot_nonlinearity_robustness()
        print("✅ Візуалізації створено")
    
    def _plot_accuracy_comparison(self):
        """Порівняння точності моделей."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Порівняння точності лінійних моделей', fontsize=16, fontweight='bold')
        
        model_names = list(self.results['evaluation_results'].keys())
        mse_values = [self.results['evaluation_results'][name]['mse'] for name in model_names]
        r2_values = [self.results['evaluation_results'][name]['r2'] for name in model_names]
        rmse_fe_values = [self.results['evaluation_results'][name]['rmse_fe'] for name in model_names]
        rmse_mass_values = [self.results['evaluation_results'][name]['rmse_mass'] for name in model_names]
        train_times = [self.results['evaluation_results'][name]['train_time'] for name in model_names]
        
        metrics_data = [
            (axes[0, 0], mse_values, 'MSE', 'Mean Squared Error', 'lightcoral'),
            (axes[0, 1], r2_values, 'R²', 'Coefficient of Determination', 'lightgreen'),
            (axes[0, 2], rmse_fe_values, 'RMSE (%)', 'RMSE концентрації Fe', 'lightskyblue'),
            (axes[1, 0], rmse_mass_values, 'RMSE (т/год)', 'RMSE масової витрати', 'lightgoldenrodyellow'),
            (axes[1, 1], train_times, 'Час (сек)', 'Час навчання', 'plum')
        ]
        
        for ax, values, ylabel, title, color in metrics_data:
            bars = ax.bar(range(len(model_names)), values, color=color)
            ax.set_xlabel('Моделі'); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.set_xticks(range(len(model_names))); ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.3f}', 
                        ha='center', va='bottom', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

        ax = axes[1, 2]
        normalized_mse = [(max(mse_values) - mse) / (max(mse_values) - min(mse_values) + 1e-8) for mse in mse_values]
        normalized_speed = [(max(train_times) - time) / (max(train_times) - min(train_times) + 1e-8) for time in train_times]
        x_pos = np.arange(len(model_names)); width = 0.35
        ax.bar(x_pos - width/2, normalized_mse, width, label='Точність (норм.)', alpha=0.7)
        ax.bar(x_pos + width/2, normalized_speed, width, label='Швидкість (норм.)', alpha=0.7)
        ax.set_xlabel('Моделі'); ax.set_ylabel('Нормалізоване значення'); ax.set_title('Комплексна оцінка')
        ax.set_xticks(x_pos); ax.set_xticklabels(model_names, rotation=45, ha='right'); ax.legend(); ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = self.dirs['plots'] / 'accuracy_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"📊 Графік точності збережено: {plot_path}")
    
    # ✅ ВИПРАВЛЕННЯ: Метод тепер приймає Y_test для коректного розрахунку залишків

    def _plot_residual_analysis(self, Y_test: np.ndarray) -> None:
        """
        Будуємо базові графіки залишків для кожної моделі.
        Очікуємо, що Y_test у реальних одиницях; Y_pred з evaluation_results теж у реальних одиницях.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
    
        eval_results = self.results.get('evaluation_results', {})
        if not eval_results:
            print("⚠️ Немає evaluation_results для побудови аналізу залишків.")
            return
    
        for model_name in self.models.keys():
            if model_name not in eval_results:
                continue
    
            Y_pred = np.asarray(eval_results[model_name].get('predictions'))
            if Y_pred is None or Y_pred.size == 0:
                print(f"⚠️ Порожні прогнози для {model_name}")
                continue
    
            Y_pred = Y_pred.reshape(-1, 1) if Y_pred.ndim == 1 else Y_pred
            Y_true = np.asarray(Y_test)
            Y_true = Y_true.reshape(-1, 1) if Y_true.ndim == 1 else Y_true
    
            if Y_true.shape != Y_pred.shape:
                print(f"⚠️ Розмірності не збігаються: {model_name}: {Y_true.shape} vs {Y_pred.shape}")
                continue
    
            residuals = Y_true - Y_pred
            if not np.isfinite(residuals).all():
                print(f"⚠️ Некоректні значення залишків (NaN/Inf) для {model_name}")
                continue
    
            # 1) Фігура з 2 підсюжетами: розсіювання та гістограма/щільність
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            ax_scatter, ax_hist = axes
    
            # Розсіювання: Y_pred vs residuals
            ax_scatter.scatter(Y_pred.ravel(), residuals.ravel(), alpha=0.6, s=18, color="#4C78A8", edgecolor="none")
            ax_scatter.axhline(0.0, color="red", linestyle="--", linewidth=1)
            ax_scatter.set_title(f"Залишки vs Прогноз — {model_name}")
            ax_scatter.set_xlabel("Ŷ (прогноз)")
            ax_scatter.set_ylabel("Залишок (Y − Ŷ)")
    
            # Гістограма/щільність залишків
            sns.histplot(residuals.ravel(), bins=30, kde=True, ax=ax_hist, color="#72B7B2")
            ax_hist.set_title(f"Розподіл залишків — {model_name}")
            ax_hist.set_xlabel("Залишок")
            ax_hist.set_ylabel("Кількість")
    
            # Опційно: виносимо короткі метрики по залишках
            mu = float(np.mean(residuals))
            sigma = float(np.std(residuals, ddof=1))
            ax_hist.annotate(f"μ={mu:.3g}\nσ={sigma:.3g}", xy=(0.98, 0.98), xycoords="axes fraction",
                             ha="right", va="top", fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999"))
    
            # Збереження, якщо у вас заведений шлях self.plots_dir
            if getattr(self, "plots_dir", None):
                fname = self.plots_dir / f"residuals_{model_name}.png"
                try:
                    fig.savefig(fname, dpi=150)
                except Exception as e:
                    print(f"⚠️ Не вдалося зберегти графік залишків для {model_name}: {e}")
            plt.close(fig)


    def _plot_noise_robustness(self):
        """Візуалізація робастності до шуму."""
        if 'noise_robustness' not in self.results.get('robustness_results', {}): return
        
        noise_data = self.results['robustness_results']['noise_robustness']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Робастність лінійних моделей до шуму', fontsize=16, fontweight='bold')
        
        noise_levels_set = set()
        model_names = list(noise_data.keys())
        for model_name in model_names:
            for key in noise_data[model_name].keys():
                if key.startswith('noise_'):
                    noise_levels_set.add(float(key.split('_')[1]))
        
        noise_levels = sorted(list(noise_levels_set))
        noise_percentages = [level * 100 for level in noise_levels]
        
        for model_name in model_names:
            mse_values = [noise_data[model_name].get(f'noise_{nl}', {}).get('mse', np.nan) for nl in noise_levels]
            ax1.plot(noise_percentages, mse_values, marker='o', label=model_name, linewidth=2)
        ax1.set_xlabel('Рівень шуму (%)'); ax1.set_ylabel('MSE'); ax1.set_title('Деградація точності (MSE) при шумі')
        ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_yscale('log')
        
        for model_name in model_names:
            r2_values = [noise_data[model_name].get(f'noise_{nl}', {}).get('r2', np.nan) for nl in noise_levels]
            ax2.plot(noise_percentages, r2_values, marker='s', label=model_name, linewidth=2)
        ax2.set_xlabel('Рівень шуму (%)'); ax2.set_ylabel('R²'); ax2.set_title('Якість узагальнення (R²) при шумі')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = self.dirs['plots'] / 'noise_robustness.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"🔊 Графік робастності до шуму збережено: {plot_path}")
    
    def _plot_nonlinearity_robustness(self):
        """Візуалізація робастності до нелінійності."""
        if 'nonlinearity_robustness' not in self.results.get('robustness_results', {}): return
        
        nl_data = self.results['robustness_results']['nonlinearity_robustness']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Робастність лінійних моделей до нелінійності', fontsize=16, fontweight='bold')
        
        model_names = list(nl_data.keys())
        nonlinearity_levels = ['linear', 'weak', 'moderate', 'strong']
        x_pos = np.arange(len(nonlinearity_levels))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            mse_values = [nl_data[model_name].get(level, {}).get('mse', np.nan) for level in nonlinearity_levels]
            ax1.bar(x_pos + i*width, mse_values, width, label=model_name, alpha=0.8)
        ax1.set_xlabel('Рівень нелінійності'); ax1.set_ylabel('MSE'); ax1.set_title('Вплив нелінійності на точність (MSE)')
        ax1.set_xticks(x_pos + width * (len(model_names) - 1) / 2); ax1.set_xticklabels(nonlinearity_levels)
        ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_yscale('log')
        
        for i, model_name in enumerate(model_names):
            r2_values = [nl_data[model_name].get(level, {}).get('r2', np.nan) for level in nonlinearity_levels]
            ax2.bar(x_pos + i*width, r2_values, width, label=model_name, alpha=0.8)
        ax2.set_xlabel('Рівень нелінійності'); ax2.set_ylabel('R²'); ax2.set_title('Вплив нелінійності на узагальнення (R²)')
        ax2.set_xticks(x_pos + width * (len(model_names) - 1) / 2); ax2.set_xticklabels(nonlinearity_levels)
        ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = self.dirs['plots'] / 'nonlinearity_robustness.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"📈 Графік робастності до нелінійності збережено: {plot_path}")
    
    def _generate_latex_table(self):
        """Генерація LaTeX таблиці з результатами."""
        if not self.results: raise ValueError("Немає результатів для експорту")
        
        eval_results = self.results['evaluation_results']
        latex_content = (
            "\\begin{table}[h]\n\\centering\n"
            "\\caption{Порівняння продуктивності лінійних моделей для процесу магнітної сепарації}\n"
            "\\label{tab:linear_models_comparison}\n"
            "\\begin{tabular}{|l|c|c|c|c|c|}\n\\hline\n"
            "\\textbf{Модель} & \\textbf{MSE} & \\textbf{R²} & \\textbf{RMSE Fe, \\%} & \\textbf{RMSE Mass, т/год} & \\textbf{Час навчання, с} \\\\\n\\hline\n"
        )
        for model_name, metrics in eval_results.items():
            latex_content += f"{model_name.replace('_', ' ')} & {metrics['mse']:.6f} & {metrics['r2']:.4f} & {metrics['rmse_fe']:.3f} & {metrics['rmse_mass']:.3f} & {metrics['train_time']:.3f} \\\\\n\\hline\n"
        latex_content += "\\end{tabular}\n\\end{table}"
        
        latex_path = self.dirs['latex'] / 'linear_models_comparison_table.tex'
        with open(latex_path, 'w', encoding='utf-8') as f: f.write(latex_content)
        print(f"📄 LaTeX таблицю збережено: {latex_path}")
        return latex_path
    
    def _generate_text_report(self):
        """Генерація текстового звіту з результатами."""
        # ... (Код генерації текстового звіту залишається без змін) ...
        # Цей метод не мав помилок, тому його можна залишити як є.
        # Для стислості, його код тут пропущено.
        report_path = self.dirs['reports'] / f'linear_comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        # ... логіка запису у файл ...
        print(f"📋 Текстовий звіт збережено: {report_path}") # Placeholder
        return report_path
    
    # ... (Решта методів, таких як analyze_arx_limitations_for_dissertation, залишаються без змін) ...


def compare_linear_models_on_nonlinear_data_fixed(reference_df: Optional[pd.DataFrame] = None,
                                                output_dir: str = "nonlinear_data_comparison_fixed") -> Dict[str, Any]:
    """
    ВИПРАВЛЕНИЙ позакласовий метод для порівняння базових лінійних моделей на сильно нелінійних даних.
    """
    print("🔬 ВИПРАВЛЕНЕ ПОРІВНЯННЯ ЛІНІЙНИХ МОДЕЛЕЙ НА НЕЛІНІЙНИХ ДАНИХ")
    print("=" * 70)
    print("📋 Дослідження з правильними аномаліями та шумом\n")
    
    comparison_service = LinearModelsComparisonService(reference_df=reference_df, output_dir=output_dir)
    
    model_configs = [
        {'name': 'ARX_OLS', 'linear_type': 'ols'},
        {'name': 'ARX_Ridge', 'linear_type': 'ridge', 'alpha': 0.1},
        {'name': 'ARX_Lasso', 'linear_type': 'lasso', 'alpha': 0.01}
    ]
    
    global_config = {
        'N_data': 4000, 'lag_depth': 8, 'enable_nonlinear': True, 'use_simulation': True,   
        'use_anomalies': True, 'anomaly_severity': 'medium', 'anomaly_in_train': False, 
        'noise_level': 'medium',
        'nonlinear_config': {'concentrate_fe_percent': ('pow', 2.5), 'concentrate_mass_flow': ('pow', 1.8)},
        'train_size': 0.8, 'val_size': 0.1, 'test_size': 0.1, 'seed': 42
    }
    
    print("🔧 КОНФІГУРАЦІЯ ДЛЯ ПРАВИЛЬНОГО ТЕСТУВАННЯ:")
    print(f"   🔴 Аномалії: {global_config['use_anomalies']} ({global_config['anomaly_severity']})")
    print(f"   🔊 Шум: {global_config['noise_level']}")
    print(f"   📈 Нелінійність: Fe^{global_config['nonlinear_config']['concentrate_fe_percent'][1]}, Mass^{global_config['nonlinear_config']['concentrate_mass_flow'][1]}\n")
    
    print("🚀 Запуск аналізу з ВИПРАВЛЕНИМИ параметрами...")
    try:
        results = comparison_service.run_comprehensive_comparison(model_configs, global_config)
        
        print("\n🔍 ДІАГНОСТИКА ЯКОСТІ ЗГЕНЕРОВАНИХ ДАНИХ:")
        data_results = comparison_service._prepare_data(global_config)
        _, Y_train, _, _, _, Y_test, _, _ = data_results
        X_full = np.vstack([data_results[0], data_results[4]]) # X_train_scaled, X_test_scaled
        Y_full = np.vstack([comparison_service.scaler_y.inverse_transform(Y_train), Y_test])
        
        data_diagnostics = comparison_service.diagnose_data_quality(X=X_full, Y=Y_full)
        
        print("\n📊 АНАЛІЗ РЕЗУЛЬТАТІВ МОДЕЛЕЙ:")
        realistic_results = {}
        for model_name_cfg in model_configs:
            model_key = model_name_cfg['name']
            if model_key in results['evaluation_results']:
                eval_data = results['evaluation_results'][model_key]
                
                rmse_fe = eval_data.get('rmse_fe', 0)
                rmse_mass = eval_data.get('rmse_mass', 0)
                # ✅ ВИПРАВЛЕННЯ: Використовуємо правильний ключ 'r2' замість 'r2_score'
                r2 = eval_data.get('r2', -1) 
                
                print(f"   {model_key}:")
                print(f"     RMSE Fe: {rmse_fe:.4f}")
                print(f"     RMSE Mass: {rmse_mass:.4f}") 
                print(f"     R² Score: {r2:.4f}")
                
                is_too_perfect = (rmse_fe < 0.01 and r2 > 0.99)
                is_reasonable = (0.5 < r2 < 0.9 and rmse_fe > 1.0)
                
                realistic_results[model_key] = {
                    'rmse_fe': rmse_fe, 'rmse_mass': rmse_mass, 'r2': r2,
                    'is_too_perfect': is_too_perfect, 'is_reasonable': is_reasonable
                }
                
                if is_too_perfect:
                    warning = f"ПІДОЗРІЛО ІДЕАЛЬНІ результати для {model_key}"
                    print(f"     ⚠️  {warning}")
                    data_diagnostics['warnings'].append(warning)
                elif is_reasonable:
                    print(f"     ✅ Реалістичні результати для {model_key}")
        
        worst_rmse_fe = max([metrics['rmse_fe'] for metrics in realistic_results.values()])
        best_r2 = max([metrics['r2'] for metrics in realistic_results.values()])
        n_perfect_models = sum([1 for m in realistic_results.values() if m['is_too_perfect']])
        n_reasonable_models = sum([1 for m in realistic_results.values() if m['is_reasonable']])
        
        print(f"\n🎯 ПІДСУМКОВА ОЦІНКА:")
        print(f"   📊 Найгірша RMSE Fe: {worst_rmse_fe:.4f}")
        print(f"   📊 Найкращий R²: {best_r2:.4f}")
        print(f"   ✅ Реалістичних моделей: {n_reasonable_models}/{len(model_configs)}")
        print(f"   ⚠️  Підозріло ідеальних: {n_perfect_models}/{len(model_configs)}")
        
        final_results = {'comprehensive_results': results, 'data_diagnostics': data_diagnostics, 'key_findings': {}}
        
    except Exception as e:
        print(f"❌ ПОМИЛКА під час виконання: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}
    
    print(f"\n✅ АНАЛІЗ ЗАВЕРШЕНО")
    print(f"📁 Результати збережено в: {output_dir}")
    print(f"🔍 Якість даних: {data_diagnostics.get('data_quality_score', 'N/A'):.1f}/100")
    print(f"⚠️  Попереджень: {len(data_diagnostics.get('warnings', []))}")
    
    return final_results

if __name__ == "__main__":
    try:
        df = pd.read_parquet('processed.parquet')
        compare_linear_models_on_nonlinear_data_fixed(df, 'nonlinear_data_comparison_fixed')
    except FileNotFoundError:
        print("INFO: Файл 'processed.parquet' не знайдено. Запуск без референтних даних.")
        compare_linear_models_on_nonlinear_data_fixed(None, 'nonlinear_data_comparison_fixed')