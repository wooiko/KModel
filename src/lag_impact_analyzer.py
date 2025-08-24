# lag_impact_analyzer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Optional, Tuple
import json

# Імпорт з модулів проекту (як у ModelComparisonService)
from model import KernelModel
from data_gen import StatefulDataGenerator


class KernelLagAnalyzer:
    """
    Спрощений інструмент для аналізу впливу кількості лагів на якість ядерних моделей.
    
    Підхід аналогічний ModelComparisonService:
    - Отримує посилання на базовий датасет
    - Використовує create_simulation_data для генерації необхідного набору
    - Фокус на ядерних моделях (KRR, SVR, GPR) з model.py
    """
    
    def __init__(self, 
                 reference_df: Optional[pd.DataFrame] = None,
                 model_types: List[str] = ["krr", "svr"], 
                 lag_range: range = range(1, 11),
                 output_dir: Optional[str] = None):
        """
        Ініціалізація аналізатора лагів.
        
        Args:
            reference_df: Референтний датасет (як у ModelComparisonService)
            model_types: Список типів ядерних моделей для тестування
            lag_range: Діапазон лагів для тестування
            output_dir: Директорія для збереження результатів
        """
        self.reference_df = self._load_reference_data(reference_df)
        self.model_types = model_types
        self.lag_range = lag_range
        self.results = {}
        
        # Створення структури директорій (як у ModelComparisonService)
        self._setup_directories(output_dir)
        
        print(f"🚀 Ініціалізовано KernelLagAnalyzer")
        print(f"   Референтний датасет: {len(self.reference_df)} записів")
        print(f"   Моделі: {model_types}")
        print(f"   Лаги: {list(lag_range)}")
        print(f"   Вихідна директорія: {self.output_dir}")
    
    def _load_reference_data(self, reference_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Завантаження референтних даних (аналогічно ModelComparisonService)"""
        if reference_df is None:
            try:
                reference_df = pd.read_parquet('processed.parquet')
                print(f"✅ Завантажено {len(reference_df)} записів з processed.parquet")
            except FileNotFoundError:
                print("❌ Файл 'processed.parquet' не знайдено")
                raise
        return reference_df
    
    def _setup_directories(self, output_dir: Optional[str]) -> None:
        """Створення структури директорій (аналогічно ModelComparisonService)"""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"lag_analysis/{timestamp}"
        
        self.output_dir = Path(output_dir)
        
        # Структура директорій
        self.dirs = {
            'data': self.output_dir / 'data',
            'plots': self.output_dir / 'plots', 
            'reports': self.output_dir / 'reports'
        }
        
        # Створення всіх директорій
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"📁 Результати зберігатимуться у: {self.output_dir.absolute()}")

    def _get_default_params(self) -> dict:
        """Базові параметри для симуляції"""
        return {
            'N_data': 5000,
            'control_pts': 500,
            'train_size': 0.8,
            'val_size': 0.1,
            'test_size': 0.1,
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
            'n_neighbors': 5,
            'noise_level': 'none',
            'enable_nonlinear': False,
            'use_anomalies': False,
            'seed': 42
        }
    
    def create_simulation_data(self, params: dict) -> Tuple[StatefulDataGenerator, pd.DataFrame]:
        """
        Створення симуляційних даних через StatefulDataGenerator.
        ТОЧНА КОПІЯ з ModelComparisonService.
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

        # Аномалії (спрощено, без них для чистоти аналізу лагів)
        anomaly_cfg = None
        if params.get('use_anomalies', False):
            anomaly_cfg = self._create_anomaly_config(
                N_data=params['N_data'],
                train_frac=params.get('train_size', 0.8),
                val_frac=params.get('val_size', 0.1),
                test_frac=params.get('test_size', 0.1),
                seed=params['seed'],
                severity=params.get('anomaly_severity', 'mild')
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
    
    def _create_anomaly_config(self, N_data: int, train_frac: float = 0.7, 
                             val_frac: float = 0.15, test_frac: float = 0.15,
                             seed: int = 42, severity: str = "mild") -> dict:
        """Спрощена версія create_anomaly_config з ModelComparisonService"""
        # Для аналізу лагів краще без аномалій, але залишаємо можливість
        return {}  # Порожня конфігурація = без аномалій
    
    def create_lagged_matrices(self, df: pd.DataFrame, lag: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Створення лагових матриць (ТОЧНА КОПІЯ з ModelComparisonService).
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
    
    def run_lag_analysis(self, base_params: Optional[Dict] = None) -> Dict:
        """
        Основний аналіз впливу лагів на якість ядерних моделей.
        
        Args:
            base_params: Базові параметри симуляції (як у ModelComparisonService)
            
        Returns:
            Dict: Результати аналізу
        """
        print(f"📊 Початок аналізу впливу лагів на ядерні моделі...")
        
        # Базові параметри симуляції (як у ModelComparisonService)
        if base_params is None:
            base_params = {
                'N_data': 5000,
                'control_pts': 500,
                'train_size': 0.8,
                'val_size': 0.1,
                'test_size': 0.1,
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
                'seed': 42,
                'n_neighbors': 5,
                'noise_level': 'none',
                'enable_nonlinear': True,
                'nonlinear_config': {
                    'concentrate_fe_percent': ('pow', 2.0),
                    'concentrate_mass_flow': ('pow', 1.5)
                },
                'use_anomalies': False,  # Для чистоти аналізу лагів
            }
        
        results = {
            'base_params': base_params,
            'lag_results': {},
            'best_lags': {},
            'model_comparison': {}
        }
        
        # Цикл по лагам
        for lag in self.lag_range:
            print(f"\n🔄 Тестування lag={lag}")
            
            # Створення симуляційних даних з поточним лагом
            sim_params = base_params.copy()
            sim_params['current_lag'] = lag
            
            true_gen, df_sim = self.create_simulation_data(sim_params)
            
            # Створення лагових матриць
            X, Y = self.create_lagged_matrices(df_sim, lag)
            
            if X.shape[0] == 0:
                print(f"   ⚠️ Недостатньо даних для lag={lag}")
                continue
            
            print(f"   Розмірність: X{X.shape}, Y{Y.shape}")
            
            # Розділення на train/test
            n = X.shape[0]
            n_train = int(sim_params['train_size'] * n)
            
            X_train, X_test = X[:n_train], X[n_train:]
            Y_train, Y_test = Y[:n_train], Y[n_train:]
            
            # Масштабування (як у ModelComparisonService)
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            X_train_scaled = x_scaler.fit_transform(X_train)
            X_test_scaled = x_scaler.transform(X_test)
            Y_train_scaled = y_scaler.fit_transform(Y_train)
            
            print(f"   Дані: train={X_train_scaled.shape[0]}, test={X_test_scaled.shape[0]}")
            
            # Тестування кожної ядерної моделі
            lag_results = {}
            for model_type in self.model_types:
                metrics = self._test_kernel_model(model_type, 
                                         X_train_scaled, Y_train_scaled,
                                         X_test_scaled, Y_test, y_scaler)
                lag_results[model_type] = metrics
                print(f"   {model_type.upper()}: RMSE={metrics['rmse']:.4f}")
            
            results['lag_results'][lag] = lag_results
        
        # Аналіз найкращих лагів для кожної моделі
        for model_type in self.model_types:
            best_lag = self._find_best_lag(results['lag_results'], model_type)
            results['best_lags'][model_type] = best_lag
            print(f"\n✅ Найкращий lag для {model_type.upper()}: {best_lag}")
        
        # Порівняння моделей
        results['model_comparison'] = self._compare_models(results['lag_results'])
        
        self.results = results
        return results
    
    def _test_kernel_model(self, model_type: str, 
                          X_train: np.ndarray, Y_train: np.ndarray,
                          X_test: np.ndarray, Y_test: np.ndarray, 
                          y_scaler: StandardScaler) -> Dict:
        """Тестування ядерної моделі з оптимізацією параметрів"""
        
        # ПРАВИЛЬНО: з оптимізацією параметрів
        if model_type == "krr":
            model = KernelModel(model_type="krr", kernel="rbf", find_optimal_params=True)  # ✅
        elif model_type == "svr":
            model = KernelModel(model_type="svr", kernel="rbf", find_optimal_params=True)  # ✅
        elif model_type == "gpr":
            model = KernelModel(model_type="gpr", find_optimal_params=True)  # ✅
        else:
            raise ValueError(f"Невідомий тип ядерної моделі: {model_type}")
        
        # Навчання з оптимізацією
        start_time = datetime.now()
        model.fit(X_train, Y_train)
        train_time = (datetime.now() - start_time).total_seconds()
        
        # Решта коду без змін...
        
        # Прогнозування
        Y_pred_scaled = model.predict(X_test)
        Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
        
        # Метрики
        mse = mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)
        
        # Метрики по кожній цільовій змінній
        rmse_per_target = []
        for i in range(Y_test.shape[1]):
            rmse_target = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
            rmse_per_target.append(rmse_target)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'rmse_per_target': rmse_per_target,
            'train_time': train_time
        }
    
    def _find_best_lag(self, lag_results: Dict, model_type: str) -> int:
        """Знаходження найкращого лагу для моделі"""
        best_rmse = float('inf')
        best_lag = None
        
        for lag, results in lag_results.items():
            if model_type in results:
                rmse = results[model_type]['rmse']
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_lag = lag
        
        return best_lag
    
    def _compare_models(self, lag_results: Dict) -> Dict:
        """Порівняння ядерних моделей на кращих лагах"""
        comparison = {}
        
        for model_type in self.model_types:
            best_lag = self._find_best_lag(lag_results, model_type)
            if best_lag and best_lag in lag_results:
                comparison[model_type] = {
                    'best_lag': best_lag,
                    'best_rmse': lag_results[best_lag][model_type]['rmse'],
                    'best_mse': lag_results[best_lag][model_type]['mse']
                }
        
        return comparison
    
    def plot_lag_analysis(self) -> None:
        """Створення основних графіків аналізу лагів"""
        if not self.results:
            raise ValueError("Немає результатів для візуалізації. Спочатку запустіть run_lag_analysis()")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Графік 1: RMSE vs Lag (ПРИНЦИПОВО ВАЖЛИВИЙ)
        colors = {'krr': 'blue', 'svr': 'red', 'gpr': 'green'}
        
        for model_type in self.model_types:
            lags = []
            rmse_values = []
            
            for lag in self.lag_range:
                if lag in self.results['lag_results']:
                    if model_type in self.results['lag_results'][lag]:
                        lags.append(lag)
                        rmse_values.append(self.results['lag_results'][lag][model_type]['rmse'])
            
            if lags:
                ax1.plot(lags, rmse_values, 
                        marker='o', linewidth=2, markersize=6,
                        color=colors.get(model_type, 'black'),
                        label=f'{model_type.upper()}')
                
                # Позначення найкращого лагу
                best_lag = self.results['best_lags'].get(model_type)
                if best_lag in lags:
                    best_idx = lags.index(best_lag)
                    ax1.scatter(best_lag, rmse_values[best_idx], 
                              s=100, color=colors.get(model_type), 
                              marker='*', zorder=5)
                    ax1.annotate(f'Оптимальний lag={best_lag}', 
                               xy=(best_lag, rmse_values[best_idx]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=9, alpha=0.8)
        
        ax1.set_xlabel('Кількість лагів')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Вплив кількості лагів на якість ядерних моделей')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Графік 2: Порівняння ядерних моделей на оптимальних лагах (ПРИНЦИПОВО ВАЖЛИВИЙ)
        model_names = []
        best_rmse = []
        best_lags = []
        
        for model_type in self.model_types:
            if model_type in self.results['model_comparison']:
                model_names.append(model_type.upper())
                best_rmse.append(self.results['model_comparison'][model_type]['best_rmse'])
                best_lags.append(self.results['model_comparison'][model_type]['best_lag'])
        
        if model_names:
            bars = ax2.bar(model_names, best_rmse, 
                          color=[colors.get(m.lower(), 'gray') for m in model_names],
                          alpha=0.7)
            
            # Додавання значень на стовпці
            for bar, rmse, lag in zip(bars, best_rmse, best_lags):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(best_rmse)*0.01,
                        f'RMSE: {rmse:.4f}\nОпт. lag: {lag}', 
                        ha='center', va='bottom', fontsize=9)
        
        ax2.set_ylabel('Найкращий RMSE')
        ax2.set_title('Порівняння ядерних моделей на оптимальних лагах')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Збереження
        plot_path = self.dirs['plots'] / 'kernel_lag_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Основний графік збережено: {plot_path}")
    
    def plot_detailed_comparison(self) -> None:
        """Детальний графік порівняння по цільових змінних"""
        if not self.results:
            raise ValueError("Немає результатів для візуалізації")
        
        # Визначення кількості цільових змінних
        n_targets = None
        for lag_data in self.results['lag_results'].values():
            for model_data in lag_data.values():
                n_targets = len(model_data['rmse_per_target'])
                break
            if n_targets:
                break
        
        if not n_targets:
            print("⚠️ Не знайдено даних для детального порівняння")
            return
        
        fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))
        if n_targets == 1:
            axes = [axes]
        
        colors = {'krr': 'blue', 'svr': 'red', 'gpr': 'green'}
        target_names = ['Концентрація Fe', 'Масовий потік'] if n_targets == 2 else [f'Ціль {i+1}' for i in range(n_targets)]
        
        for target_idx in range(n_targets):
            ax = axes[target_idx]
            
            for model_type in self.model_types:
                lags = []
                rmse_values = []
                
                for lag in self.lag_range:
                    if (lag in self.results['lag_results'] and 
                        model_type in self.results['lag_results'][lag]):
                        lags.append(lag)
                        rmse_values.append(self.results['lag_results'][lag][model_type]['rmse_per_target'][target_idx])
                
                if lags:
                    ax.plot(lags, rmse_values, 
                           marker='o', linewidth=2, markersize=5,
                           color=colors.get(model_type, 'black'),
                           label=f'{model_type.upper()}')
            
            ax.set_xlabel('Кількість лагів')
            ax.set_ylabel('RMSE')
            ax.set_title(f'{target_names[target_idx]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Збереження
        plot_path = self.dirs['plots'] / 'detailed_kernel_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Детальний графік збережено: {plot_path}")
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Збереження результатів у JSON"""
        if not self.results:
            raise ValueError("Немає результатів для збереження")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'kernel_lag_analysis_{timestamp}.json'
        
        filepath = self.dirs['data'] / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"💾 Результати збережено: {filepath}")
        return str(filepath)
    
    def generate_report(self) -> str:
        """Генерація текстового звіту"""
        if not self.results:
            raise ValueError("Немає результатів для звіту")
        
        report = f"""
ЗВІТ ПРО АНАЛІЗ ВПЛИВУ ЛАГІВ НА ЯДЕРНІ МОДЕЛІ
{'='*55}

ПАРАМЕТРИ АНАЛІЗУ:
    Ядерні моделі: {', '.join([m.upper() for m in self.model_types])}
    Діапазон лагів: {list(self.lag_range)}
    Дата аналізу: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Референтний датасет: {len(self.reference_df)} записів

НАЙКРАЩІ РЕЗУЛЬТАТИ ЗА МОДЕЛЯМИ:
"""
        
        for model_type in self.model_types:
            if model_type in self.results['model_comparison']:
                comp_data = self.results['model_comparison'][model_type]
                report += f"""
    {model_type.upper()}:
        Оптимальний lag: {comp_data['best_lag']}
        Найкращий RMSE: {comp_data['best_rmse']:.6f}
        Найкращий MSE: {comp_data['best_mse']:.6f}
"""
        
        # Порівняння моделей
        report += f"\nПОРІВНЯННЯ ЯДЕРНИХ МОДЕЛЕЙ:\n"
        
        if len(self.model_types) > 1:
            model_performances = []
            for model_type in self.model_types:
                if model_type in self.results['model_comparison']:
                    model_performances.append({
                        'model': model_type.upper(),
                        'rmse': self.results['model_comparison'][model_type]['best_rmse']
                    })
            
            if model_performances:
                best_model = min(model_performances, key=lambda x: x['rmse'])
                worst_model = max(model_performances, key=lambda x: x['rmse'])
                
                improvement = ((worst_model['rmse'] - best_model['rmse']) / worst_model['rmse']) * 100
                
                report += f"    Найкраща модель: {best_model['model']} (RMSE: {best_model['rmse']:.6f})\n"
                report += f"    Найгірша модель: {worst_model['model']} (RMSE: {worst_model['rmse']:.6f})\n"
                report += f"    Покращення: {improvement:.1f}%\n"
        
        report += f"\n{'='*55}\n"
        
        # Збереження звіту
        report_path = self.dirs['reports'] / 'kernel_lag_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📝 Звіт збережено: {report_path}")
        return str(report_path)


# =============================================================================
# ПОЗАКЛАСОВІ ПРИКЛАДИ З ВИКОРИСТАННЯМ БАЗОВОГО ДАТАСЕТУ
# =============================================================================

def example_1_basic_kernel_lag_analysis():
    """Приклад 1: Базовий аналіз лагів для ядерних моделей"""
    print("=== ПРИКЛАД 1: БАЗОВИЙ АНАЛІЗ ЛАГІВ ДЛЯ ЯДЕРНИХ МОДЕЛЕЙ ===\n")
    
    # Створення аналізатора з базовим датасетом
    analyzer = KernelLagAnalyzer(
        # reference_df=None,  # Автоматично завантажить processed.parquet
        model_types=["krr", "svr"],
        lag_range=range(1, 8)
    )
    
    # Запуск аналізу
    results = analyzer.run_lag_analysis()
    
    # Візуалізація
    analyzer.plot_lag_analysis()
    
    # Збереження результатів
    analyzer.save_results()
    analyzer.generate_report()
    
    return analyzer

def example_2_compare_all_kernel_models():
    """Приклад 2: Порівняння всіх ядерних моделей (KRR, SVR, GPR)"""
    print("\n=== ПРИКЛАД 2: ПОРІВНЯННЯ ВСІХ ЯДЕРНИХ МОДЕЛЕЙ ===\n")
    
    # Аналізатор з усіма ядерними моделями
    analyzer = KernelLagAnalyzer(
        model_types=["krr", "svr", "gpr"],
        lag_range=range(1, 10),
        output_dir="kernel_comparison_full"
    )
    
    # Спеціальні параметри для складнішого процесу
    custom_params = {
        'N_data': 6000,
        'control_pts': 600,
        'enable_nonlinear': True,
        'nonlinear_config': {
            'concentrate_fe_percent': ('pow', 2.2),
            'concentrate_mass_flow': ('pow', 1.8)
        }
    }
    
    # Аналіз
    results = analyzer.run_lag_analysis(custom_params)
    
    # Детальні візуалізації
    analyzer.plot_lag_analysis()
    analyzer.plot_detailed_comparison()
    
    # Збереження
    analyzer.save_results("full_kernel_lag_comparison.json")
    analyzer.generate_report()
    
    return analyzer

def example_3_focus_on_krr_optimization():
    """Приклад 3: Фокус на оптимізації KRR для різних лагів"""
    print("\n=== ПРИКЛАД 3: ОПТИМІЗАЦІЯ KRR ДЛЯ РІЗНИХ ЛАГІВ ===\n")
    
    # Тільки KRR для детального аналізу
    analyzer = KernelLagAnalyzer(
        model_types=["krr"],
        lag_range=range(1, 15),  # Ширший діапазон
        output_dir="krr_lag_optimization"
    )
    
    # Параметри з більшим акцентом на якість
    quality_params = {
        'N_data': 8000,
        'control_pts': 800,
        'train_size': 0.85,  # Більше тренувальних даних
        'enable_nonlinear': True,
        'nonlinear_config': {
            'concentrate_fe_percent': ('pow', 2.5),
            'concentrate_mass_flow': ('pow', 2.0)
        }
    }
    
    # Аналіз
    results = analyzer.run_lag_analysis(quality_params)
    
    # Візуалізація
    analyzer.plot_lag_analysis()
    analyzer.plot_detailed_comparison()
    
    # Детальний звіт
    analyzer.generate_report()
    
    print(f"\n📊 РЕЗУЛЬТАТИ ОПТИМІЗАЦІЇ KRR:")
    if 'krr' in analyzer.results['model_comparison']:
        best_result = analyzer.results['model_comparison']['krr']
        print(f"   Оптимальний lag: {best_result['best_lag']}")
        print(f"   Найкращий RMSE: {best_result['best_rmse']:.6f}")
    
    return analyzer

def example_4_realistic_process_conditions():
    """Приклад 4: Реалістичні умови процесу магнітної сепарації"""
    print("\n=== ПРИКЛАД 4: РЕАЛІСТИЧНІ УМОВИ ПРОЦЕСУ ===\n")
    
    # Аналіз для практичного використання
    analyzer = KernelLagAnalyzer(
        model_types=["krr", "svr"],  # Найбільш практичні
        lag_range=range(2, 12),  # Практичний діапазон
        output_dir="realistic_process_analysis"
    )
    
    # Реалістичні параметри процесу
    realistic_params = {
        'N_data': 10000,  # Багато історичних даних
        'control_pts': 1000,
        'train_size': 0.75,
        'val_size': 0.15,
        'test_size': 0.10,
        'time_step_s': 5,  # 5-секундний інтервал вимірювань
        'enable_nonlinear': True,
        'nonlinear_config': {
            'concentrate_fe_percent': ('pow', 1.8),  # Реалістична нелінійність
            'concentrate_mass_flow': ('pow', 1.4)
        },
        'noise_level': 'low',  # Реальний шум процесу
        'seed': 42
    }
    
    # Запуск аналізу
    results = analyzer.run_lag_analysis(realistic_params)
    
    # Створення візуалізацій
    analyzer.plot_lag_analysis()
    analyzer.plot_detailed_comparison()
    
    # Звіт
    analyzer.save_results("realistic_process_results.json")
    report_path = analyzer.generate_report()
    
    print(f"\n📊 АНАЛІЗ РЕАЛІСТИЧНОГО ПРОЦЕСУ:")
    for model_type in analyzer.model_types:
        if model_type in analyzer.results['model_comparison']:
            comp = analyzer.results['model_comparison'][model_type]
            print(f"   {model_type.upper()}: оптимальний lag={comp['best_lag']}, RMSE={comp['best_rmse']:.6f}")
    
    return analyzer

def example_5_quick_lag_screening():
    """Приклад 5: Швидкий скринінг оптимальних лагів"""
    print("\n=== ПРИКЛАД 5: ШВИДКИЙ СКРИНІНГ ЛАГІВ ===\n")
    
    # Швидкий тест для попереднього вибору
    analyzer = KernelLagAnalyzer(
        model_types=["krr"],  # Тільки одна модель для швидкості
        lag_range=[2, 4, 6, 8, 10],  # Вибіркові лаги
        output_dir="quick_lag_screening"
    )
    
    # ВИПРАВЛЕНІ параметри з усіма обов'язковими ключами
    quick_params = {
        'N_data': 3000,
        'control_pts': 300,
        'train_size': 0.8,
        'val_size': 0.1,
        'test_size': 0.1,
        'time_step_s': 5,  # ДОДАНО
        'time_constants_s': {  # ДОДАНО
            'concentrate_fe_percent': 8.0,
            'tailings_fe_percent': 10.0,
            'concentrate_mass_flow': 5.0,
            'tailings_mass_flow': 7.0
        },
        'dead_times_s': {  # ДОДАНО
            'concentrate_fe_percent': 20.0,
            'tailings_fe_percent': 25.0,
            'concentrate_mass_flow': 20.0,
            'tailings_mass_flow': 25.0
        },
        'plant_model_type': 'rf',  # ДОДАНО
        'n_neighbors': 5,
        'enable_nonlinear': False,  # Без нелінійності для швидкості
        'use_anomalies': False,
        'seed': 123
    }
    
    results = analyzer.run_lag_analysis(quick_params)
    analyzer.plot_lag_analysis()
    
    print(f"Швидкий скринінг завершено. Рекомендований lag: {analyzer.results['best_lags']['krr']}")
    
    return analyzer

def run_all_kernel_examples():
    """Запуск всіх прикладів KernelLagAnalyzer"""
    print("🚀 ЗАПУСК ВСІХ ПРИКЛАДІВ KERNELLAGANALYZER З БАЗОВИМ ДАТАСЕТОМ")
    print("="*65)
    
    examples = [
        example_1_basic_kernel_lag_analysis,
        example_2_compare_all_kernel_models,
        example_3_focus_on_krr_optimization,
        example_4_realistic_process_conditions,
        example_5_quick_lag_screening
    ]
    
    results = []
    
    for i, example in enumerate(examples, 1):
        try:
            print(f"\n📊 Приклад {i}:")
            analyzer = example()
            results.append(analyzer)
            print(f"✅ Приклад {i} завершено успішно")
        except Exception as e:
            print(f"❌ Помилка в прикладі {i}: {e}")
            # import traceback
            # traceback.print_exc()
            continue
    
    print(f"\n🎉 ЗАВЕРШЕНО {len(results)} прикладів з {len(examples)}")
    
    # Порівняльний аналіз результатів
    if results:
        print("\n📊 ПОРІВНЯЛЬНИЙ АНАЛІЗ РЕЗУЛЬТАТІВ:")
        for i, analyzer in enumerate(results, 1):
            if hasattr(analyzer, 'results') and analyzer.results.get('best_lags'):
                print(f"   Приклад {i}:")
                for model, lag in analyzer.results['best_lags'].items():
                    rmse = analyzer.results['model_comparison'][model]['best_rmse']
                    print(f"     {model.upper()}: lag={lag}, RMSE={rmse:.6f}")
    
    return results

def example_krr_svr():
    """Приклад 4: KRR vs SVR оптимізований"""
    print("\n=== ПРИКЛАД KRR vs SVR ===\n")
    
    analyzer = KernelLagAnalyzer(
        model_types=["krr", "svr"],
        lag_range=[2, 4, 6, 8, 10, 12],
        output_dir="evaluation_results/lags"
    )
    
    realistic_params = {
        'N_data': 7000,
        'control_pts': 700,
        'train_size': 0.75,
        'val_size': 0.15,
        'test_size': 0.10,
        'time_step_s': 5,
        'time_constants_s': {
            'concentrate_fe_percent': 8.0,
            'tailings_fe_percent': 10.0,
            'concentrate_mass_flow': 5.0,
            'tailings_mass_flow': 7.0
        },
        'dead_times_s': {  # ДОДАНО
            'concentrate_fe_percent': 20.0,
            'tailings_fe_percent': 25.0,
            'concentrate_mass_flow': 20.0,
            'tailings_mass_flow': 25.0
        },
        'plant_model_type': 'rf',  # ДОДАНО
        'n_neighbors': 5,  # ДОДАНО
        'enable_nonlinear': True,
        'nonlinear_config': {
            'concentrate_fe_percent': ('pow', 1.8),
            'concentrate_mass_flow': ('pow', 1.4)
        },
        'noise_level': 'low',
        'use_anomalies': False,  # ДОДАНО для чистоти
        'seed': 42
    }
    
    # Решта коду залишається без змін
    results = analyzer.run_lag_analysis(realistic_params)
    analyzer.plot_lag_analysis()
    analyzer.plot_detailed_comparison()
    analyzer.save_results("krr_svr_results.json")
    report_path = analyzer.generate_report()
    
    print(f"\n📊 АНАЛІЗ KRR vs SVR:")
    for model_type in analyzer.model_types:
        if model_type in analyzer.results['model_comparison']:
            comp = analyzer.results['model_comparison'][model_type]
            print(f"   {model_type.upper()}: lag={comp['best_lag']}, RMSE={comp['best_rmse']:.6f}")
    
    return analyzer
if __name__ == "__main__":
    example_krr_svr()
    
    """Приклад 1: Базовий аналіз лагів для ядерних моделей"""
    # example_1_basic_kernel_lag_analysis()
    
    """Приклад 2: Порівняння всіх ядерних моделей (KRR, SVR, GPR)"""
    # example_2_compare_all_kernel_models
    
    """Приклад 3: Фокус на оптимізації KRR для різних лагів"""
    # example_3_focus_on_krr_optimization()
    
    """Приклад 4: Реалістичні умови процесу магнітної сепарації"""
    # example_4_realistic_process_conditions()
        
    """Приклад 5: Швидкий скринінг оптимальних лагів"""
    # example_5_quick_lag_screening()
    
    
    """Запуск всіх прикладів KernelLagAnalyzer"""
    # run_all_kernel_examples()
