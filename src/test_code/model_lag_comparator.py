# model_lag_comparator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path
from model import KernelModel
from data_gen import StatefulDataGenerator

class ModelLagComparator:
    """
    Мінімальний клас для порівняння ядерних моделей (KRR, SVR) 
    з різною кількістю лагів для підтвердження впливу кількості 
    лагів на параметри моделі.
    """
    
    def __init__(self, 
                 reference_df: pd.DataFrame,
                 model_types: List[str] = ["krr", "svr"], 
                 lag_range: List[int] = [2, 4, 6, 8],
                 output_dir: Optional[str] = None):
        """
        Ініціалізація компаратора моделей з різною кількістю лагів.
        
        Args:
            reference_df: Референтний датасет для аналізу
            model_types: Список типів ядерних моделей для тестування
            lag_range: Список значень лагів для тестування
            output_dir: Директорія для збереження результатів
        """
        self.reference_df = reference_df
        self.model_types = model_types
        self.lag_range = lag_range
        self.results = {}
        
        # Створення структури директорій
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = f"lag_comparison/{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Ініціалізовано ModelLagComparator")
        print(f"   Моделі: {model_types}")
        print(f"   Лаги: {lag_range}")
    
    def create_lagged_matrices(self, df: pd.DataFrame, lag: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Створення лагових матриць на основі підходу з ModelComparisonService.
        
        Args:
            df: DataFrame з даними
            lag: Кількість лагів
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Матриці X (входи) та Y (виходи)
        """
        # Виведемо інформацію про наявні колонки
        print(f"Доступні колонки: {list(df.columns)}")
        
        # Перевірка альтернативних назв змінних
        if 'feed_fe_percent' in df.columns:
            input_vars = ['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']
        else:
            # Якщо стандартних назв немає, спробуємо знайти аналогічні колонки
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            print(f"Числові колонки: {numeric_cols}")
            
            # Для демонстрації використаємо перші числові колонки як вхідні
            input_vars = numeric_cols[:3]
            print(f"Використовуємо як вхідні змінні: {input_vars}")
        
        # Перевірка цільових змінних
        if 'concentrate_fe' in df.columns and 'concentrate_mass' in df.columns:
            output_vars = ['concentrate_fe', 'concentrate_mass']
        elif 'concentrate_fe_percent' in df.columns and 'concentrate_mass_flow' in df.columns:
            output_vars = ['concentrate_fe_percent', 'concentrate_mass_flow']
        else:
            # Для демонстрації використаємо дві наступні числові колонки як вихідні
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 3:
                output_vars = numeric_cols[3:5]
            else:
                # Якщо колонок менше, використаємо перші дві для входу і виходу
                output_vars = numeric_cols[:2]
            print(f"Використовуємо як вихідні змінні: {output_vars}")
        
        # Перевірка чи існують необхідні колонки
        for var_list, var_type in [(input_vars, "вхідні"), (output_vars, "вихідні")]:
            missing = [var for var in var_list if var not in df.columns]
            if missing:
                print(f"⚠️ Відсутні {var_type} колонки: {missing}")
                print(f"📋 Спроба використати StatefulDataGenerator.create_lagged_dataset")
                try:
                    from data_gen import StatefulDataGenerator
                    return StatefulDataGenerator.create_lagged_dataset(df, lags=lag)
                except (ImportError, AttributeError):
                    print("❌ Не вдалося викликати StatefulDataGenerator.create_lagged_dataset")
                    # Використаємо альтернативний підхід
                    return self._create_lagged_matrices_fallback(df, lag)
        
        # Створення матриць з лагами
        n = len(df)
        X, Y = [], []
        
        print(f"Створення лагових матриць з {len(input_vars)} вхідними і {len(output_vars)} вихідними змінними, lag={lag}")
        
        for i in range(lag, n):
            row = []
            for var in input_vars:
                for j in range(lag + 1):
                    row.append(df[var].iloc[i - j])
            X.append(row)
            Y.append([df[var].iloc[i] for var in output_vars])
        
        X_array = np.array(X)
        Y_array = np.array(Y)
        
        print(f"Створено матриці: X shape={X_array.shape}, Y shape={Y_array.shape}")
        
        return X_array, Y_array

    def _create_lagged_matrices_fallback(self, df: pd.DataFrame, lag: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Резервний метод створення лагових матриць, якщо основний метод не працює.
        
        Args:
            df: DataFrame з даними
            lag: Кількість лагів
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Матриці X (входи) та Y (виходи)
        """
        # Виберемо всі числові колонки
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 3:
            raise ValueError(f"Недостатньо числових колонок у датафреймі. Знайдено тільки {len(numeric_cols)}")
        
        # Розділимо колонки на вхідні та вихідні
        input_cols = numeric_cols[:-2]  # Всі крім останніх двох
        output_cols = numeric_cols[-2:]  # Останні дві
        
        print(f"Використання резервного методу з колонками:")
        print(f"Вхідні: {input_cols}")
        print(f"Вихідні: {output_cols}")
        
        n = len(df)
        X, Y = [], []
        
        for i in range(lag, n):
            row = []
            for col in input_cols:
                for j in range(lag + 1):
                    row.append(df[col].iloc[i - j])
            X.append(row)
            Y.append([df[col].iloc[i] for col in output_cols])
        
        return np.array(X), np.array(Y)
    
    def create_simulation_data(self, params: dict) -> Tuple[StatefulDataGenerator, pd.DataFrame]:
        """
        Створення симуляційних даних через StatefulDataGenerator.
        Аналогічно методу в ModelComparisonService.
        
        Args:
            params: Параметри симуляції
            
        Returns:
            Tuple[StatefulDataGenerator, pd.DataFrame]: Генератор та симуляційні дані
        """
        print(f"📊 Генерація симуляційних даних з параметрами:")
        print(f"   N_data: {params.get('N_data', 5000)}")
        print(f"   control_pts: {params.get('control_pts', 500)}")
        print(f"   lag: {params.get('lag', 2)}")
        print(f"   enable_nonlinear: {params.get('enable_nonlinear', False)}")
        
        # Створення генератора даних
        true_gen = StatefulDataGenerator(
            self.reference_df,
            ore_flow_var_pct=3.0,
            time_step_s=params.get('time_step_s', 5),
            time_constants_s=params.get('time_constants_s', {
                'concentrate_fe_percent': 8.0,
                'tailings_fe_percent': 10.0,
                'concentrate_mass_flow': 5.0,
                'tailings_mass_flow': 7.0
            }),
            dead_times_s=params.get('dead_times_s', {
                'concentrate_fe_percent': 20.0,
                'tailings_fe_percent': 25.0,
                'concentrate_mass_flow': 20.0,
                'tailings_mass_flow': 25.0
            }),
            true_model_type=params.get('plant_model_type', 'rf'),
            seed=params.get('seed', 42)
        )
    
        # Налаштування аномалій (якщо потрібно)
        anomaly_cfg = None
        if params.get('use_anomalies', False):
            try:
                anomaly_cfg = self._create_anomaly_config(
                    N_data=params.get('N_data', 5000),
                    train_frac=params.get('train_size', 0.8),
                    val_frac=params.get('val_size', 0.1),
                    test_frac=params.get('test_size', 0.1),
                    seed=params.get('seed', 42),
                    severity=params.get('anomaly_severity', 'mild')
                )
            except Exception as e:
                print(f"⚠️ Помилка при створенні конфігурації аномалій: {str(e)}")
                print("   Аномалії не будуть додані")
    
        # Генерація базових даних
        df_true_orig = true_gen.generate(
            T=params.get('N_data', 5000),
            control_pts=params.get('control_pts', 500),
            n_neighbors=params.get('n_neighbors', 5),
            noise_level=params.get('noise_level', 'none'),
            anomaly_config=anomaly_cfg
        )
    
        # Генерація нелінійного варіанту (якщо вказано)
        if params.get('enable_nonlinear', False):
            if 'nonlinear_config' in params:
                print(f"   Створення нелінійного варіанту даних")
                df_true = true_gen.generate_nonlinear_variant(
                    base_df=df_true_orig,
                    non_linear_factors=params['nonlinear_config'],
                    noise_level='none',
                    anomaly_config=anomaly_cfg
                )
            else:
                print("⚠️ Не вказана конфігурація нелінійності (nonlinear_config)")
                print("   Використовуємо лінійні дані")
                df_true = df_true_orig
        else:
            df_true = df_true_orig
    
        print(f"✅ Згенеровано датасет: {len(df_true)} записів")
        return true_gen, df_true
    
    def _create_anomaly_config(self, N_data: int, train_frac: float = 0.7, 
                             val_frac: float = 0.15, test_frac: float = 0.15,
                             seed: int = 42, severity: str = "mild") -> dict:
        """
        Створення конфігурації аномалій для симуляційних даних.
        
        Args:
            N_data: Кількість точок даних
            train_frac: Частка тренувальних даних
            val_frac: Частка валідаційних даних
            test_frac: Частка тестових даних
            seed: Випадкове зерно
            severity: Рівень аномалій ('mild', 'medium', 'strong')
            
        Returns:
            dict: Конфігурація аномалій
        """
        try:
            # Імпорт методу з оригінального коду, якщо доступний
            from data_gen import StatefulDataGenerator
            if hasattr(StatefulDataGenerator, 'generate_anomaly_config'):
                return StatefulDataGenerator.generate_anomaly_config(
                    N_data=N_data,
                    train_frac=train_frac,
                    val_frac=val_frac,
                    test_frac=test_frac,
                    seed=seed,
                    severity=severity
                )
        except (ImportError, AttributeError):
            pass
        
        # Базова реалізація, якщо оригінальний метод недоступний
        # Це спрощена версія, яка відповідає структурі методу з ModelComparisonService
        
        # Кількість записів у кожному наборі
        n_train = int(N_data * train_frac)
        n_val = int(N_data * val_frac)
        n_test = N_data - n_train - n_val
        
        # Налаштування параметрів аномалій на основі рівня
        if severity == 'mild':
            prob = 0.02
            max_duration = 5
        elif severity == 'medium':
            prob = 0.05
            max_duration = 10
        elif severity == 'strong':
            prob = 0.08
            max_duration = 15
        else:
            prob = 0.03
            max_duration = 7
        
        # Створюємо базову конфігурацію
        np.random.seed(seed)
        
        # Створюємо позиції аномалій (для демонстрації)
        anomaly_positions = []
        for i in range(n_val + n_test):
            if np.random.random() < prob:
                duration = np.random.randint(1, max_duration + 1)
                start_pos = n_train + i
                anomaly_positions.append((start_pos, duration))
        
        return {
            'positions': anomaly_positions,
            'severity': severity,
            'probability': prob,
            'max_duration': max_duration,
            'in_train': False,  # Аномалії лише в val/test наборах
            'in_val': True,
            'in_test': True
        }

    def train_and_evaluate_models(self, X: np.ndarray, Y: np.ndarray, 
                                   lag: int, train_ratio: float = 0.8) -> Dict:
        """
        Навчання та оцінка моделей для заданого набору даних з лагами.
        
        Args:
            X: Вхідні дані з лагами
            Y: Вихідні дані
            lag: Поточна кількість лагів (для звітності)
            train_ratio: Частка тренувальних даних
            
        Returns:
            Dict: Результати оцінки моделей
        """
        # Розділення на тренувальний та тестовий набори
        n = X.shape[0]
        n_train = int(train_ratio * n)
        
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]
        
        # Масштабування даних
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)
        Y_train_scaled = y_scaler.fit_transform(Y_train)
        
        # Результати для поточного лагу
        lag_results = {}
        
        # Тестування для кожного типу моделі
        for model_type in self.model_types:
            # Імпорт KernelModel з проекту
            from model import KernelModel
            
            # Ініціалізація моделі
            print(f"   Навчання моделі {model_type.upper()} з lag={lag}...")
            
            try:
                # Створення моделі через фасад KernelModel
                model = KernelModel(
                    model_type=model_type, 
                    kernel="rbf", 
                    find_optimal_params=True
                )
                
                # Навчання моделі
                start_time = datetime.now()
                model.fit(X_train_scaled, Y_train_scaled)
                train_time = (datetime.now() - start_time).total_seconds()
                
                # Прогнозування
                start_time = datetime.now()
                Y_pred_scaled = model.predict(X_test_scaled)
                predict_time = (datetime.now() - start_time).total_seconds()
                
                # Перетворення назад до оригінального масштабу
                Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
                
                # Метрики оцінки
                mse = mean_squared_error(Y_test, Y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(Y_test, Y_pred)
                
                # Метрики по окремих цільових змінних
                rmse_per_target = []
                r2_per_target = []
                
                for i in range(Y_test.shape[1]):
                    rmse_target = np.sqrt(mean_squared_error(Y_test[:, i], Y_pred[:, i]))
                    r2_target = r2_score(Y_test[:, i], Y_pred[:, i])
                    rmse_per_target.append(rmse_target)
                    r2_per_target.append(r2_target)
                
                # Отримання параметрів ядра через _impl (внутрішня реалізація)
                kernel_params = {}
                
                # Доступ до реалізації через _impl
                impl = model._impl
                
                # Для KRR
                if model_type == "krr":
                    if hasattr(impl, 'alpha'):
                        kernel_params['alpha'] = impl.alpha
                    if hasattr(impl, 'gamma'):
                        kernel_params['gamma'] = impl.gamma
                    if hasattr(impl, 'model') and hasattr(impl.model, 'alpha'):
                        kernel_params['alpha'] = impl.model.alpha
                    if hasattr(impl, 'model') and hasattr(impl.model, 'gamma'):
                        kernel_params['gamma'] = impl.model.gamma
                
                # Для SVR
                elif model_type == "svr":
                    if hasattr(impl, 'C'):
                        kernel_params['C'] = impl.C
                    if hasattr(impl, 'epsilon'):
                        kernel_params['epsilon'] = impl.epsilon
                    if hasattr(impl, 'gamma'):
                        kernel_params['gamma'] = impl.gamma
                    
                    # Перевірка на наявність моделей (для багатовимірних виходів)
                    if hasattr(impl, 'models') and len(impl.models) > 0:
                        first_model = impl.models[0]
                        if hasattr(first_model, 'C'):
                            kernel_params['C'] = first_model.C
                        if hasattr(first_model, 'epsilon'):
                            kernel_params['epsilon'] = first_model.epsilon
                        if hasattr(first_model, 'gamma'):
                            kernel_params['gamma'] = first_model.gamma
                        if hasattr(first_model, '_actual_gamma'):
                            kernel_params['gamma'] = first_model._actual_gamma
                
                # Збереження результатів
                lag_results[model_type] = {
                    'lag': lag,
                    'rmse': rmse,
                    'r2': r2,
                    'rmse_per_target': rmse_per_target,
                    'r2_per_target': r2_per_target,
                    'train_time': train_time,
                    'predict_time': predict_time,
                    'kernel_params': kernel_params,
                    'Y_pred': Y_pred.tolist()  # Зберігаємо для подальшого аналізу
                }
                
                print(f"   {model_type.upper()}: RMSE={rmse:.4f}, R²={r2:.4f}, Час навчання={train_time:.2f}с")
                
                # Виведення інформації про параметри ядра
                print(f"   Параметри ядра для {model_type.upper()}:")
                for param, value in kernel_params.items():
                    print(f"      {param}: {value}")
            
            except Exception as e:
                print(f"   ❌ Помилка навчання моделі {model_type.upper()}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return lag_results    
    
    def run_comparison(self, **kwargs) -> Dict:
        """
        Запуск порівняння моделей з різною кількістю лагів.
        
        Args:
            **kwargs: Параметри симуляції, які можуть включати:
                - N_data: Кількість точок даних для симуляції
                - control_pts: Кількість контрольних точок
                - time_step_s: Крок часу в секундах
                - train_size: Частка тренувальних даних
                - val_size: Частка валідаційних даних
                - test_size: Частка тестових даних
                - enable_nonlinear: Чи використовувати нелінійність
                - nonlinear_config: Конфігурація нелінійності
                - noise_level: Рівень шуму
                - use_anomalies: Чи додавати аномалії
                - anomaly_severity: Рівень аномалій
                - seed: Випадкове зерно
            
        Returns:
            Dict: Результати порівняння
        """
        print(f"📊 Початок порівняння моделей з різною кількістю лагів...")
        
        # Базові параметри симуляції
        base_params = {
            'N_data': kwargs.get('N_data', 5000),
            'control_pts': kwargs.get('control_pts', 500),
            'train_size': kwargs.get('train_size', 0.8),
            'val_size': kwargs.get('val_size', 0.1),
            'test_size': kwargs.get('test_size', 0.1),
            'time_step_s': kwargs.get('time_step_s', 5),
            'time_constants_s': kwargs.get('time_constants_s', {
                'concentrate_fe_percent': 8.0,
                'tailings_fe_percent': 10.0,
                'concentrate_mass_flow': 5.0,
                'tailings_mass_flow': 7.0
            }),
            'dead_times_s': kwargs.get('dead_times_s', {
                'concentrate_fe_percent': 20.0,
                'tailings_fe_percent': 25.0,
                'concentrate_mass_flow': 20.0,
                'tailings_mass_flow': 25.0
            }),
            'plant_model_type': kwargs.get('plant_model_type', 'rf'),
            'n_neighbors': kwargs.get('n_neighbors', 5),
            'noise_level': kwargs.get('noise_level', 'none'),
            'enable_nonlinear': kwargs.get('enable_nonlinear', False),
            'nonlinear_config': kwargs.get('nonlinear_config', {
                'concentrate_fe_percent': ('pow', 2.0),
                'concentrate_mass_flow': ('pow', 1.5)
            }),
            'use_anomalies': kwargs.get('use_anomalies', False),
            'anomaly_severity': kwargs.get('anomaly_severity', 'mild'),
            'seed': kwargs.get('seed', 42)
        }
        
        # Аналіз структури даних
        print("\nАналіз структури вхідних даних:")
        print(f"Розмірність референтного датафрейму: {self.reference_df.shape}")
        print(f"Колонки: {self.reference_df.columns.tolist()}")
        print("\n")
        
        # Зберігаємо результати
        results = {
            'base_params': base_params,
            'lag_results': {},
            'best_lags': {},
            'model_comparison': {}
        }
        
        # Визначаємо, чи використовувати симуляцію або реальні дані
        use_simulation = kwargs.get('use_simulation', True)
        
        if use_simulation:
            print("🔄 Використовуємо симуляційні дані")
            # Створюємо симуляційні дані один раз
            sim_params = base_params.copy()
            true_gen, df_sim = self.create_simulation_data(sim_params)
            print(f"   Створено симуляційний датасет: {len(df_sim)} записів")
        else:
            print("🔄 Використовуємо реальні дані з референтного датафрейму")
            df_sim = self.reference_df
        
        # Цикл по різним значенням лагів
        for lag in self.lag_range:
            print(f"\n🔄 Аналіз lag={lag}")
            
            try:
                # Створення лагових матриць з даних (симуляційних або реальних)
                X, Y = self.create_lagged_matrices(df_sim, lag)
                
                if X.shape[0] == 0:
                    print(f"   ⚠️ Недостатньо даних для lag={lag}")
                    continue
                
                print(f"   Розмірність: X{X.shape}, Y{Y.shape}")
                
                # Навчання та оцінка моделей
                lag_results = self.train_and_evaluate_models(X, Y, lag, train_ratio=base_params['train_size'])
                
                # Збереження результатів для поточного лагу
                results['lag_results'][lag] = lag_results
                
            except Exception as e:
                print(f"   ❌ Помилка при аналізі lag={lag}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Перевірка, чи є результати
        if not results['lag_results']:
            print("❌ Немає результатів аналізу. Перевірте дані та налаштування.")
            return results
        
        # Знаходження найкращих лагів для кожної моделі
        for model_type in self.model_types:
            best_lag = self._find_best_lag(results['lag_results'], model_type)
            if best_lag is not None:
                results['best_lags'][model_type] = best_lag
                print(f"\n✅ Найкращий lag для {model_type.upper()}: {best_lag}")
            else:
                print(f"\n⚠️ Не вдалося визначити найкращий lag для {model_type.upper()}")
        
        # Порівняння моделей на оптимальних лагах
        results['model_comparison'] = self._compare_models(results['lag_results'])
        
        self.results = results
        return results
     
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
        """Порівняння моделей на кращих лагах"""
        comparison = {}
        
        for model_type in self.model_types:
            best_lag = self._find_best_lag(lag_results, model_type)
            if best_lag and best_lag in lag_results:
                if model_type in lag_results[best_lag]:
                    comparison[model_type] = {
                        'best_lag': best_lag,
                        'metrics': lag_results[best_lag][model_type]
                    }
        
        return comparison
    
    def plot_results(self) -> None:
        """
        Компактна візуалізація результатів порівняння моделей з різною кількістю лагів.
        """
        if not self.results:
            raise ValueError("Немає результатів для візуалізації. Спочатку запустіть run_comparison()")
        
        # Створення компактної фігури
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)
        
        # Колірна схема
        colors = {'krr': 'blue', 'svr': 'red', 'gpr': 'green'}
        
        # Графік 1: RMSE vs Lag
        ax = axes[0, 0]
        for model_type in self.model_types:
            lags = []
            rmse_values = []
            
            for lag in self.lag_range:
                if lag in self.results['lag_results']:
                    if model_type in self.results['lag_results'][lag]:
                        lags.append(lag)
                        rmse_values.append(self.results['lag_results'][lag][model_type]['rmse'])
            
            if lags:
                ax.plot(lags, rmse_values, 
                       marker='o', linewidth=1.5, markersize=4,
                       color=colors.get(model_type, 'black'),
                       label=f'{model_type.upper()}')
                
                # Позначення найкращого лагу зірочкою без підписів
                best_lag = self.results['best_lags'].get(model_type)
                if best_lag in lags:
                    best_idx = lags.index(best_lag)
                    ax.scatter(best_lag, rmse_values[best_idx], 
                               s=80, color=colors.get(model_type), 
                               marker='*', zorder=5)
        
        ax.set_xlabel('Лаг')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        
        # Графік 2: R² vs Lag
        ax = axes[0, 1]
        for model_type in self.model_types:
            lags = []
            r2_values = []
            
            for lag in self.lag_range:
                if lag in self.results['lag_results']:
                    if model_type in self.results['lag_results'][lag]:
                        if 'r2' in self.results['lag_results'][lag][model_type]:
                            lags.append(lag)
                            r2_values.append(self.results['lag_results'][lag][model_type]['r2'])
            
            if lags:
                ax.plot(lags, r2_values, 
                       marker='o', linewidth=1.5, markersize=4,
                       color=colors.get(model_type, 'black'),
                       label=f'{model_type.upper()}')
                
                # Позначення найкращого лагу зірочкою без підписів
                best_lag = self.results['best_lags'].get(model_type)
                if best_lag in lags:
                    best_idx = lags.index(best_lag)
                    ax.scatter(best_lag, r2_values[best_idx], 
                               s=80, color=colors.get(model_type), 
                               marker='*', zorder=5)
        
        ax.set_xlabel('Лаг')
        ax.set_ylabel('R²')
        ax.set_title('Коефіцієнт детермінації')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        
        # Графік 3: Час навчання vs Lag
        ax = axes[1, 0]
        for model_type in self.model_types:
            lags = []
            times = []
            
            for lag in self.lag_range:
                if lag in self.results['lag_results']:
                    if model_type in self.results['lag_results'][lag]:
                        if 'train_time' in self.results['lag_results'][lag][model_type]:
                            lags.append(lag)
                            times.append(self.results['lag_results'][lag][model_type]['train_time'])
            
            if lags:
                ax.plot(lags, times, 
                       marker='o', linewidth=1.5, markersize=4,
                       color=colors.get(model_type, 'black'),
                       label=f'{model_type.upper()}')
        
        ax.set_xlabel('Лаг')
        ax.set_ylabel('Час (с)')
        ax.set_title('Час навчання')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        
        # Графік 4: Параметри ядра vs Lag
        ax = axes[1, 1]
        for model_type in self.model_types:
            # Знаходимо параметр для відображення
            param_to_plot = None
            if model_type == 'krr':
                param_to_plot = 'alpha'
            elif model_type == 'svr':
                param_to_plot = 'C'
            else:
                param_to_plot = 'gamma'
                
            lags = []
            param_values = []
            
            for lag in self.lag_range:
                if lag in self.results['lag_results']:
                    if model_type in self.results['lag_results'][lag]:
                        kernel_params = self.results['lag_results'][lag][model_type].get('kernel_params', {})
                        if param_to_plot in kernel_params:
                            lags.append(lag)
                            param_values.append(kernel_params[param_to_plot])
            
            if lags:
                ax.plot(lags, param_values, 
                       marker='o', linewidth=1.5, markersize=4,
                       color=colors.get(model_type, 'black'),
                       label=f'{model_type.upper()} - {param_to_plot}')
        
        ax.set_xlabel('Лаг')
        ax.set_ylabel('Значення')
        ax.set_title('Параметри ядра')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        
        # Логарифмічна шкала для значень, якщо розкид великий
        if ax.get_ylim()[1] / max(1e-10, ax.get_ylim()[0]) > 100:
            ax.set_yscale('log')
        
        # Загальний заголовок
        fig.suptitle('Вплив кількості лагів на параметри моделей', fontsize=12)
        
        plt.tight_layout()
        
        # Збереження
        plot_path = self.output_dir / 'lag_comparison_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Результати порівняння збережено: {plot_path}")  
        
    def _plot_metric_vs_lag(self, ax, metric_name, metric_label, colors, lower_is_better=True):
        """Допоміжний метод для відображення метрики від лагу"""
        for model_type in self.model_types:
            lags = []
            metric_values = []
            
            for lag in self.lag_range:
                if lag in self.results['lag_results']:
                    if model_type in self.results['lag_results'][lag]:
                        lags.append(lag)
                        metric_values.append(self.results['lag_results'][lag][model_type][metric_name])
            
            if lags:
                ax.plot(lags, metric_values, 
                       marker='o', linewidth=2, markersize=6,
                       color=colors.get(model_type, 'black'),
                       label=f'{model_type.upper()}')
                
                # Позначення найкращого лагу
                best_lag = self.results['best_lags'].get(model_type)
                if best_lag in lags:
                    best_idx = lags.index(best_lag)
                    ax.scatter(best_lag, metric_values[best_idx], 
                              s=100, color=colors.get(model_type), 
                              marker='*', zorder=5)
                    ax.annotate(f'Оптимальний lag={best_lag}', 
                               xy=(best_lag, metric_values[best_idx]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=9, alpha=0.8)
        
        ax.set_xlabel('Кількість лагів')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} в залежності від кількості лагів')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_kernel_params(self, ax, colors):
        """Відображення зміни параметрів ядра в залежності від лагу"""
        for model_type in self.model_types:
            lags = []
            
            # Знаходимо спільні параметри для всіх лагів цієї моделі
            common_params = set()
            first = True
            
            for lag in self.lag_range:
                if lag in self.results['lag_results']:
                    if model_type in self.results['lag_results'][lag]:
                        kernel_params = self.results['lag_results'][lag][model_type].get('kernel_params', {})
                        if first:
                            common_params = set(kernel_params.keys())
                            first = False
                        else:
                            common_params &= set(kernel_params.keys())
            
            if not common_params:
                continue
                
            # Для KRR і SVR виберемо найважливіші параметри
            param_to_plot = None
            if model_type == 'krr' and 'alpha' in common_params:
                param_to_plot = 'alpha'
            elif model_type == 'svr' and 'C' in common_params:
                param_to_plot = 'C'
            elif 'gamma' in common_params:
                param_to_plot = 'gamma'
            
            if not param_to_plot:
                continue
            
            # Збираємо значення параметра для різних лагів
            param_values = []
            lags = []
            
            for lag in self.lag_range:
                if lag in self.results['lag_results']:
                    if model_type in self.results['lag_results'][lag]:
                        kernel_params = self.results['lag_results'][lag][model_type].get('kernel_params', {})
                        if param_to_plot in kernel_params:
                            lags.append(lag)
                            param_values.append(kernel_params[param_to_plot])
            
            if lags:
                ax.plot(lags, param_values, 
                       marker='o', linewidth=2, markersize=6,
                       color=colors.get(model_type, 'black'),
                       label=f'{model_type.upper()} - {param_to_plot}')
        
        ax.set_xlabel('Кількість лагів')
        ax.set_ylabel('Значення параметру')
        ax.set_title('Зміна параметрів ядра в залежності від лагів')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Логарифмічна шкала для значень, якщо розкид великий
        if ax.get_ylim()[1] / max(1e-10, ax.get_ylim()[0]) > 100:
            ax.set_yscale('log')
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Збереження результатів порівняння у JSON.
        
        Args:
            filename: Назва файлу для збереження
            
        Returns:
            str: Шлях до збереженого файлу
        """
        if not self.results:
            raise ValueError("Немає результатів для збереження")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'lag_comparison_{timestamp}.json'
        
        filepath = self.output_dir / filename
        
        # Перетворення результатів у формат, який можна серіалізувати
        serializable_results = {}
        
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        serializable_results[key][k] = {}
                        for kk, vv in v.items():
                            if isinstance(vv, dict):
                                serializable_results[key][k][kk] = {}
                                for kkk, vvv in vv.items():
                                    if isinstance(vvv, np.ndarray):
                                        serializable_results[key][k][kk][kkk] = vvv.tolist()
                                    elif isinstance(vvv, (np.int64, np.float64)):
                                        serializable_results[key][k][kk][kkk] = float(vvv)
                                    else:
                                        serializable_results[key][k][kk][kkk] = vvv
                            elif isinstance(vv, np.ndarray):
                                serializable_results[key][k][kk] = vv.tolist()
                            elif isinstance(vv, (np.int64, np.float64)):
                                serializable_results[key][k][kk] = float(vv)
                            else:
                                serializable_results[key][k][kk] = vv
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Результати збережено: {filepath}")
        return str(filepath)
    
    def generate_report(self) -> str:
        """
        Генерація текстового звіту з результатами порівняння.
        
        Returns:
            str: Текст звіту
        """
        if not self.results:
            raise ValueError("Немає результатів для звіту")
        
        report = f"""
ЗВІТ ПРО ВПЛИВ КІЛЬКОСТІ ЛАГІВ НА ПАРАМЕТРИ МОДЕЛЕЙ
{'='*55}

ПАРАМЕТРИ АНАЛІЗУ:
    Моделі: {', '.join([m.upper() for m in self.model_types])}
    Діапазон лагів: {self.lag_range}
    Дата аналізу: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

НАЙКРАЩІ РЕЗУЛЬТАТИ ЗА МОДЕЛЯМИ:
"""
        
        for model_type in self.model_types:
            if model_type in self.results['model_comparison']:
                comp_data = self.results['model_comparison'][model_type]
                metrics = comp_data['metrics']
                
                report += f"""
    {model_type.upper()}:
        Оптимальний lag: {comp_data['best_lag']}
        RMSE: {metrics['rmse']:.6f}
        R² Score: {metrics['r2']:.6f}
        Час навчання: {metrics['train_time']:.3f} с
"""
                # Додаємо інформацію про параметри ядра
                if 'kernel_params' in metrics:
                    report += f"        Параметри ядра:\n"
                    for param, value in metrics['kernel_params'].items():
                        report += f"            {param}: {value}\n"
        
        # Додаємо аналіз зміни параметрів
        report += f"\nАНАЛІЗ ЗМІНИ ПАРАМЕТРІВ В ЗАЛЕЖНОСТІ ВІД ЛАГУ:\n"
        
        for model_type in self.model_types:
            report += f"\n    {model_type.upper()}:\n"
            report += f"        {'Лаг':<6} {'RMSE':<10} {'R²':<10} {'Параметри ядра'}\n"
            report += f"        {'-'*50}\n"
            
            for lag in sorted(self.lag_range):
                if lag in self.results['lag_results']:
                    if model_type in self.results['lag_results'][lag]:
                        data = self.results['lag_results'][lag][model_type]
                        kernel_params = data.get('kernel_params', {})
                        
                        # Форматуємо параметри ядра
                        params_str = ", ".join([f"{k}={v:.6g}" for k, v in kernel_params.items()])
                        
                        report += f"        {lag:<6} {data['rmse']:<10.6f} {data['r2']:<10.6f} {params_str}\n"
        
        # Висновки
        report += f"\nВИСНОВКИ:\n"
        
        for model_type in self.model_types:
            if model_type in self.results['best_lags']:
                best_lag = self.results['best_lags'][model_type]
                
                # Аналіз зміни параметрів для цієї моделі
                params_trend = {}
                
                for param_name in ['alpha', 'C', 'gamma', 'epsilon']:
                    values = []
                    lags = []
                    
                    for lag in sorted(self.lag_range):
                        if lag in self.results['lag_results']:
                            if model_type in self.results['lag_results'][lag]:
                                kernel_params = self.results['lag_results'][lag][model_type].get('kernel_params', {})
                                if param_name in kernel_params:
                                    lags.append(lag)
                                    values.append(kernel_params[param_name])
                    
                    if values:
                        # Визначаємо тренд (зростання/спадання)
                        if len(values) >= 2:
                            if values[-1] > values[0]:
                                trend = "зростає"
                            elif values[-1] < values[0]:
                                trend = "спадає"
                            else:
                                trend = "не змінюється"
                            
                            params_trend[param_name] = trend
                
                # Додаємо висновки по моделі
                report += f"\n    {model_type.upper()}:\n"
                report += f"        Оптимальна кількість лагів: {best_lag}\n"
                
                # Додаємо висновки по параметрам
                for param, trend in params_trend.items():
                    report += f"        Параметр {param} {trend} зі збільшенням кількості лагів\n"
        
        report += f"\n{'='*55}\n"
        
        # Збереження звіту
        report_path = self.output_dir / 'lag_comparison_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📝 Звіт збережено: {report_path}")
        return report
        
# Приклад використання класу
# Приклад використання класу ModelLagComparator з симуляцією даних

def run_lag_comparison_example(df_path='processed.parquet', use_simulation=True):
    """
    Демонстраційний приклад використання класу ModelLagComparator
    для аналізу впливу кількості лагів на параметри моделей.
    
    Args:
        df_path: Шлях до референтного датасету
        use_simulation: Чи використовувати симуляцію даних
        
    Returns:
        ModelLagComparator: Екземпляр компаратора з результатами
    """
    print("=" * 70)
    print("ДЕМОНСТРАЦІЯ ВПЛИВУ КІЛЬКОСТІ ЛАГІВ НА ПАРАМЕТРИ МОДЕЛЕЙ")
    print("=" * 70)
    
    # Завантаження даних
    try:
        import pandas as pd
        from model_lag_comparator import ModelLagComparator
        
        df = pd.read_parquet(df_path)
        print(f"✅ Завантажено {len(df)} записів з {df_path}")
    except FileNotFoundError:
        print(f"❌ Файл '{df_path}' не знайдено")
        return None
    except Exception as e:
        print(f"❌ Помилка при завантаженні даних: {str(e)}")
        return None
    
    print("\nПочаткові дані:")
    print(df.head())
    print(f"Розмірність даних: {df.shape}")
    
    # Створення компаратора моделей
    lag_comparator = ModelLagComparator(
        reference_df=df,
        model_types=["krr", "svr", 'gpr'],  # Порівнюємо Kernel Ridge і Support Vector Regression
        lag_range=[2, 4, 6, 8, 10, 12, 14, 16],  # Різні значення лагів для аналізу
        output_dir="lag_comparison_results"
    )
    
    # Запуск порівняння з параметрами симуляції
    print("\n📊 Запуск порівняння моделей з різною кількістю лагів...")
    
    # Параметри симуляції (використовуються тільки якщо use_simulation=True)
    simulation_params = {
        'N_data': 3000,           # Кількість точок даних
        'control_pts': 300,       # Контрольні точки
        'train_size': 0.8,        # Частка тренувальних даних
        'val_size': 0.1,          # Частка валідаційних даних
        'test_size': 0.1,         # Частка тестових даних
        'time_step_s': 5,         # Крок часу в секундах
        'noise_level': 'none',     # Рівень шуму
        'enable_nonlinear': True, # Використовувати нелінійність
        'nonlinear_config': {     # Конфігурація нелінійності
            'concentrate_fe_percent': ('pow', 1.8),
            'concentrate_mass_flow': ('pow', 1.4)
        },
        'use_simulation': use_simulation,  # Чи використовувати симуляцію
        'seed': 42                # Випадкове зерно для відтворюваності
    }
    
    # Запуск порівняння
    results = lag_comparator.run_comparison(**simulation_params)
    
    # Візуалізація результатів
    print("\n📈 Створення візуалізацій...")
    try:
        lag_comparator.plot_results()
    except Exception as e:
        print(f"❌ Помилка при створенні візуалізацій: {str(e)}")
    
    # Генерація та вивід звіту
    print("\n📝 Генерація звіту...")
    try:
        report = lag_comparator.generate_report()
    except Exception as e:
        print(f"❌ Помилка при генерації звіту: {str(e)}")
    
    # Збереження результатів
    print("\n💾 Збереження результатів...")
    try:
        lag_comparator.save_results("lag_comparison_results.json")
    except Exception as e:
        print(f"❌ Помилка при збереженні результатів: {str(e)}")
    
    print("\n" + "=" * 70)
    print("ВИСНОВКИ:")
    
    # Вивід ключових висновків
    for model_type in lag_comparator.model_types:
        if model_type in lag_comparator.results.get('best_lags', {}):
            best_lag = lag_comparator.results['best_lags'][model_type]
            print(f"\n📌 Для моделі {model_type.upper()}:")
            print(f"   ▶ Оптимальна кількість лагів: {best_lag}")
            
            # Аналіз впливу лагів на параметри моделі
            param_changes = {}
            
            # Визначаємо ключовий параметр для кожної моделі
            key_param = 'alpha' if model_type == 'krr' else 'C' if model_type == 'svr' else None
            
            if key_param:
                values = []
                lags = []
                
                # Збираємо значення параметра для різних лагів
                for lag in sorted(lag_comparator.lag_range):
                    if lag in lag_comparator.results.get('lag_results', {}):
                        if model_type in lag_comparator.results['lag_results'][lag]:
                            kernel_params = lag_comparator.results['lag_results'][lag][model_type].get('kernel_params', {})
                            if key_param in kernel_params:
                                lags.append(lag)
                                values.append(kernel_params[key_param])
                
                if len(values) >= 2:
                    # Визначаємо тренд зміни параметра
                    if values[-1] > values[0] * 1.1:  # Збільшення більше ніж на 10%
                        trend = "значно зростає"
                    elif values[-1] > values[0]:
                        trend = "зростає"
                    elif values[-1] < values[0] * 0.9:  # Зменшення більше ніж на 10%
                        trend = "значно зменшується"
                    elif values[-1] < values[0]:
                        trend = "зменшується"
                    else:
                        trend = "суттєво не змінюється"
                    
                    param_changes[key_param] = trend
                    
                    print(f"   ▶ Параметр {key_param} {trend} зі збільшенням кількості лагів")
                    print(f"     (значення для лагів {lags}: {[round(v, 6) for v in values]})")
    
    print("\n" + "=" * 70)
    print("✅ АНАЛІЗ ЗАВЕРШЕНО")
    print(f"Результати збережено в директорії: {lag_comparator.output_dir}")
    print("=" * 70)
    
    return lag_comparator

def check_krr(df_path='processed.parquet', use_simulation=True):
    """
    Демонстраційний приклад використання класу ModelLagComparator
    для аналізу впливу кількості лагів на параметри моделей.
    
    Args:
        df_path: Шлях до референтного датасету
        use_simulation: Чи використовувати симуляцію даних
        
    Returns:
        ModelLagComparator: Екземпляр компаратора з результатами
    """
    print("=" * 70)
    print("ДЕМОНСТРАЦІЯ ВПЛИВУ КІЛЬКОСТІ ЛАГІВ НА ПАРАМЕТРИ МОДЕЛЕЙ")
    print("=" * 70)
    
    # Завантаження даних
    try:
        import pandas as pd
        from model_lag_comparator import ModelLagComparator
        
        df = pd.read_parquet(df_path)
        print(f"✅ Завантажено {len(df)} записів з {df_path}")
    except FileNotFoundError:
        print(f"❌ Файл '{df_path}' не знайдено")
        return None
    except Exception as e:
        print(f"❌ Помилка при завантаженні даних: {str(e)}")
        return None
    
    print("\nПочаткові дані:")
    print(df.head())
    print(f"Розмірність даних: {df.shape}")
    
    # Створення компаратора моделей
    lag_comparator = ModelLagComparator(
        reference_df=df,
        model_types=["krr"],  # Порівнюємо Kernel Ridge і Support Vector Regression
        lag_range=[2, 4, 6, 8, 10, 12, 14, 16],  # Різні значення лагів для аналізу
        output_dir="lag_comparison_results"
    )
    
    # Запуск порівняння з параметрами симуляції
    print("\n📊 Запуск порівняння моделей з різною кількістю лагів...")
    
    # Параметри симуляції (використовуються тільки якщо use_simulation=True)
    simulation_params = {
        'N_data': 5000,           # Кількість точок даних
        'control_pts': 500,       # Контрольні точки
        'train_size': 0.88,        # Частка тренувальних даних
        'val_size': 0.08,          # Частка валідаційних даних
        'test_size': 0.04,         # Частка тестових даних
        'time_step_s': 5,         # Крок часу в секундах
        'noise_level': 'none',     # Рівень шуму
        'enable_nonlinear': True, # Використовувати нелінійність
        'nonlinear_config': {     # Конфігурація нелінійності
            'concentrate_fe_percent': ('pow', 2.0),
            'concentrate_mass_flow': ('pow', 1.5)
        },
        'use_simulation': use_simulation,  # Чи використовувати симуляцію
        'seed': 42                # Випадкове зерно для відтворюваності
    }
    
    # Запуск порівняння
    results = lag_comparator.run_comparison(**simulation_params)
    
    # Візуалізація результатів
    print("\n📈 Створення візуалізацій...")
    try:
        lag_comparator.plot_results()
    except Exception as e:
        print(f"❌ Помилка при створенні візуалізацій: {str(e)}")
    
    # Генерація та вивід звіту
    print("\n📝 Генерація звіту...")
    try:
        report = lag_comparator.generate_report()
    except Exception as e:
        print(f"❌ Помилка при генерації звіту: {str(e)}")
    
    # Збереження результатів
    print("\n💾 Збереження результатів...")
    try:
        lag_comparator.save_results("lag_comparison_results.json")
    except Exception as e:
        print(f"❌ Помилка при збереженні результатів: {str(e)}")
    
    print("\n" + "=" * 70)
    print("ВИСНОВКИ:")
    
    # Вивід ключових висновків
    for model_type in lag_comparator.model_types:
        if model_type in lag_comparator.results.get('best_lags', {}):
            best_lag = lag_comparator.results['best_lags'][model_type]
            print(f"\n📌 Для моделі {model_type.upper()}:")
            print(f"   ▶ Оптимальна кількість лагів: {best_lag}")
            
            # Аналіз впливу лагів на параметри моделі
            param_changes = {}
            
            # Визначаємо ключовий параметр для кожної моделі
            key_param = 'alpha' if model_type == 'krr' else 'C' if model_type == 'svr' else None
            
            if key_param:
                values = []
                lags = []
                
                # Збираємо значення параметра для різних лагів
                for lag in sorted(lag_comparator.lag_range):
                    if lag in lag_comparator.results.get('lag_results', {}):
                        if model_type in lag_comparator.results['lag_results'][lag]:
                            kernel_params = lag_comparator.results['lag_results'][lag][model_type].get('kernel_params', {})
                            if key_param in kernel_params:
                                lags.append(lag)
                                values.append(kernel_params[key_param])
                
                if len(values) >= 2:
                    # Визначаємо тренд зміни параметра
                    if values[-1] > values[0] * 1.1:  # Збільшення більше ніж на 10%
                        trend = "значно зростає"
                    elif values[-1] > values[0]:
                        trend = "зростає"
                    elif values[-1] < values[0] * 0.9:  # Зменшення більше ніж на 10%
                        trend = "значно зменшується"
                    elif values[-1] < values[0]:
                        trend = "зменшується"
                    else:
                        trend = "суттєво не змінюється"
                    
                    param_changes[key_param] = trend
                    
                    print(f"   ▶ Параметр {key_param} {trend} зі збільшенням кількості лагів")
                    print(f"     (значення для лагів {lags}: {[round(v, 6) for v in values]})")
    
    print("\n" + "=" * 70)
    print("✅ АНАЛІЗ ЗАВЕРШЕНО")
    print(f"Результати збережено в директорії: {lag_comparator.output_dir}")
    print("=" * 70)
    
    return lag_comparator
if __name__ == "__main__":
    run_lag_comparison_example()