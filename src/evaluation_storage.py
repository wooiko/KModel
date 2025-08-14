# evaluation_storage.py - Механізм збереження/завантаження результатів оцінювання MPC

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import zipfile
import hashlib
from dataclasses import dataclass, asdict
import logging

# Імпортуємо структури з основного модуля
try:
    from evaluation_simple import EvaluationResults
except ImportError:
    # Fallback для автономного використання
    from dataclasses import dataclass
    
    @dataclass 
    class EvaluationResults:
        """Контейнер для всіх результатів оцінювання"""
        # Модель (10 метрик)
        model_rmse_fe: float = 0.0
        model_rmse_mass: float = 0.0
        model_r2_fe: float = 0.0
        model_r2_mass: float = 0.0
        model_bias_fe: float = 0.0
        model_bias_mass: float = 0.0
        model_mae_fe: float = 0.0
        model_mae_mass: float = 0.0
        model_mape_fe: float = 0.0
        model_mape_mass: float = 0.0
        
        # EKF метрики (8 метрик)
        ekf_rmse_fe: float = 0.0
        ekf_rmse_mass: float = 0.0
        ekf_normalized_rmse_fe: float = 0.0
        ekf_normalized_rmse_mass: float = 0.0
        ekf_rmse_total: float = 0.0
        ekf_nees_mean: float = 0.0
        ekf_nis_mean: float = 0.0
        ekf_consistency: float = 0.0
        
        # Trust Region метрики (6 метрик)
        trust_radius_mean: float = 0.0
        trust_radius_std: float = 0.0
        trust_radius_min: float = 0.0
        trust_radius_max: float = 0.0
        trust_adaptivity_coeff: float = 0.0
        trust_stability_index: float = 0.0
        
        # Часові метрики (4 метрики)
        initial_training_time: float = 0.0
        avg_retraining_time: float = 0.0
        avg_prediction_time: float = 0.0
        total_retraining_count: float = 0.0
        
        # Керування (13 метрик)
        tracking_error_fe: float = 0.0
        tracking_error_mass: float = 0.0
        control_smoothness: float = 0.0
        setpoint_achievement_fe: float = 0.0
        setpoint_achievement_mass: float = 0.0
        ise_fe: float = 0.0
        ise_mass: float = 0.0
        iae_fe: float = 0.0
        iae_mass: float = 0.0
        tracking_mae_fe: float = 0.0
        tracking_mae_mass: float = 0.0
        tracking_mape_fe: float = 0.0
        tracking_mape_mass: float = 0.0
        
        # Загальна ефективність (2 метрики)
        overall_score: float = 0.0
        process_stability: float = 0.0
        
        # Агресивність керування (11 метрик)
        control_aggressiveness: float = 0.0
        control_variability: float = 0.0
        control_energy: float = 0.0
        control_stability_index: float = 0.0
        control_utilization: float = 0.0
        significant_changes_frequency: float = 0.0
        significant_changes_count: float = 0.0
        max_control_change: float = 0.0
        directional_switches_per_step: float = 0.0
        directional_switches_count: float = 0.0
        steps_at_max_delta_u: float = 0.0

        def to_dict(self) -> Dict:
            return asdict(self)

# =============================================================================
# === СТРУКТУРИ ДАНИХ ДЛЯ ЗБЕРЕЖЕННЯ ===
# =============================================================================

@dataclass
class SimulationMetadata:
    """Метадані симуляції для ідентифікації та відтворення"""
    
    # Основна інформація
    timestamp: str
    simulation_id: str
    description: str
    version: str = "1.0"
    
    # Параметри симуляції
    simulation_steps: int = 0
    dt: float = 1.0
    ref_fe: float = 53.5
    ref_mass: float = 57.0
    
    # Параметри моделі
    model_type: str = "Unknown"
    model_params: Dict[str, Any] = None
    
    # Параметри MPC
    horizon: int = 10
    delta_u_max: float = 1.0
    lambda_u: float = 0.1
    
    # Параметри EKF
    Q_matrix: List[List[float]] = None
    R_matrix: List[List[float]] = None
    
    # Параметри Trust Region
    initial_trust_radius: float = 1.0
    min_trust_radius: float = 0.1
    max_trust_radius: float = 5.0
    
    # Хеш для перевірки цілісності
    data_hash: str = ""
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {}
        if self.Q_matrix is None:
            self.Q_matrix = [[0.01, 0], [0, 0.01]]
        if self.R_matrix is None:
            self.R_matrix = [[0.1, 0], [0, 0.1]]

@dataclass 
class EvaluationPackage:
    """Повний пакет результатів оцінювання"""
    
    metadata: SimulationMetadata
    evaluation_results: EvaluationResults
    simulation_data: pd.DataFrame
    analysis_data: Dict[str, Any]
    parameters: Dict[str, Any]
    
    # Додаткові аналітичні дані
    recommendations: List[str] = None
    performance_summary: str = ""
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

# =============================================================================
# === КЛАСИ ДЛЯ ЗБЕРЕЖЕННЯ/ЗАВАНТАЖЕННЯ ===
# =============================================================================

class EvaluationStorage:
    """Менеджер збереження та завантаження результатів оцінювання"""
    
    def __init__(self, base_directory: str = "evaluation_results"):
        """
        Ініціалізація менеджера збереження
        
        Args:
            base_directory: Базова директорія для збереження результатів
        """
        self.base_dir = Path(base_directory)
        self.base_dir.mkdir(exist_ok=True)
        
        # Налаштування логування
        self._setup_logging()
        
        # Підтримувані формати
        self.supported_formats = {
            'json': self._save_json,
            'pickle': self._save_pickle, 
            'excel': self._save_excel,
            'csv': self._save_csv,
            'zip': self._save_zip_archive
        }
        
        self.load_formats = {
            'json': self._load_json,
            'pickle': self._load_pickle,
            'excel': self._load_excel,
            'csv': self._load_csv,
            'zip': self._load_zip_archive
        }
    
    def _setup_logging(self):
        """Налаштування системи логування"""
        log_file = self.base_dir / "evaluation_storage.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_simulation_id(self, params: Dict) -> str:
        """Генерує унікальний ID симуляції на основі параметрів"""
        
        # Ключові параметри для хешування
        key_params = {
            'ref_fe': params.get('ref_fe', 53.5),
            'ref_mass': params.get('ref_mass', 57.0),
            'horizon': params.get('horizon', 10),
            'lambda_u': params.get('lambda_u', 0.1),
            'delta_u_max': params.get('delta_u_max', 1.0)
        }
        
        # Створюємо хеш
        param_str = json.dumps(key_params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        # Додаємо часову мітку
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"sim_{timestamp}_{param_hash}"
    
    def calculate_data_hash(self, data: Any) -> str:
        """Обчислює хеш даних для перевірки цілісності"""
        
        if isinstance(data, pd.DataFrame):
            # Для DataFrame використовуємо значення
            data_str = data.to_csv()
        elif isinstance(data, dict):
            # Для словника - JSON представлення
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            # Для інших типів - строкове представлення
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def create_evaluation_package(self, 
                                results_df: pd.DataFrame,
                                eval_results: EvaluationResults,
                                analysis_data: Dict,
                                params: Dict,
                                description: str = "",
                                simulation_id: Optional[str] = None) -> EvaluationPackage:
        """
        Створює повний пакет результатів оцінювання
        
        Args:
            results_df: DataFrame з результатами симуляції
            eval_results: Результати оцінювання
            analysis_data: Додаткові дані аналізу
            params: Параметри симуляції
            description: Опис симуляції
            simulation_id: ID симуляції (генерується автоматично якщо не вказано)
            
        Returns:
            EvaluationPackage з усіма даними
        """
        
        if simulation_id is None:
            simulation_id = self.generate_simulation_id(params)
        
        # Створюємо метадані
        metadata = SimulationMetadata(
            timestamp=datetime.now().isoformat(),
            simulation_id=simulation_id,
            description=description,
            simulation_steps=len(results_df),
            dt=params.get('dt', 1.0),
            ref_fe=params.get('ref_fe', 53.5),
            ref_mass=params.get('ref_mass', 57.0),
            model_type=params.get('model_type', 'Unknown'),
            model_params=params.get('model_params', {}),
            horizon=params.get('horizon', 10),
            delta_u_max=params.get('delta_u_max', 1.0),
            lambda_u=params.get('lambda_u', 0.1),
            Q_matrix=params.get('Q_matrix', [[0.01, 0], [0, 0.01]]),
            R_matrix=params.get('R_matrix', [[0.1, 0], [0, 0.1]]),
            initial_trust_radius=params.get('initial_trust_radius', 1.0),
            min_trust_radius=params.get('min_trust_radius', 0.1),
            max_trust_radius=params.get('max_trust_radius', 5.0)
        )
        
        # Обчислюємо хеш даних
        combined_data = {
            'results': results_df.to_dict(),
            'evaluation': eval_results.to_dict(),
            'analysis': analysis_data,
            'params': params
        }
        metadata.data_hash = self.calculate_data_hash(combined_data)
        
        # Генеруємо рекомендації (якщо доступна функція)
        recommendations = []
        try:
            from evaluation_simple import generate_recommendations
            recommendations = generate_recommendations(eval_results, len(results_df))
        except ImportError:
            self.logger.warning("Не вдалося імпортувати generate_recommendations")
        
        # Створюємо резюме продуктивності
        try:
            from evaluation_simple import get_performance_summary
            performance_summary = get_performance_summary(eval_results)
        except ImportError:
            performance_summary = f"Загальна оцінка: {eval_results.overall_score:.1f}/100"
        
        return EvaluationPackage(
            metadata=metadata,
            evaluation_results=eval_results,
            simulation_data=results_df.copy(),
            analysis_data=analysis_data.copy() if analysis_data else {},
            parameters=params.copy(),
            recommendations=recommendations,
            performance_summary=performance_summary
        )
    
    # =============================================================================
    # === МЕТОДИ ЗБЕРЕЖЕННЯ ===
    # =============================================================================
    
    def save_evaluation(self, 
                       package: EvaluationPackage,
                       format_type: str = 'zip',
                       custom_name: Optional[str] = None) -> str:
        """
        Зберігає пакет оцінювання у вказаному форматі
        
        Args:
            package: Пакет результатів оцінювання
            format_type: Формат збереження ('json', 'pickle', 'excel', 'csv', 'zip')
            custom_name: Користувацьке ім'я файлу
            
        Returns:
            Шлях до збереженого файлу
        """
        
        if format_type not in self.supported_formats:
            raise ValueError(f"Непідтримуваний формат: {format_type}. "
                           f"Доступні: {list(self.supported_formats.keys())}")
        
        # Визначаємо ім'я файлу
        if custom_name:
            filename = custom_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{package.metadata.simulation_id}_{timestamp}"
        
        # Викликаємо відповідний метод збереження
        filepath = self.supported_formats[format_type](package, filename)
        
        self.logger.info(f"Результати збережено: {filepath}")
        return str(filepath)
    
    def _save_json(self, package: EvaluationPackage, filename: str) -> Path:
        """Збереження у JSON форматі"""
        
        filepath = self.base_dir / f"{filename}.json"
        
        # Конвертуємо все у JSON-сумісний формат
        json_data = {
            'metadata': asdict(package.metadata),
            'evaluation_results': package.evaluation_results.to_dict(),
            'simulation_data': package.simulation_data.to_dict('records'),
            'analysis_data': self._serialize_analysis_data(package.analysis_data),
            'parameters': package.parameters,
            'recommendations': package.recommendations,
            'performance_summary': package.performance_summary
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        
        return filepath
    
    def _save_pickle(self, package: EvaluationPackage, filename: str) -> Path:
        """Збереження у Pickle форматі (найшвидший і найповніший)"""
        
        filepath = self.base_dir / f"{filename}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return filepath
    
    def _save_excel(self, package: EvaluationPackage, filename: str) -> Path:
        """Збереження у Excel форматі з кількома листами"""
        
        filepath = self.base_dir / f"{filename}.xlsx"
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Лист 1: Результати симуляції
            package.simulation_data.to_excel(writer, sheet_name='Simulation_Data', index=False)
            
            # Лист 2: Метрики оцінювання
            eval_df = pd.DataFrame([package.evaluation_results.to_dict()]).T
            eval_df.columns = ['Value']
            eval_df.to_excel(writer, sheet_name='Evaluation_Metrics')
            
            # Лист 3: Метадані та параметри
            metadata_df = pd.DataFrame([asdict(package.metadata)]).T
            metadata_df.columns = ['Value']
            metadata_df.to_excel(writer, sheet_name='Metadata')
            
            # Лист 4: Рекомендації
            if package.recommendations:
                rec_df = pd.DataFrame(package.recommendations, columns=['Recommendations'])
                rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            # Лист 5: Аналітичні дані (спрощені)
            try:
                analysis_simple = self._simplify_analysis_data(package.analysis_data)
                if analysis_simple:
                    analysis_df = pd.DataFrame([analysis_simple]).T
                    analysis_df.columns = ['Value']
                    analysis_df.to_excel(writer, sheet_name='Analysis_Summary')
            except Exception as e:
                self.logger.warning(f"Не вдалося зберегти аналітичні дані в Excel: {e}")
        
        return filepath
    
    def _save_csv(self, package: EvaluationPackage, filename: str) -> Path:
        """Збереження основних даних у CSV форматі"""
        
        # Створюємо директорію для CSV файлів
        csv_dir = self.base_dir / f"{filename}_csv"
        csv_dir.mkdir(exist_ok=True)
        
        # Зберігаємо дані симуляції
        sim_path = csv_dir / "simulation_data.csv"
        package.simulation_data.to_csv(sim_path, index=False)
        
        # Зберігаємо метрики оцінювання
        eval_path = csv_dir / "evaluation_metrics.csv"
        eval_df = pd.DataFrame([package.evaluation_results.to_dict()]).T
        eval_df.to_csv(eval_path)
        
        # Зберігаємо метадані
        meta_path = csv_dir / "metadata.csv"
        meta_df = pd.DataFrame([asdict(package.metadata)]).T
        meta_df.to_csv(meta_path)
        
        # Зберігаємо рекомендації
        if package.recommendations:
            rec_path = csv_dir / "recommendations.csv"
            rec_df = pd.DataFrame(package.recommendations, columns=['Recommendations'])
            rec_df.to_csv(rec_path, index=False)
        
        return csv_dir
    
    def _save_zip_archive(self, package: EvaluationPackage, filename: str) -> Path:
        """Збереження у стиснутому ZIP архіві"""
        
        zip_path = self.base_dir / f"{filename}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # JSON з усіма даними
            json_data = {
                'metadata': asdict(package.metadata),
                'evaluation_results': package.evaluation_results.to_dict(),
                'analysis_data': self._serialize_analysis_data(package.analysis_data),
                'parameters': package.parameters,
                'recommendations': package.recommendations,
                'performance_summary': package.performance_summary
            }
            zf.writestr('evaluation_data.json', 
                       json.dumps(json_data, indent=2, default=str))
            
            # CSV з результатами симуляції
            sim_csv = package.simulation_data.to_csv(index=False)
            zf.writestr('simulation_data.csv', sim_csv)
            
            # Readme файл
            readme_content = self._generate_readme(package)
            zf.writestr('README.txt', readme_content)
        
        return zip_path
    
    # =============================================================================
    # === МЕТОДИ ЗАВАНТАЖЕННЯ ===
    # =============================================================================
    
    def load_evaluation(self, filepath: Union[str, Path]) -> EvaluationPackage:
        """
        Завантажує пакет оцінювання з файлу
        
        Args:
            filepath: Шлях до файлу
            
        Returns:
            EvaluationPackage з завантаженими даними
        """
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Файл не знайдено: {filepath}")
        
        # Визначаємо формат за розширенням
        suffix = filepath.suffix.lower()
        
        if suffix == '.json':
            return self._load_json(filepath)
        elif suffix == '.pkl':
            return self._load_pickle(filepath) 
        elif suffix in ['.xlsx', '.xls']:
            return self._load_excel(filepath)
        elif suffix == '.zip':
            return self._load_zip_archive(filepath)
        elif filepath.is_dir():  # CSV директорія
            return self._load_csv(filepath)
        else:
            raise ValueError(f"Непідтримуваний формат файлу: {suffix}")
    
    def _load_json(self, filepath: Path) -> EvaluationPackage:
        """Завантаження з JSON файлу"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Відновлюємо структури даних
        metadata = SimulationMetadata(**data['metadata'])
        eval_results = EvaluationResults(**data['evaluation_results'])
        simulation_data = pd.DataFrame(data['simulation_data'])
        
        return EvaluationPackage(
            metadata=metadata,
            evaluation_results=eval_results,
            simulation_data=simulation_data,
            analysis_data=data.get('analysis_data', {}),
            parameters=data.get('parameters', {}),
            recommendations=data.get('recommendations', []),
            performance_summary=data.get('performance_summary', '')
        )
    
    def _load_pickle(self, filepath: Path) -> EvaluationPackage:
        """Завантаження з Pickle файлу"""
        
        with open(filepath, 'rb') as f:
            package = pickle.load(f)
        
        return package
    
    def _load_excel(self, filepath: Path) -> EvaluationPackage:
        """Завантаження з Excel файлу"""
        
        # Читаємо різні листи
        simulation_data = pd.read_excel(filepath, sheet_name='Simulation_Data')
        
        eval_df = pd.read_excel(filepath, sheet_name='Evaluation_Metrics', index_col=0)
        eval_dict = eval_df['Value'].to_dict()
        eval_results = EvaluationResults(**eval_dict)
        
        meta_df = pd.read_excel(filepath, sheet_name='Metadata', index_col=0)
        meta_dict = meta_df['Value'].to_dict()
        metadata = SimulationMetadata(**meta_dict)
        
        # Рекомендації (якщо є)
        recommendations = []
        try:
            rec_df = pd.read_excel(filepath, sheet_name='Recommendations')
            recommendations = rec_df['Recommendations'].tolist()
        except:
            pass
        
        return EvaluationPackage(
            metadata=metadata,
            evaluation_results=eval_results,
            simulation_data=simulation_data,
            analysis_data={},  # Аналітичні дані втрачаються в Excel
            parameters={},
            recommendations=recommendations,
            performance_summary=""
        )
    
    def _load_csv(self, dirpath: Path) -> EvaluationPackage:
        """Завантаження з CSV директорії"""
        
        # Завантажуємо основні файли
        simulation_data = pd.read_csv(dirpath / 'simulation_data.csv')
        
        eval_df = pd.read_csv(dirpath / 'evaluation_metrics.csv', index_col=0)
        eval_dict = eval_df.iloc[:, 0].to_dict()
        eval_results = EvaluationResults(**eval_dict)
        
        meta_df = pd.read_csv(dirpath / 'metadata.csv', index_col=0)
        meta_dict = meta_df.iloc[:, 0].to_dict()
        metadata = SimulationMetadata(**meta_dict)
        
        # Рекомендації
        recommendations = []
        rec_path = dirpath / 'recommendations.csv'
        if rec_path.exists():
            rec_df = pd.read_csv(rec_path)
            recommendations = rec_df['Recommendations'].tolist()
        
        return EvaluationPackage(
            metadata=metadata,
            evaluation_results=eval_results,
            simulation_data=simulation_data,
            analysis_data={},
            parameters={},
            recommendations=recommendations,
            performance_summary=""
        )
    
    def _load_zip_archive(self, filepath: Path) -> EvaluationPackage:
        """Завантаження з ZIP архіву"""
        
        with zipfile.ZipFile(filepath, 'r') as zf:
            # Завантажуємо JSON дані
            json_content = zf.read('evaluation_data.json').decode('utf-8')
            data = json.loads(json_content)
            
            # Завантажуємо CSV з результатами
            csv_content = zf.read('simulation_data.csv').decode('utf-8')
            from io import StringIO
            simulation_data = pd.read_csv(StringIO(csv_content))
        
        # Відновлюємо структури
        metadata = SimulationMetadata(**data['metadata'])
        eval_results = EvaluationResults(**data['evaluation_results'])
        
        return EvaluationPackage(
            metadata=metadata,
            evaluation_results=eval_results,
            simulation_data=simulation_data,
            analysis_data=data.get('analysis_data', {}),
            parameters=data.get('parameters', {}),
            recommendations=data.get('recommendations', []),
            performance_summary=data.get('performance_summary', '')
        )
    
    # =============================================================================
    # === ДОПОМІЖНІ МЕТОДИ ===
    # =============================================================================
    
    def _serialize_analysis_data(self, analysis_data: Dict) -> Dict:
        """Серіалізує аналітичні дані для JSON"""
        
        serialized = {}
        
        for key, value in analysis_data.items():
            try:
                if isinstance(value, np.ndarray):
                    serialized[key] = value.tolist()
                elif isinstance(value, pd.DataFrame):
                    serialized[key] = value.to_dict('records')
                elif isinstance(value, (list, dict, str, int, float, bool)):
                    serialized[key] = value
                else:
                    # Для інших типів використовуємо строкове представлення
                    serialized[key] = str(value)
            except Exception as e:
                self.logger.warning(f"Не вдалося серіалізувати {key}: {e}")
                serialized[key] = f"<Не вдалося серіалізувати: {type(value).__name__}>"
        
        return serialized
    
    def _simplify_analysis_data(self, analysis_data: Dict) -> Dict:
        """Спрощує аналітичні дані для Excel"""
        
        simplified = {}
        
        for key, value in analysis_data.items():
            try:
                if isinstance(value, np.ndarray):
                    # Для масивів зберігаємо тільки основну статистику
                    if value.size > 0:
                        simplified[f"{key}_mean"] = float(np.mean(value))
                        simplified[f"{key}_std"] = float(np.std(value))
                        simplified[f"{key}_min"] = float(np.min(value))
                        simplified[f"{key}_max"] = float(np.max(value))
                        simplified[f"{key}_size"] = int(value.size)
                elif isinstance(value, (list, tuple)):
                    simplified[f"{key}_length"] = len(value)
                    if len(value) > 0 and isinstance(value[0], (int, float)):
                        simplified[f"{key}_first"] = value[0]
                        simplified[f"{key}_last"] = value[-1]
                elif isinstance(value, dict):
                    simplified[f"{key}_keys_count"] = len(value)
                elif isinstance(value, (str, int, float, bool)):
                    simplified[key] = value
            except Exception as e:
                self.logger.warning(f"Не вдалося спростити {key}: {e}")
        
        return simplified
    
    def _generate_readme(self, package: EvaluationPackage) -> str:
        """Генерує README файл для архіву"""
        
        readme = f"""
🎯 РЕЗУЛЬТАТИ ОЦІНЮВАННЯ MPC СИМУЛЯЦІЇ
=====================================

ЗАГАЛЬНА ІНФОРМАЦІЯ:
-------------------
ID Симуляції: {package.metadata.simulation_id}
Час створення: {package.metadata.timestamp}
Опис: {package.metadata.description}
Версія: {package.metadata.version}

ПАРАМЕТРИ СИМУЛЯЦІЇ:
-------------------
Кроки симуляції: {package.metadata.simulation_steps}
Час кроку (dt): {package.metadata.dt}
Уставка Fe: {package.metadata.ref_fe}%
Уставка Mass: {package.metadata.ref_mass} т/г

ПАРАМЕТРИ MPC:
--------------
Горизонт: {package.metadata.horizon}
Максимальна зміна керування: {package.metadata.delta_u_max}
Lambda регуляризація: {package.parameters.get('lambda_u', 'N/A')}

ПАРАМЕТРИ EKF:
--------------
Q матриця: {package.metadata.Q_matrix}
R матриця: {package.metadata.R_matrix}

ПАРАМЕТРИ TRUST REGION:
----------------------
Початковий радіус: {package.metadata.initial_trust_radius}
Мінімальний радіус: {package.metadata.min_trust_radius}
Максимальний радіус: {package.metadata.max_trust_radius}

КЛЮЧОВІ РЕЗУЛЬТАТИ:
------------------
Загальна оцінка: {package.evaluation_results.overall_score:.1f}/100
Стабільність процесу: {package.evaluation_results.process_stability:.3f}

Модель Fe:
  - RMSE: {package.evaluation_results.model_rmse_fe:.3f}
  - R²: {package.evaluation_results.model_r2_fe:.3f}
  - MAE: {package.evaluation_results.model_mae_fe:.3f}
  - MAPE: {package.evaluation_results.model_mape_fe:.2f}%

Модель Mass:
  - RMSE: {package.evaluation_results.model_rmse_mass:.3f}
  - R²: {package.evaluation_results.model_r2_mass:.3f}
  - MAE: {package.evaluation_results.model_mae_mass:.3f}
  - MAPE: {package.evaluation_results.model_mape_mass:.2f}%

EKF Ефективність:
  - NEES: {package.evaluation_results.ekf_nees_mean:.2f} (ідеал ≈ 2)
  - NIS: {package.evaluation_results.ekf_nis_mean:.2f} (ідеал ≈ 2)
  - Консистентність: {package.evaluation_results.ekf_consistency:.3f}

Trust Region:
  - Середній радіус: {package.evaluation_results.trust_radius_mean:.3f}
  - Стабільність: {package.evaluation_results.trust_stability_index:.3f}
  - Адаптивність: {package.evaluation_results.trust_adaptivity_coeff:.3f}

Відстеження уставок:
  - Fe досягнення: {package.evaluation_results.setpoint_achievement_fe:.1f}%
  - Mass досягнення: {package.evaluation_results.setpoint_achievement_mass:.1f}%
  - Згладженість керування: {package.evaluation_results.control_smoothness:.3f}

Часові метрики:
  - Початкове навчання: {package.evaluation_results.initial_training_time:.2f} сек
  - Середній час прогнозування: {package.evaluation_results.avg_prediction_time:.2f} мс
  - Кількість перенавчань: {package.evaluation_results.total_retraining_count:.0f}

ПРОДУКТИВНІСТЬ:
--------------
{package.performance_summary}

РЕКОМЕНДАЦІЇ:
------------"""

        if package.recommendations:
            for i, rec in enumerate(package.recommendations, 1):
                readme += f"\n{i}. {rec}"
        else:
            readme += "\nРекомендації не згенеровано"

        readme += f"""

ФАЙЛИ В АРХІВІ:
--------------
- evaluation_data.json: Повні результати оцінювання в JSON форматі
- simulation_data.csv: Дані симуляції в CSV форматі
- README.txt: Цей файл з описом

ПЕРЕВІРКА ЦІЛІСНОСТІ:
--------------------
Хеш даних: {package.metadata.data_hash}

ВИКОРИСТАННЯ:
------------
Для завантаження використовуйте:

```python
from evaluation_storage import EvaluationStorage

storage = EvaluationStorage()
package = storage.load_evaluation('путь_к_файлу.zip')

# Доступ до даних
results_df = package.simulation_data
eval_results = package.evaluation_results
metadata = package.metadata
```

Створено модулем evaluation_storage.py
"""
        return readme
    
    # =============================================================================
    # === МЕТОДИ ПОШУКУ ТА УПРАВЛІННЯ ===
    # =============================================================================
    
    def list_saved_evaluations(self) -> pd.DataFrame:
        """Повертає список всіх збережених оцінювань"""
        
        evaluations = []
        
        # Шукаємо всі підтримувані формати
        for file_path in self.base_dir.rglob('*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in ['.json', '.pkl', '.xlsx', '.zip']:
                    try:
                        # Спробуємо швидко витягти метадані
                        metadata = self._extract_metadata_fast(file_path)
                        if metadata:
                            eval_info = {
                                'filename': file_path.name,
                                'filepath': str(file_path),
                                'format': suffix[1:],
                                'size_mb': file_path.stat().st_size / (1024*1024),
                                'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                                'simulation_id': metadata.get('simulation_id', 'Unknown'),
                                'description': metadata.get('description', ''),
                                'simulation_steps': metadata.get('simulation_steps', 0),
                                'overall_score': metadata.get('overall_score', 0.0)
                            }
                            evaluations.append(eval_info)
                    except Exception as e:
                        self.logger.warning(f"Не вдалося обробити файл {file_path}: {e}")
        
        return pd.DataFrame(evaluations)
    
    def _extract_metadata_fast(self, filepath: Path) -> Optional[Dict]:
        """Швидке витягування основних метаданих без повного завантаження"""
        
        try:
            suffix = filepath.suffix.lower()
            
            if suffix == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                metadata = data.get('metadata', {})
                eval_results = data.get('evaluation_results', {})
                return {**metadata, 'overall_score': eval_results.get('overall_score', 0.0)}
            
            elif suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zf:
                    json_content = zf.read('evaluation_data.json').decode('utf-8')
                    data = json.loads(json_content)
                    metadata = data.get('metadata', {})
                    eval_results = data.get('evaluation_results', {})
                    return {**metadata, 'overall_score': eval_results.get('overall_score', 0.0)}
            
            # Для інших форматів повертаємо базову інформацію
            return {'simulation_id': filepath.stem, 'description': '', 'simulation_steps': 0}
            
        except Exception:
            return None
    
    def find_evaluations(self, 
                        simulation_id: Optional[str] = None,
                        min_score: Optional[float] = None,
                        max_score: Optional[float] = None,
                        date_from: Optional[str] = None,
                        date_to: Optional[str] = None) -> pd.DataFrame:
        """
        Знаходить оцінювання за критеріями
        
        Args:
            simulation_id: ID симуляції (часткове співпадіння)
            min_score: Мінімальна загальна оцінка
            max_score: Максимальна загальна оцінка  
            date_from: Дата від (YYYY-MM-DD)
            date_to: Дата до (YYYY-MM-DD)
            
        Returns:
            DataFrame з відфільтрованими результатами
        """
        
        df = self.list_saved_evaluations()
        
        if df.empty:
            return df
        
        # Фільтрація
        if simulation_id:
            df = df[df['simulation_id'].str.contains(simulation_id, case=False, na=False)]
        
        if min_score is not None:
            df = df[df['overall_score'] >= min_score]
        
        if max_score is not None:
            df = df[df['overall_score'] <= max_score]
        
        if date_from:
            date_from_dt = pd.to_datetime(date_from)
            df = df[df['modified'] >= date_from_dt]
        
        if date_to:
            date_to_dt = pd.to_datetime(date_to)
            df = df[df['modified'] <= date_to_dt]
        
        return df.sort_values('overall_score', ascending=False)
    
    def delete_evaluation(self, filepath: Union[str, Path]) -> bool:
        """
        Видаляє збережене оцінювання
        
        Args:
            filepath: Шлях до файлу або директорії
            
        Returns:
            True якщо успішно видалено
        """
        
        filepath = Path(filepath)
        
        try:
            if filepath.is_file():
                filepath.unlink()
                self.logger.info(f"Файл видалено: {filepath}")
                return True
            elif filepath.is_dir():
                import shutil
                shutil.rmtree(filepath)
                self.logger.info(f"Директорію видалено: {filepath}")
                return True
            else:
                self.logger.warning(f"Файл не знайдено: {filepath}")
                return False
        except Exception as e:
            self.logger.error(f"Помилка при видаленні {filepath}: {e}")
            return False
    
    def cleanup_old_evaluations(self, days_old: int = 30) -> int:
        """
        Видаляє старі оцінювання
        
        Args:
            days_old: Видалити файли старші за кількість днів
            
        Returns:
            Кількість видалених файлів
        """
        
        cutoff_date = datetime.now() - pd.Timedelta(days=days_old)
        df = self.list_saved_evaluations()
        
        old_files = df[df['modified'] < cutoff_date]
        deleted_count = 0
        
        for _, row in old_files.iterrows():
            if self.delete_evaluation(row['filepath']):
                deleted_count += 1
        
        self.logger.info(f"Видалено {deleted_count} старих файлів (старше {days_old} днів)")
        return deleted_count
    
    # =============================================================================
    # === МЕТОДИ ПОРІВНЯННЯ ===
    # =============================================================================
    
    def compare_evaluations_from_files(self, filepaths: List[Union[str, Path]]) -> pd.DataFrame:
        """
        Порівнює результати з кількох файлів
        
        Args:
            filepaths: Список шляхів до файлів
            
        Returns:
            DataFrame з порівнянням метрик
        """
        
        packages = {}
        
        for filepath in filepaths:
            try:
                package = self.load_evaluation(filepath)
                name = f"{package.metadata.simulation_id[:8]}..."
                packages[name] = package
            except Exception as e:
                self.logger.error(f"Не вдалося завантажити {filepath}: {e}")
        
        if not packages:
            return pd.DataFrame()
        
        return self._create_comparison_dataframe(packages)
    
    def _create_comparison_dataframe(self, packages: Dict[str, EvaluationPackage]) -> pd.DataFrame:
        """Створює DataFrame для порівняння результатів"""
        
        comparison_data = {}
        
        for name, package in packages.items():
            eval_dict = package.evaluation_results.to_dict()
            comparison_data[name] = eval_dict
        
        df = pd.DataFrame(comparison_data)
        
        # Додаємо категорії метрик для кращої організації
        metric_categories = {
            'Модель Fe': ['model_rmse_fe', 'model_r2_fe', 'model_mae_fe', 'model_mape_fe', 'model_bias_fe'],
            'Модель Mass': ['model_rmse_mass', 'model_r2_mass', 'model_mae_mass', 'model_mape_mass', 'model_bias_mass'],
            'EKF': ['ekf_rmse_fe', 'ekf_rmse_mass', 'ekf_nees_mean', 'ekf_nis_mean', 'ekf_consistency'],
            'Trust Region': ['trust_radius_mean', 'trust_radius_std', 'trust_stability_index', 'trust_adaptivity_coeff'],
            'Відстеження': ['tracking_error_fe', 'tracking_error_mass', 'setpoint_achievement_fe', 'setpoint_achievement_mass'],
            'Керування': ['control_smoothness', 'control_aggressiveness', 'control_stability_index'],
            'Час': ['initial_training_time', 'avg_retraining_time', 'avg_prediction_time'],
            'Загальне': ['overall_score', 'process_stability']
        }
        
        # Реорганізуємо DataFrame за категоріями
        organized_df = pd.DataFrame()
        for category, metrics in metric_categories.items():
            for metric in metrics:
                if metric in df.index:
                    organized_df.loc[f"{category}: {metric}", :] = df.loc[metric, :]
        
        return organized_df
    
    # =============================================================================
    # === МЕТОДИ ЕКСПОРТУ ЗВІТІВ ===
    # =============================================================================
    
    def generate_report(self, package: EvaluationPackage, 
                       report_type: str = 'html') -> str:
        """
        Генерує звіт про оцінювання
        
        Args:
            package: Пакет результатів
            report_type: Тип звіту ('html', 'pdf', 'markdown')
            
        Returns:
            Шлях до створеного звіту
        """
        
        if report_type == 'html':
            return self._generate_html_report(package)
        elif report_type == 'markdown':
            return self._generate_markdown_report(package)
        else:
            raise ValueError(f"Непідтримуваний тип звіту: {report_type}")
    
    def _generate_html_report(self, package: EvaluationPackage) -> str:
        """Генерує HTML звіт"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Звіт MPC Оцінювання - {package.metadata.simulation_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .metric-group {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: flex; justify-content: space-between; margin: 5px 0; }}
        .score {{ font-size: 2em; color: #2c3e50; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .good {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .bad {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 Звіт MPC Оцінювання</h1>
        <p><strong>ID Симуляції:</strong> {package.metadata.simulation_id}</p>
        <p><strong>Дата:</strong> {package.metadata.timestamp}</p>
        <p><strong>Опис:</strong> {package.metadata.description}</p>
        <div class="score">Загальна оцінка: {package.evaluation_results.overall_score:.1f}/100</div>
    </div>
    
    <div class="metric-group">
        <h2>📊 Ключові Метрики</h2>
        <div class="metric"><span>Стабільність процесу:</span><span>{package.evaluation_results.process_stability:.3f}</span></div>
        <div class="metric"><span>EKF Консистентність:</span><span>{package.evaluation_results.ekf_consistency:.3f}</span></div>
        <div class="metric"><span>Trust Region Стабільність:</span><span>{package.evaluation_results.trust_stability_index:.3f}</span></div>
    </div>
    
    <div class="metric-group">
        <h2>🎯 Якість Моделей</h2>
        <table>
            <tr><th>Метрика</th><th>Fe</th><th>Mass</th></tr>
            <tr><td>RMSE</td><td>{package.evaluation_results.model_rmse_fe:.3f}</td><td>{package.evaluation_results.model_rmse_mass:.3f}</td></tr>
            <tr><td>R²</td><td>{package.evaluation_results.model_r2_fe:.3f}</td><td>{package.evaluation_results.model_r2_mass:.3f}</td></tr>
            <tr><td>MAE</td><td>{package.evaluation_results.model_mae_fe:.3f}</td><td>{package.evaluation_results.model_mae_mass:.3f}</td></tr>
            <tr><td>MAPE (%)</td><td>{package.evaluation_results.model_mape_fe:.2f}</td><td>{package.evaluation_results.model_mape_mass:.2f}</td></tr>
        </table>
    </div>
    
    <div class="metric-group">
        <h2>🔍 EKF Аналіз</h2>
        <table>
            <tr><th>Метрика</th><th>Значення</th><th>Ідеал</th></tr>
            <tr><td>NEES</td><td>{package.evaluation_results.ekf_nees_mean:.2f}</td><td>≈ 2</td></tr>
            <tr><td>NIS</td><td>{package.evaluation_results.ekf_nis_mean:.2f}</td><td>≈ 2</td></tr>
            <tr><td>Загальна консистентність</td><td>{package.evaluation_results.ekf_consistency:.3f}</td><td>> 0.7</td></tr>
        </table>
    </div>
    
    <div class="metric-group">
        <h2>🎛️ Trust Region</h2>
        <div class="metric"><span>Середній радіус:</span><span>{package.evaluation_results.trust_radius_mean:.3f}</span></div>
        <div class="metric"><span>Діапазон:</span><span>[{package.evaluation_results.trust_radius_min:.3f}, {package.evaluation_results.trust_radius_max:.3f}]</span></div>
        <div class="metric"><span>Адаптивність:</span><span>{package.evaluation_results.trust_adaptivity_coeff:.3f}</span></div>
    </div>
    
    <div class="metric-group">
        <h2>🎮 Якість Керування</h2>
        <table>
            <tr><th>Метрика</th><th>Fe</th><th>Mass</th></tr>
            <tr><td>Досягнення уставки (%)</td><td>{package.evaluation_results.setpoint_achievement_fe:.1f}</td><td>{package.evaluation_results.setpoint_achievement_mass:.1f}</td></tr>
            <tr><td>Tracking MAE</td><td>{package.evaluation_results.tracking_mae_fe:.3f}</td><td>{package.evaluation_results.tracking_mae_mass:.3f}</td></tr>
            <tr><td>Tracking MAPE (%)</td><td>{package.evaluation_results.tracking_mape_fe:.2f}</td><td>{package.evaluation_results.tracking_mape_mass:.2f}</td></tr>
        </table>
        <div class="metric"><span>Згладженість керування:</span><span>{package.evaluation_results.control_smoothness:.3f}</span></div>
    </div>
    
    <div class="metric-group">
        <h2>⏱️ Продуктивність</h2>
        <div class="metric"><span>Початкове навчання:</span><span>{package.evaluation_results.initial_training_time:.2f} сек</span></div>
        <div class="metric"><span>Час прогнозування:</span><span>{package.evaluation_results.avg_prediction_time:.2f} мс</span></div>
        <div class="metric"><span>Кількість перенавчань:</span><span>{package.evaluation_results.total_retraining_count:.0f}</span></div>
    </div>
    
    <div class="metric-group">
        <h2>💡 Рекомендації</h2>
        <ul>"""
        
        for rec in package.recommendations:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
        </ul>
    </div>
</body>
</html>"""
        
        # Зберігаємо файл
        report_path = self.base_dir / f"report_{package.metadata.simulation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML звіт створено: {report_path}")
        return str(report_path)
    
    def _generate_markdown_report(self, package: EvaluationPackage) -> str:
        """Генерує Markdown звіт"""
        
        md_content = f"""# 🎯 Звіт MPC Оцінювання

## Загальна Інформація
- **ID Симуляції:** {package.metadata.simulation_id}
- **Дата:** {package.metadata.timestamp}
- **Опис:** {package.metadata.description}
- **Загальна оцінка:** **{package.evaluation_results.overall_score:.1f}/100**

## 📊 Ключові Метрики
- Стабільність процесу: {package.evaluation_results.process_stability:.3f}
- EKF Консистентність: {package.evaluation_results.ekf_consistency:.3f}
- Trust Region Стабільність: {package.evaluation_results.trust_stability_index:.3f}

## 🎯 Якість Моделей

| Метрика | Fe | Mass |
|---------|-------|-------|
| RMSE | {package.evaluation_results.model_rmse_fe:.3f} | {package.evaluation_results.model_rmse_mass:.3f} |
| R² | {package.evaluation_results.model_r2_fe:.3f} | {package.evaluation_results.model_r2_mass:.3f} |
| MAE | {package.evaluation_results.model_mae_fe:.3f} | {package.evaluation_results.model_mae_mass:.3f} |
| MAPE (%) | {package.evaluation_results.model_mape_fe:.2f} | {package.evaluation_results.model_mape_mass:.2f} |

## 🔍 EKF Аналіз

| Метрика | Значення | Ідеал |
|---------|----------|-------|
| NEES | {package.evaluation_results.ekf_nees_mean:.2f} | ≈ 2 |
| NIS | {package.evaluation_results.ekf_nis_mean:.2f} | ≈ 2 |
| Консистентність | {package.evaluation_results.ekf_consistency:.3f} | > 0.7 |

## 🎛️ Trust Region
- Середній радіус: {package.evaluation_results.trust_radius_mean:.3f}
- Діапазон: [{package.evaluation_results.trust_radius_min:.3f}, {package.evaluation_results.trust_radius_max:.3f}]
- Адаптивність: {package.evaluation_results.trust_adaptivity_coeff:.3f}

## 🎮 Якість Керування

| Метрика | Fe | Mass |
|---------|-------|-------|
| Досягнення уставки (%) | {package.evaluation_results.setpoint_achievement_fe:.1f} | {package.evaluation_results.setpoint_achievement_mass:.1f} |
| Tracking MAE | {package.evaluation_results.tracking_mae_fe:.3f} | {package.evaluation_results.tracking_mae_mass:.3f} |
| Tracking MAPE (%) | {package.evaluation_results.tracking_mape_fe:.2f} | {package.evaluation_results.tracking_mape_mass:.2f} |

- Згладженість керування: {package.evaluation_results.control_smoothness:.3f}

## ⏱️ Продуктивність
- Початкове навчання: {package.evaluation_results.initial_training_time:.2f} сек
- Час прогнозування: {package.evaluation_results.avg_prediction_time:.2f} мс  
- Кількість перенавчань: {package.evaluation_results.total_retraining_count:.0f}

## 💡 Рекомендації
"""
        
        for i, rec in enumerate(package.recommendations, 1):
            md_content += f"{i}. {rec}\n"
        
        # Зберігаємо файл
        report_path = self.base_dir / f"report_{package.metadata.simulation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        self.logger.info(f"Markdown звіт створено: {report_path}")
        return str(report_path)

# =============================================================================
# === ФУНКЦІЇ ВИСОКОГО РІВНЯ ===
# =============================================================================

def quick_save(results_df: pd.DataFrame, 
               eval_results: EvaluationResults,
               analysis_data: Dict,
               params: Dict,
               description: str = "",
               base_dir: str = "evaluation_results") -> str:
    """
    Швидке збереження результатів у ZIP форматі
    
    Args:
        results_df: DataFrame з результатами симуляції
        eval_results: Результати оцінювання
        analysis_data: Додаткові дані аналізу
        params: Параметри симуляції
        description: Опис симуляції
        base_dir: Базова директорія
        
    Returns:
        Шлях до збереженого файлу
    """
    
    storage = EvaluationStorage(base_dir)
    
    package = storage.create_evaluation_package(
        results_df=results_df,
        eval_results=eval_results,
        analysis_data=analysis_data,
        params=params,
        description=description
    )
    
    return storage.save_evaluation(package, format_type='zip')

def quick_load(filepath: Union[str, Path]) -> EvaluationPackage:
    """
    Швидке завантаження результатів
    
    Args:
        filepath: Шлях до файлу
        
    Returns:
        EvaluationPackage з завантаженими даними
    """
    
    storage = EvaluationStorage()
    return storage.load_evaluation(filepath)

def compare_saved_evaluations(filepaths: List[Union[str, Path]]) -> pd.DataFrame:
    """
    Порівняння збережених оцінювань
    
    Args:
        filepaths: Список шляхів до файлів
        
    Returns:
        DataFrame з порівнянням
    """
    
    storage = EvaluationStorage()
    return storage.compare_evaluations_from_files(filepaths)

# =============================================================================
# === ПРИКЛАД ВИКОРИСТАННЯ ===
# =============================================================================

if __name__ == "__main__":
    print("🔧 evaluation_storage.py - Модуль збереження/завантаження результатів")
    print("\nОсновні функції:")
    print("1. quick_save() - Швидке збереження")
    print("2. quick_load() - Швидке завантаження")
    print("3. EvaluationStorage() - Повний менеджер")
    print("4. compare_saved_evaluations() - Порівняння результатів")
    
    print("\nПриклад використання:")
    print("""
# Збереження результатів
from evaluation_storage import quick_save, EvaluationStorage

# Швидке збереження
filepath = quick_save(
    results_df=simulation_results,
    eval_results=evaluation_results, 
    analysis_data=analysis_data,
    params=simulation_params,
    description="Тест нових параметрів MPC"
)

# Розширене збереження з різними форматами
storage = EvaluationStorage("my_results")

package = storage.create_evaluation_package(
    results_df=results_df,
    eval_results=eval_results,
    analysis_data=analysis_data,
    params=params,
    description="Експеримент з Trust Region"
)

# Збереження у різних форматах
zip_path = storage.save_evaluation(package, 'zip')
excel_path = storage.save_evaluation(package, 'excel') 
json_path = storage.save_evaluation(package, 'json')

# Завантаження та аналіз
loaded_package = storage.load_evaluation(zip_path)
print(f"Загальна оцінка: {loaded_package.evaluation_results.overall_score}")

# Пошук збережених результатів
saved_evals = storage.list_saved_evaluations()
print(saved_evals)

# Пошук за критеріями
good_results = storage.find_evaluations(min_score=80.0)
recent_results = storage.find_evaluations(date_from="2024-01-01")

# Порівняння результатів
comparison = storage.compare_evaluations_from_files([
    "eval1.zip", "eval2.zip", "eval3.zip"
])
print(comparison)

# Генерація звітів
html_report = storage.generate_report(package, 'html')
md_report = storage.generate_report(package, 'markdown')

# Очищення старих файлів
deleted_count = storage.cleanup_old_evaluations(days_old=30)
print(f"Видалено {deleted_count} старих файлів")
""")
    
    print("\nПідтримувані формати:")
    print("- ZIP: Повний архів з усіма даними (рекомендується)")
    print("- JSON: Текстовий формат, легко читається")
    print("- Pickle: Найшвидший, повністю зберігає Python об'єкти")
    print("- Excel: Зручний для перегляду, кілька листів")
    print("- CSV: Базові дані в папці")
    
    print("\nМожливості:")
    print("✅ Автоматична генерація ID симуляції")
    print("✅ Перевірка цілісності даних (хешування)")
    print("✅ Метадані та параметри симуляції")
    print("✅ Пошук та фільтрація збережених результатів")
    print("✅ Порівняння результатів з різних експериментів")
    print("✅ Генерація HTML/Markdown звітів")
    print("✅ Автоматичне очищення старих файлів")
    print("✅ Логування всіх операцій")
    print("✅ Підтримка великих файлів через ZIP стиснення")
    
    # Демонстрація створення тестового пакета
    try:
        import pandas as pd
        import numpy as np
        
        print("\n🧪 ДЕМОНСТРАЦІЯ:")
        
        # Створюємо тестові дані
        test_results_df = pd.DataFrame({
            'conc_fe': np.random.normal(53.5, 0.5, 100),
            'conc_mass': np.random.normal(57.0, 1.0, 100),
            'solid_feed_percent': np.random.normal(75.0, 2.0, 100)
        })
        
        test_eval_results = EvaluationResults(
            overall_score=85.2,
            process_stability=0.92,
            model_r2_fe=0.89,
            model_r2_mass=0.91,
            ekf_consistency=0.75,
            trust_stability_index=0.88
        )
        
        test_analysis_data = {
            'timing_metrics': {'initial_training_time': 2.5},
            'trust_region_stats': [{'current_radius': 1.0}] * 50,
            'innovation_seq': np.random.normal(0, 0.1, (100, 2)).tolist()
        }
        
        test_params = {
            'ref_fe': 53.5,
            'ref_mass': 57.0,
            'horizon': 10,
            'lambda_u': 0.1
        }
        
        # Створюємо storage
        storage = EvaluationStorage("demo_results")
        
        # Створюємо пакет
        package = storage.create_evaluation_package(
            results_df=test_results_df,
            eval_results=test_eval_results,
            analysis_data=test_analysis_data,
            params=test_params,
            description="Демонстраційний експеримент"
        )
        
        print(f"Створено тестовий пакет:")
        print(f"  ID: {package.metadata.simulation_id}")
        print(f"  Оцінка: {package.evaluation_results.overall_score}")
        print(f"  Кроки: {package.metadata.simulation_steps}")
        
        # Збереження
        saved_path = storage.save_evaluation(package, 'zip')
        print(f"  Збережено: {saved_path}")
        
        # Завантаження
        loaded = storage.load_evaluation(saved_path)
        print(f"  Завантажено, оцінка: {loaded.evaluation_results.overall_score}")
        
        print("✅ Демонстрація успішна!")
        
    except ImportError as e:
        print(f"⚠️ Для демонстрації потрібні pandas/numpy: {e}")
    except Exception as e:
        print(f"❌ Помилка демонстрації: {e}")