# evaluation_database.py - База даних результатів оцінювання MPC

import sqlite3
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

try:
    from evaluation_storage import EvaluationPackage, EvaluationResults, SimulationMetadata
except ImportError:
    print("⚠️ Не вдалося імпортувати evaluation_storage. Деякі функції можуть бути недоступні.")
    EvaluationPackage = None
    EvaluationResults = None
    SimulationMetadata = None

# =============================================================================
# === СТРУКТУРИ ДАНИХ ДЛЯ БД ===
# =============================================================================

@dataclass
class ExperimentSeries:
    """Серія експериментів для групування пов'язаних симуляцій"""
    
    series_id: str
    name: str
    description: str
    created_at: str
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass 
class EvaluationRecord:
    """Запис оцінювання в базі даних"""
    
    id: Optional[int] = None
    simulation_id: str = ""
    series_id: Optional[str] = None
    timestamp: str = ""
    description: str = ""
    
    # Основні параметри
    simulation_steps: int = 0
    ref_fe: float = 0.0
    ref_mass: float = 0.0
    horizon: int = 0
    delta_u_max: float = 0.0
    lambda_u: float = 0.0
    
    # Ключові результати
    overall_score: float = 0.0
    process_stability: float = 0.0
    model_r2_fe: float = 0.0
    model_r2_mass: float = 0.0
    ekf_consistency: float = 0.0
    trust_stability_index: float = 0.0
    setpoint_achievement_fe: float = 0.0
    setpoint_achievement_mass: float = 0.0
    
    # Часові метрики
    initial_training_time: float = 0.0
    avg_prediction_time: float = 0.0
    
    # Шлях до файлу з повними даними
    file_path: str = ""
    file_format: str = ""
    file_size_mb: float = 0.0
    
    # Мета інформація
    data_hash: str = ""
    tags: str = ""  # JSON список тегів
    notes: str = ""

# =============================================================================
# === МЕНЕДЖЕР БАЗИ ДАНИХ ===
# =============================================================================

class EvaluationDatabase:
    """Менеджер бази даних для збереження та пошуку результатів оцінювання"""
    
    def __init__(self, db_path: str = "evaluation_results.db"):
        """
        Ініціалізація бази даних
        
        Args:
            db_path: Шлях до файлу бази даних SQLite
        """
        self.db_path = Path(db_path)
        self.connection = None
        
        # Налаштування логування
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Створюємо/підключаємось до БД
        self._initialize_database()
    
    def _initialize_database(self):
        """Створює таблиці бази даних якщо вони не існують"""
        
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row  # Для зручного доступу до колонок
        
        # Створюємо таблиці
        self._create_tables()
        
        self.logger.info(f"База даних ініціалізована: {self.db_path}")
    
    def _create_tables(self):
        """Створює таблиці в базі даних"""
        
        cursor = self.connection.cursor()
        
        # Таблиця серій експериментів
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_series (
                series_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                tags TEXT,
                UNIQUE(series_id)
            )
        """)
        
        # Основна таблиця оцінювань
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id TEXT UNIQUE NOT NULL,
                series_id TEXT,
                timestamp TEXT NOT NULL,
                description TEXT,
                
                -- Параметри симуляції
                simulation_steps INTEGER,
                ref_fe REAL,
                ref_mass REAL,
                horizon INTEGER,
                delta_u_max REAL,
                lambda_u REAL,
                
                -- Ключові результати
                overall_score REAL,
                process_stability REAL,
                model_r2_fe REAL,
                model_r2_mass REAL,
                ekf_consistency REAL,
                trust_stability_index REAL,
                setpoint_achievement_fe REAL,
                setpoint_achievement_mass REAL,
                
                -- Часові метрики
                initial_training_time REAL,
                avg_prediction_time REAL,
                
                -- Файл з даними
                file_path TEXT,
                file_format TEXT,
                file_size_mb REAL,
                
                -- Мета інформація
                data_hash TEXT,
                tags TEXT,
                notes TEXT,
                
                FOREIGN KEY (series_id) REFERENCES experiment_series (series_id)
            )
        """)
        
        # Додаткова таблиця для повних метрик (якщо потрібно)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detailed_metrics (
                evaluation_id INTEGER,
                metric_name TEXT,
                metric_value REAL,
                metric_category TEXT,
                PRIMARY KEY (evaluation_id, metric_name),
                FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
            )
        """)
        
        # Таблиця тегів для швидкого пошуку
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_tags (
                evaluation_id INTEGER,
                tag TEXT,
                PRIMARY KEY (evaluation_id, tag),
                FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
            )
        """)
        
        # Індекси для швидкого пошуку
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_overall_score ON evaluations (overall_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluations (timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_series ON evaluations (series_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags ON evaluation_tags (tag)")
        
        self.connection.commit()
    
    def close(self):
        """Закриває з'єднання з базою даних"""
        if self.connection:
            self.connection.close()
            self.logger.info("З'єднання з базою даних закрито")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # =============================================================================
    # === ОПЕРАЦІЇ З СЕРІЯМИ ЕКСПЕРИМЕНТІВ ===
    # =============================================================================
    
    def create_experiment_series(self, series_id: str, name: str, 
                               description: str = "", tags: List[str] = None) -> bool:
        """
        Створює нову серію експериментів
        
        Args:
            series_id: Унікальний ID серії
            name: Назва серії
            description: Опис серії
            tags: Список тегів
            
        Returns:
            True якщо серія створена успішно
        """
        
        try:
            cursor = self.connection.cursor()
            
            series = ExperimentSeries(
                series_id=series_id,
                name=name,
                description=description,
                created_at=datetime.now().isoformat(),
                tags=tags or []
            )
            
            cursor.execute("""
                INSERT INTO experiment_series 
                (series_id, name, description, created_at, tags)
                VALUES (?, ?, ?, ?, ?)
            """, (
                series.series_id,
                series.name,
                series.description,
                series.created_at,
                json.dumps(series.tags)
            ))
            
            self.connection.commit()
            self.logger.info(f"Створена серія експериментів: {series_id}")
            return True
            
        except sqlite3.IntegrityError:
            self.logger.warning(f"Серія {series_id} вже існує")
            return False
        except Exception as e:
            self.logger.error(f"Помилка створення серії {series_id}: {e}")
            return False
    
    def list_experiment_series(self) -> pd.DataFrame:
        """Повертає список всіх серій експериментів"""
        
        query = """
            SELECT s.*, COUNT(e.id) as evaluation_count
            FROM experiment_series s
            LEFT JOIN evaluations e ON s.series_id = e.series_id
            GROUP BY s.series_id
            ORDER BY s.created_at DESC
        """
        
        return pd.read_sql_query(query, self.connection)
    
    # =============================================================================
    # === ОСНОВНІ ОПЕРАЦІЇ З ОЦІНЮВАННЯМИ ===
    # =============================================================================
    
    def add_evaluation(self, package: 'EvaluationPackage', 
                      series_id: Optional[str] = None,
                      tags: List[str] = None,
                      notes: str = "") -> int:
        """
        Додає результати оцінювання до бази даних
        
        Args:
            package: Пакет з результатами оцінювання
            series_id: ID серії експериментів
            tags: Додаткові теги
            notes: Примітки
            
        Returns:
            ID створеного запису в БД
        """
        
        if package is None:
            raise ValueError("Package не може бути None")
        
        cursor = self.connection.cursor()
        
        # Створюємо запис
        record = EvaluationRecord(
            simulation_id=package.metadata.simulation_id,
            series_id=series_id,
            timestamp=package.metadata.timestamp,
            description=package.metadata.description,
            
            # Параметри
            simulation_steps=package.metadata.simulation_steps,
            ref_fe=package.metadata.ref_fe,
            ref_mass=package.metadata.ref_mass,
            horizon=package.metadata.horizon,
            delta_u_max=package.metadata.delta_u_max,
            lambda_u=package.parameters.get('lambda_u', 0.0),
            
            # Результати
            overall_score=package.evaluation_results.overall_score,
            process_stability=package.evaluation_results.process_stability,
            model_r2_fe=package.evaluation_results.model_r2_fe,
            model_r2_mass=package.evaluation_results.model_r2_mass,
            ekf_consistency=package.evaluation_results.ekf_consistency,
            trust_stability_index=package.evaluation_results.trust_stability_index,
            setpoint_achievement_fe=package.evaluation_results.setpoint_achievement_fe,
            setpoint_achievement_mass=package.evaluation_results.setpoint_achievement_mass,
            
            # Час
            initial_training_time=package.evaluation_results.initial_training_time,
            avg_prediction_time=package.evaluation_results.avg_prediction_time,
            
            # Мета дані
            data_hash=package.metadata.data_hash,
            tags=json.dumps(tags or []),
            notes=notes
        )
        
        # Вставляємо запис
        cursor.execute("""
            INSERT INTO evaluations (
                simulation_id, series_id, timestamp, description,
                simulation_steps, ref_fe, ref_mass, horizon, delta_u_max, lambda_u,
                overall_score, process_stability, model_r2_fe, model_r2_mass,
                ekf_consistency, trust_stability_index, 
                setpoint_achievement_fe, setpoint_achievement_mass,
                initial_training_time, avg_prediction_time,
                file_path, file_format, file_size_mb,
                data_hash, tags, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.simulation_id, record.series_id, record.timestamp, record.description,
            record.simulation_steps, record.ref_fe, record.ref_mass, 
            record.horizon, record.delta_u_max, record.lambda_u,
            record.overall_score, record.process_stability, 
            record.model_r2_fe, record.model_r2_mass,
            record.ekf_consistency, record.trust_stability_index,
            record.setpoint_achievement_fe, record.setpoint_achievement_mass,
            record.initial_training_time, record.avg_prediction_time,
            record.file_path, record.file_format, record.file_size_mb,
            record.data_hash, record.tags, record.notes
        ))
        
        evaluation_id = cursor.lastrowid
        
        # Додаємо детальні метрики
        self._add_detailed_metrics(evaluation_id, package.evaluation_results)
        
        # Додаємо теги
        if tags:
            self._add_tags(evaluation_id, tags)
        
        self.connection.commit()
        self.logger.info(f"Додано оцінювання: {package.metadata.simulation_id} (ID: {evaluation_id})")
        
        return evaluation_id
    
    def _add_detailed_metrics(self, evaluation_id: int, eval_results: 'EvaluationResults'):
        """Додає детальні метрики до окремої таблиці"""
        
        cursor = self.connection.cursor()
        
        # Категорії метрик
        metric_categories = {
            'model': ['model_rmse_fe', 'model_rmse_mass', 'model_mae_fe', 'model_mae_mass', 
                     'model_mape_fe', 'model_mape_mass', 'model_bias_fe', 'model_bias_mass'],
            'ekf': ['ekf_rmse_fe', 'ekf_rmse_mass', 'ekf_normalized_rmse_fe', 
                   'ekf_normalized_rmse_mass', 'ekf_rmse_total', 'ekf_nees_mean', 'ekf_nis_mean'],
            'trust_region': ['trust_radius_mean', 'trust_radius_std', 'trust_radius_min', 
                           'trust_radius_max', 'trust_adaptivity_coeff'],
            'tracking': ['tracking_error_fe', 'tracking_error_mass', 'tracking_mae_fe', 
                        'tracking_mae_mass', 'tracking_mape_fe', 'tracking_mape_mass',
                        'ise_fe', 'ise_mass', 'iae_fe', 'iae_mass'],
            'control': ['control_smoothness', 'control_aggressiveness', 'control_variability',
                       'control_energy', 'control_stability_index', 'control_utilization',
                       'significant_changes_frequency', 'max_control_change']
        }
        
        eval_dict = eval_results.to_dict()
        
        for category, metrics in metric_categories.items():
            for metric in metrics:
                if metric in eval_dict:
                    cursor.execute("""
                        INSERT INTO detailed_metrics (evaluation_id, metric_name, metric_value, metric_category)
                        VALUES (?, ?, ?, ?)
                    """, (evaluation_id, metric, eval_dict[metric], category))
    
    def _add_tags(self, evaluation_id: int, tags: List[str]):
        """Додає теги до таблиці тегів"""
        
        cursor = self.connection.cursor()
        
        for tag in tags:
            cursor.execute("""
                INSERT OR IGNORE INTO evaluation_tags (evaluation_id, tag)
                VALUES (?, ?)
            """, (evaluation_id, tag.strip().lower()))
    
    def update_file_info(self, simulation_id: str, file_path: str, 
                        file_format: str, file_size_mb: float):
        """Оновлює інформацію про файл з даними"""
        
        cursor = self.connection.cursor()
        
        cursor.execute("""
            UPDATE evaluations 
            SET file_path = ?, file_format = ?, file_size_mb = ?
            WHERE simulation_id = ?
        """, (file_path, file_format, file_size_mb, simulation_id))
        
        self.connection.commit()
    
    # =============================================================================
    # === ПОШУК ТА ФІЛЬТРАЦІЯ ===
    # =============================================================================
    
    def search_evaluations(self, 
                          series_id: Optional[str] = None,
                          min_score: Optional[float] = None,
                          max_score: Optional[float] = None,
                          tags: Optional[List[str]] = None,
                          date_from: Optional[str] = None,
                          date_to: Optional[str] = None,
                          text_search: Optional[str] = None,
                          limit: int = 100) -> pd.DataFrame:
        """
        Пошук оцінювань за критеріями
        
        Args:
            series_id: ID серії експериментів
            min_score: Мінімальна загальна оцінка
            max_score: Максимальна загальна оцінка
            tags: Список тегів (OR логіка)
            date_from: Дата від (ISO формат)
            date_to: Дата до (ISO формат)
            text_search: Пошук в описі та примітках
            limit: Максимальна кількість результатів
            
        Returns:
            DataFrame з результатами пошуку
        """
        
        query_parts = ["SELECT DISTINCT e.* FROM evaluations e"]
        where_conditions = []
        params = []
        
        # JOIN з тегами якщо потрібно
        if tags:
            query_parts.append("JOIN evaluation_tags et ON e.id = et.evaluation_id")
        
        # Умови фільтрації
        if series_id:
            where_conditions.append("e.series_id = ?")
            params.append(series_id)
        
        if min_score is not None:
            where_conditions.append("e.overall_score >= ?")
            params.append(min_score)
        
        if max_score is not None:
            where_conditions.append("e.overall_score <= ?")
            params.append(max_score)
        
        if date_from:
            where_conditions.append("e.timestamp >= ?")
            params.append(date_from)
        
        if date_to:
            where_conditions.append("e.timestamp <= ?")
            params.append(date_to)
        
        if text_search:
            where_conditions.append("(e.description LIKE ? OR e.notes LIKE ?)")
            search_term = f"%{text_search}%"
            params.extend([search_term, search_term])
        
        if tags:
            tag_placeholders = ",".join("?" * len(tags))
            where_conditions.append(f"et.tag IN ({tag_placeholders})")
            params.extend([tag.strip().lower() for tag in tags])
        
        # Складаємо запит
        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
        
        query_parts.append("ORDER BY e.overall_score DESC, e.timestamp DESC")
        query_parts.append(f"LIMIT {limit}")
        
        query = " ".join(query_parts)
        
        return pd.read_sql_query(query, self.connection, params=params)
    
    def get_best_evaluations(self, limit: int = 10, 
                           series_id: Optional[str] = None) -> pd.DataFrame:
        """Повертає найкращі оцінювання за загальною оцінкою"""
        
        return self.search_evaluations(
            series_id=series_id,
            limit=limit
        ).head(limit)
    
    def get_recent_evaluations(self, days: int = 7, 
                             limit: int = 20) -> pd.DataFrame:
        """Повертає недавні оцінювання"""
        
        cutoff_date = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        
        return self.search_evaluations(
            date_from=cutoff_date,
            limit=limit
        )
    
    def get_evaluation_by_id(self, evaluation_id: int) -> Optional[Dict]:
        """Повертає оцінювання за ID"""
        
        cursor = self.connection.cursor()
        
        cursor.execute("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_evaluation_by_simulation_id(self, simulation_id: str) -> Optional[Dict]:
        """Повертає оцінювання за simulation_id"""
        
        cursor = self.connection.cursor()
        
        cursor.execute("SELECT * FROM evaluations WHERE simulation_id = ?", (simulation_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    # =============================================================================
    # === АНАЛІТИКА ТА ЗВІТИ ===
    # =============================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Повертає загальну статистику бази даних"""
        
        cursor = self.connection.cursor()
        
        stats = {}
        
        # Загальна кількість
        cursor.execute("SELECT COUNT(*) FROM evaluations")
        stats['total_evaluations'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM experiment_series")
        stats['total_series'] = cursor.fetchone()[0]
        
        # Статистика оцінок
        cursor.execute("""
            SELECT 
                AVG(overall_score) as avg_score,
                MIN(overall_score) as min_score,
                MAX(overall_score) as max_score,
                AVG(process_stability) as avg_stability,
                AVG(ekf_consistency) as avg_ekf_consistency,
                AVG(trust_stability_index) as avg_trust_stability
            FROM evaluations
        """)
        score_stats = cursor.fetchone()
        if score_stats:
            stats.update({
                'avg_score': round(score_stats[0] or 0, 2),
                'min_score': round(score_stats[1] or 0, 2),
                'max_score': round(score_stats[2] or 0, 2),
                'avg_stability': round(score_stats[3] or 0, 3),
                'avg_ekf_consistency': round(score_stats[4] or 0, 3),
                'avg_trust_stability': round(score_stats[5] or 0, 3)
            })
        
        # Топ теги
        cursor.execute("""
            SELECT tag, COUNT(*) as count 
            FROM evaluation_tags 
            GROUP BY tag 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_tags = cursor.fetchall()
        stats['top_tags'] = [(tag, count) for tag, count in top_tags]
        
        # Динаміка по часу (останні 30 днів)
        cutoff_date = (datetime.now() - pd.Timedelta(days=30)).isoformat()
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as count,
                AVG(overall_score) as avg_score
            FROM evaluations 
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """, (cutoff_date,))
        recent_activity = cursor.fetchall()
        stats['recent_activity'] = [(date, count, round(score or 0, 1)) 
                                   for date, count, score in recent_activity]
        
        return stats
    
    def get_series_comparison(self) -> pd.DataFrame:
        """Порівняння серій експериментів"""
        
        query = """
            SELECT 
                s.series_id,
                s.name,
                COUNT(e.id) as evaluation_count,
                AVG(e.overall_score) as avg_score,
                MAX(e.overall_score) as best_score,
                AVG(e.process_stability) as avg_stability,
                AVG(e.ekf_consistency) as avg_ekf_consistency,
                AVG(e.initial_training_time) as avg_training_time,
                AVG(e.avg_prediction_time) as avg_prediction_time
            FROM experiment_series s
            LEFT JOIN evaluations e ON s.series_id = e.series_id
            GROUP BY s.series_id, s.name
            ORDER BY avg_score DESC
        """
        
        return pd.read_sql_query(query, self.connection)
    
    def get_parameter_analysis(self) -> pd.DataFrame:
        """Аналіз впливу параметрів на результати"""
        
        query = """
            SELECT 
                horizon,
                delta_u_max,
                lambda_u,
                COUNT(*) as count,
                AVG(overall_score) as avg_score,
                AVG(process_stability) as avg_stability,
                AVG(ekf_consistency) as avg_ekf,
                AVG(trust_stability_index) as avg_trust
            FROM evaluations
            GROUP BY horizon, delta_u_max, lambda_u
            HAVING count >= 2
            ORDER BY avg_score DESC
        """
        
        return pd.read_sql_query(query, self.connection)
    
    def get_performance_trends(self, metric: str = 'overall_score', 
                             days: int = 90) -> pd.DataFrame:
        """Аналіз трендів продуктивності за часом"""
        
        cutoff_date = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        
        query = f"""
            SELECT 
                DATE(timestamp) as date,
                AVG({metric}) as avg_value,
                MIN({metric}) as min_value,
                MAX({metric}) as max_value,
                COUNT(*) as evaluation_count
            FROM evaluations
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """
        
        return pd.read_sql_query(query, self.connection, params=[cutoff_date])
    
    # =============================================================================
    # === EXPORT/IMPORT ===
    # =============================================================================
    
    def export_to_csv(self, output_dir: str = "database_export") -> List[str]:
        """Експортує всю базу даних в CSV файли"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        exported_files = []
        
        # Експорт основних таблиць
        tables = {
            'evaluations': 'SELECT * FROM evaluations',
            'experiment_series': 'SELECT * FROM experiment_series',
            'detailed_metrics': 'SELECT * FROM detailed_metrics',
            'evaluation_tags': 'SELECT * FROM evaluation_tags'
        }
        
        for table_name, query in tables.items():
            df = pd.read_sql_query(query, self.connection)
            
            if not df.empty:
                file_path = output_path / f"{table_name}.csv"
                df.to_csv(file_path, index=False)
                exported_files.append(str(file_path))
                self.logger.info(f"Експортовано {table_name}: {len(df)} записів")
        
        # Експорт аналітики
        stats = self.get_statistics()
        stats_df = pd.DataFrame([stats])
        stats_path = output_path / "statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        exported_files.append(str(stats_path))
        
        return exported_files
    
    def backup_database(self, backup_path: str = None) -> str:
        """Створює резервну копію бази даних"""
        
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"evaluation_backup_{timestamp}.db"
        
        # Простий спосіб - копіювання файлу
        import shutil
        shutil.copy2(self.db_path, backup_path)
        
        self.logger.info(f"Резервна копія створена: {backup_path}")
        return backup_path
    
    # =============================================================================
    # === УПРАВЛІННЯ ДАНИМИ ===
    # =============================================================================
    
    def delete_evaluation(self, evaluation_id: int) -> bool:
        """Видаляє оцінювання за ID"""
        
        try:
            cursor = self.connection.cursor()
            
            # Видаляємо пов'язані дані
            cursor.execute("DELETE FROM detailed_metrics WHERE evaluation_id = ?", (evaluation_id,))
            cursor.execute("DELETE FROM evaluation_tags WHERE evaluation_id = ?", (evaluation_id,))
            cursor.execute("DELETE FROM evaluations WHERE id = ?", (evaluation_id,))
            
            self.connection.commit()
            
            if cursor.rowcount > 0:
                self.logger.info(f"Видалено оцінювання ID: {evaluation_id}")
                return True
            else:
                self.logger.warning(f"Оцінювання ID {evaluation_id} не знайдено")
                return False
                
        except Exception as e:
            self.logger.error(f"Помилка видалення оцінювання {evaluation_id}: {e}")
            return False
    
    def cleanup_old_evaluations(self, days_old: int = 90) -> int:
        """Видаляє старі оцінювання"""
        
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_old)).isoformat()
        
        # Отримуємо ID старих записів
        old_ids = pd.read_sql_query("""
            SELECT id FROM evaluations WHERE timestamp < ?
        """, self.connection, params=[cutoff_date])
        
        deleted_count = 0
        for _, row in old_ids.iterrows():
            if self.delete_evaluation(row['id']):
                deleted_count += 1
        
        self.logger.info(f"Видалено {deleted_count} старих оцінювань (старше {days_old} днів)")
        return deleted_count
    
    def update_tags(self, evaluation_id: int, tags: List[str]) -> bool:
        """Оновлює теги для оцінювання"""
        
        try:
            cursor = self.connection.cursor()
            
            # Видаляємо старі теги
            cursor.execute("DELETE FROM evaluation_tags WHERE evaluation_id = ?", (evaluation_id,))
            
            # Додаємо нові
            self._add_tags(evaluation_id, tags)
            
            # Оновлюємо JSON в основній таблиці
            cursor.execute("""
                UPDATE evaluations SET tags = ? WHERE id = ?
            """, (json.dumps(tags), evaluation_id))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Помилка оновлення тегів для {evaluation_id}: {e}")
            return False
    
    def add_notes(self, evaluation_id: int, notes: str) -> bool:
        """Додає або оновлює примітки до оцінювання"""
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                UPDATE evaluations SET notes = ? WHERE id = ?
            """, (notes, evaluation_id))
            
            self.connection.commit()
            
            if cursor.rowcount > 0:
                self.logger.info(f"Оновлено примітки для оцінювання {evaluation_id}")
                return True
            else:
                self.logger.warning(f"Оцінювання {evaluation_id} не знайдено")
                return False
                
        except Exception as e:
            self.logger.error(f"Помилка оновлення приміток для {evaluation_id}: {e}")
            return False

# =============================================================================
# === УТІЛІТИ ТА ІНТЕГРАЦІЯ ===
# =============================================================================

class EvaluationAnalyzer:
    """Аналітичні інструменти для роботи з базою даних оцінювань"""
    
    def __init__(self, database: EvaluationDatabase):
        self.db = database
    
    def find_optimal_parameters(self, target_metric: str = 'overall_score',
                              min_evaluations: int = 3) -> pd.DataFrame:
        """
        Знаходить оптимальні параметри на основі історичних даних
        
        Args:
            target_metric: Метрика для оптимізації
            min_evaluations: Мінімальна кількість оцінювань для групи
            
        Returns:
            DataFrame з рекомендованими параметрами
        """
        
        query = f"""
            SELECT 
                horizon,
                delta_u_max,
                lambda_u,
                ref_fe,
                ref_mass,
                COUNT(*) as evaluation_count,
                AVG({target_metric}) as avg_target,
                STDEV({target_metric}) as std_target,
                MAX({target_metric}) as best_target
            FROM evaluations
            GROUP BY horizon, delta_u_max, lambda_u, ref_fe, ref_mass
            HAVING evaluation_count >= ?
            ORDER BY avg_target DESC, std_target ASC
        """
        
        try:
            return pd.read_sql_query(query, self.db.connection, params=[min_evaluations])
        except Exception as e:
            # Fallback без STDEV функції (SQLite може не підтримувати)
            query_simple = f"""
                SELECT 
                    horizon,
                    delta_u_max,
                    lambda_u,
                    ref_fe,
                    ref_mass,
                    COUNT(*) as evaluation_count,
                    AVG({target_metric}) as avg_target,
                    MAX({target_metric}) as best_target
                FROM evaluations
                GROUP BY horizon, delta_u_max, lambda_u, ref_fe, ref_mass
                HAVING evaluation_count >= ?
                ORDER BY avg_target DESC
            """
            return pd.read_sql_query(query_simple, self.db.connection, params=[min_evaluations])
    
    def analyze_failure_patterns(self, score_threshold: float = 50.0) -> Dict[str, Any]:
        """Аналізує патерни неуспішних симуляцій"""
        
        # Отримуємо неуспішні симуляції
        failed = self.db.search_evaluations(max_score=score_threshold)
        successful = self.db.search_evaluations(min_score=score_threshold + 20)
        
        analysis = {
            'failed_count': len(failed),
            'successful_count': len(successful),
            'failure_rate': len(failed) / (len(failed) + len(successful)) if (len(failed) + len(successful)) > 0 else 0
        }
        
        if len(failed) > 0:
            analysis['common_failed_params'] = {
                'avg_horizon': failed['horizon'].mean(),
                'avg_delta_u_max': failed['delta_u_max'].mean(),
                'avg_lambda_u': failed['lambda_u'].mean(),
                'avg_ekf_consistency': failed['ekf_consistency'].mean(),
                'avg_trust_stability': failed['trust_stability_index'].mean()
            }
        
        if len(successful) > 0:
            analysis['common_successful_params'] = {
                'avg_horizon': successful['horizon'].mean(),
                'avg_delta_u_max': successful['delta_u_max'].mean(),
                'avg_lambda_u': successful['lambda_u'].mean(),
                'avg_ekf_consistency': successful['ekf_consistency'].mean(),
                'avg_trust_stability': successful['trust_stability_index'].mean()
            }
        
        return analysis
    
    def suggest_experiments(self, current_best_score: float = None) -> List[Dict]:
        """Пропонує нові експерименти на основі gap analysis"""
        
        if current_best_score is None:
            # Знаходимо поточний найкращий результат
            best = self.db.get_best_evaluations(limit=1)
            if len(best) > 0:
                current_best_score = best.iloc[0]['overall_score']
            else:
                current_best_score = 0.0
        
        suggestions = []
        
        # Аналіз недосліджених параметрів
        param_ranges = pd.read_sql_query("""
            SELECT 
                MIN(horizon) as min_horizon, MAX(horizon) as max_horizon,
                MIN(delta_u_max) as min_delta_u, MAX(delta_u_max) as max_delta_u,
                MIN(lambda_u) as min_lambda, MAX(lambda_u) as max_lambda
            FROM evaluations
        """, self.db.connection)
        
        if len(param_ranges) > 0:
            ranges = param_ranges.iloc[0]
            
            # Пропозиції на основі поточних діапазонів
            suggestions.extend([
                {
                    'type': 'parameter_exploration',
                    'description': 'Дослідити більший горизонт прогнозування',
                    'params': {'horizon': int(ranges['max_horizon'] * 1.5)},
                    'rationale': 'Збільшення горизонту може покращити довгострокове керування'
                },
                {
                    'type': 'parameter_exploration', 
                    'description': 'Дослідити менший lambda_u для більшої агресивності',
                    'params': {'lambda_u': float(ranges['min_lambda'] * 0.5)},
                    'rationale': 'Зменшення регуляризації може покращити відстеження'
                },
                {
                    'type': 'robustness_test',
                    'description': 'Тестування стійкості з різними збуреннями',
                    'params': {'add_disturbances': True},
                    'rationale': 'Перевірка роботи в умовах невизначеності'
                }
            ])
        
        return suggestions

# =============================================================================
# === ФУНКЦІЇ ВИСОКОГО РІВНЯ ===
# =============================================================================

def create_evaluation_database(db_path: str = "evaluation_results.db") -> EvaluationDatabase:
    """Створює або підключається до бази даних оцінювань"""
    return EvaluationDatabase(db_path)

def quick_add_to_database(package: 'EvaluationPackage', 
                         db_path: str = "evaluation_results.db",
                         series_id: Optional[str] = None,
                         tags: List[str] = None) -> int:
    """
    Швидко додає результати до бази даних
    
    Args:
        package: Пакет з результатами оцінювання
        db_path: Шлях до бази даних
        series_id: ID серії експериментів
        tags: Теги для оцінювання
        
    Returns:
        ID створеного запису
    """
    
    with EvaluationDatabase(db_path) as db:
        return db.add_evaluation(package, series_id=series_id, tags=tags)

def search_database(db_path: str = "evaluation_results.db", **kwargs) -> pd.DataFrame:
    """Пошук в базі даних з параметрами"""
    
    with EvaluationDatabase(db_path) as db:
        return db.search_evaluations(**kwargs)

def get_database_stats(db_path: str = "evaluation_results.db") -> Dict[str, Any]:
    """Отримує статистику бази даних"""
    
    with EvaluationDatabase(db_path) as db:
        return db.get_statistics()

# =============================================================================
# === ПРИКЛАД ВИКОРИСТАННЯ ===
# =============================================================================

if __name__ == "__main__":
    print("🗄️ evaluation_database.py - База даних результатів оцінювання")
    print("\nОсновні функції:")
    print("1. EvaluationDatabase() - Повний менеджер БД")
    print("2. quick_add_to_database() - Швидке додавання")
    print("3. search_database() - Пошук результатів")
    print("4. EvaluationAnalyzer() - Аналітичні інструменти")
    
    print("\nПриклад використання:")
    print("""
# Створення бази даних
from evaluation_database import EvaluationDatabase, EvaluationAnalyzer

# Основна робота з БД
db = EvaluationDatabase("my_experiments.db")

# Створення серії експериментів
db.create_experiment_series(
    series_id="trust_region_study",
    name="Дослідження Trust Region параметрів",
    description="Систематичне вивчення впливу Trust Region на ефективність",
    tags=["trust_region", "optimization", "mpc"]
)

# Додавання результатів (припускаємо що package вже створений)
eval_id = db.add_evaluation(
    package=evaluation_package,
    series_id="trust_region_study", 
    tags=["baseline", "test_run"],
    notes="Початковий тест з стандартними параметрами"
)

# Пошук найкращих результатів
best_results = db.get_best_evaluations(limit=5)
print("Топ 5 результатів:")
print(best_results[['simulation_id', 'overall_score', 'description']])

# Пошук за критеріями
high_ekf = db.search_evaluations(
    min_score=70.0,
    tags=["trust_region"],
    date_from="2024-01-01"
)

# Аналітика
analyzer = EvaluationAnalyzer(db)

# Знаходження оптимальних параметрів
optimal_params = analyzer.find_optimal_parameters('overall_score')
print("Оптимальні параметри:")
print(optimal_params.head())

# Аналіз неуспішних симуляцій  
failures = analyzer.analyze_failure_patterns(score_threshold=60.0)
print(f"Частота невдач: {failures['failure_rate']:.2%}")

# Пропозиції нових експериментів
suggestions = analyzer.suggest_experiments()
for suggestion in suggestions:
    print(f"- {suggestion['description']}: {suggestion['rationale']}")

# Статистика бази даних
stats = db.get_statistics()
print(f"Всього оцінювань: {stats['total_evaluations']}")
print(f"Середня оцінка: {stats['avg_score']}")
print(f"Найкращі теги: {stats['top_tags'][:3]}")

# Порівняння серій
series_comparison = db.get_series_comparison()
print(series_comparison)

# Тренди продуктивності
trends = db.get_performance_trends('overall_score', days=30)
print("Тренди за 30 днів:")
print(trends)

# Експорт даних
exported_files = db.export_to_csv("analysis_export")
print(f"Експортовано файли: {exported_files}")

# Резервна копія
backup_path = db.backup_database()
print(f"Резервна копія: {backup_path}")

db.close()
""")
    
    print("\nМожливості БД:")
    print("✅ Збереження метаданих та ключових метрик")
    print("✅ Організація в серії експериментів")
    print("✅ Теги та повнотекстовий пошук")
    print("✅ Аналітика та статистика")
    print("✅ Пошук оптимальних параметрів")
    print("✅ Аналіз паттернів невдач")
    print("✅ Пропозиції нових експериментів")
    print("✅ Експорт та резервне копіювання")
    print("✅ Тренди та порівняння серій")
    
    # Демонстрація створення БД
    try:
        print("\n🧪 ДЕМОНСТРАЦІЯ СТВОРЕННЯ БД:")
        
        # Створюємо тестову БД
        test_db = EvaluationDatabase("demo_evaluations.db")
        
        # Створюємо тестову серію
        test_db.create_experiment_series(
            series_id="demo_series",
            name="Демонстраційна серія",
            description="Тестова серія для демонстрації функціональності",
            tags=["demo", "test"]
        )
        
        print("✅ База даних створена")
        print("✅ Тестова серія додана")
        
        # Статистика
        stats = test_db.get_statistics()
        print(f"Серій експериментів: {stats['total_series']}")
        print(f"Оцінювань: {stats['total_evaluations']}")
        
        test_db.close()
        print("✅ Демонстрація завершена")
        
    except Exception as e:
        print(f"❌ Помилка демонстрації: {e}")