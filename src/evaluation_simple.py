# evaluation_simple.py - Простий модуль оцінювання ефективності MPC-симулятора

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from dataclasses import dataclass

# =============================================================================
# === СТРУКТУРИ ДАНИХ ===
# =============================================================================

@dataclass
class EvaluationResults:
    """Контейнер для всіх результатів оцінювання"""
    # Модель (6 метрик)
    model_rmse_fe: float
    model_rmse_mass: float
    model_r2_fe: float
    model_r2_mass: float
    model_bias_fe: float
    model_bias_mass: float
    
    # Керування (9 метрик)
    tracking_error_fe: float
    tracking_error_mass: float
    control_smoothness: float
    setpoint_achievement_fe: float  # % часу в межах толерантності
    setpoint_achievement_mass: float
    # Додаємо ISE/IAE метрики
    ise_fe: float           # Integral of Squared Error для Fe
    ise_mass: float         # Integral of Squared Error для Mass
    iae_fe: float           # Integral of Absolute Error для Fe
    iae_mass: float         # Integral of Absolute Error для Mass
    
    # Загальна ефективність (2 метрики)
    overall_score: float
    process_stability: float
    
    control_aggressiveness: float           # mean_delta_u або mean_abs_nonzero_delta_u
    control_variability: float              # std_delta_u
    control_energy: float                   # energy_u
    control_stability_index: float          # 1 - directional_switch_frequency
    control_utilization: float              # percentage_of_max_delta_u_used
    significant_changes_frequency: float    # significant_magnitude_changes_frequency
    significant_changes_count: float        # significant_magnitude_changes_count
    max_control_change: float               # max_abs_delta_u
    directional_switches_per_step: float    # directional_switch_frequency
    directional_switches_count: float       # directional_switch_count
    steps_at_max_delta_u: float            # num_steps_at_delta_u_max

    def to_dict(self) -> Dict:
        """Конвертує в словник для зручності"""
        return {
            'model_rmse_fe': self.model_rmse_fe,
            'model_rmse_mass': self.model_rmse_mass,
            'model_r2_fe': self.model_r2_fe,
            'model_r2_mass': self.model_r2_mass,
            'model_bias_fe': self.model_bias_fe,
            'model_bias_mass': self.model_bias_mass,
            'tracking_error_fe': self.tracking_error_fe,
            'tracking_error_mass': self.tracking_error_mass,
            'control_smoothness': self.control_smoothness,
            'setpoint_achievement_fe': self.setpoint_achievement_fe,
            'setpoint_achievement_mass': self.setpoint_achievement_mass,
            'ise_fe': self.ise_fe,
            'ise_mass': self.ise_mass,
            'iae_fe': self.iae_fe,
            'iae_mass': self.iae_mass,
            'overall_score': self.overall_score,
            'process_stability': self.process_stability,
            'control_aggressiveness': self.control_aggressiveness,
            'control_variability': self.control_variability,
            'control_energy': self.control_energy,
            'control_stability_index': self.control_stability_index,
            'control_utilization': self.control_utilization,
            'significant_changes_frequency': self.significant_changes_frequency,
            'significant_changes_count': self.significant_changes_count,
            'max_control_change': self.max_control_change,
            'directional_switches_per_step': self.directional_switches_per_step,
            'directional_switches_count': self.directional_switches_count,
            'steps_at_max_delta_u': self.steps_at_max_delta_u
        }

# =============================================================================
# === ФУНКЦІЇ ОЦІНЮВАННЯ МОДЕЛЕЙ ===
# =============================================================================

def evaluate_model_performance(results_df: pd.DataFrame, analysis_data: Dict) -> Dict[str, float]:
    """Оцінює якість роботи моделей прогнозування"""
    
    # Витягуємо дані для порівняння
    y_true = analysis_data.get('y_true_seq', [])
    y_pred = analysis_data.get('y_pred_seq', [])
    
    if not y_true or not y_pred:
        # Fallback: використовуємо інновації EKF як проксі для помилок моделі
        print("⚠️ Прямі дані моделі недоступні, використовуємо EKF інновації")
        innovations = analysis_data.get('innov', np.array([]))
        
        if len(innovations) > 0:
            # Використовуємо інновації як оцінку помилки моделі
            rmse_fe = np.sqrt(np.mean(innovations[:, 0]**2))
            rmse_mass = np.sqrt(np.mean(innovations[:, 1]**2))
            
            # Оцінюємо R² через дисперсію інновацій
            fe_values = results_df['conc_fe'].values
            mass_values = results_df['conc_mass'].values
            
            r2_fe = max(0, 1 - np.var(innovations[:len(fe_values), 0]) / np.var(fe_values))
            r2_mass = max(0, 1 - np.var(innovations[:len(mass_values), 1]) / np.var(mass_values))
            
            bias_fe = np.mean(innovations[:, 0])
            bias_mass = np.mean(innovations[:, 1])
        else:
            # Якщо взагалі нема даних
            rmse_fe = rmse_mass = 0.0
            r2_fe = r2_mass = 0.0
            bias_fe = bias_mass = 0.0
    else:
        # Основний шлях: є дані передбачень
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Обрізаємо до однакової довжини
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Розрахунки для Fe (колонка 0)
        rmse_fe = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        r2_fe = r2_score(y_true[:, 0], y_pred[:, 0])
        bias_fe = np.mean(y_pred[:, 0] - y_true[:, 0])
        
        # Розрахунки для Mass (колонка 1)
        rmse_mass = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
        r2_mass = r2_score(y_true[:, 1], y_pred[:, 1])
        bias_mass = np.mean(y_pred[:, 1] - y_true[:, 1])
    
    return {
        'model_rmse_fe': rmse_fe,
        'model_rmse_mass': rmse_mass,
        'model_r2_fe': r2_fe,
        'model_r2_mass': r2_mass,
        'model_bias_fe': bias_fe,
        'model_bias_mass': bias_mass
    }

# =============================================================================
# === ФУНКЦІЇ ОЦІНЮВАННЯ КЕРУВАННЯ ===
# =============================================================================

def evaluate_control_performance(results_df: pd.DataFrame, params: Dict) -> Dict[str, float]:
    """Оцінює якість роботи системи керування"""
    
    # Уставки
    ref_fe = params.get('ref_fe', 53.5)
    ref_mass = params.get('ref_mass', 57.0)
    
    # ✅ НАСТРОЮВАНІ ТОЛЕРАНТНОСТІ
    tolerance_fe_percent = params.get('tolerance_fe_percent', 2.0)    
    tolerance_mass_percent = params.get('tolerance_mass_percent', 2.0) 
    
    # Фактичні значення
    fe_values = results_df['conc_fe'].values
    mass_values = results_df['conc_mass'].values
    control_values = results_df['solid_feed_percent'].values
    
    # Помилки відстеження
    error_fe = fe_values - ref_fe
    error_mass = mass_values - ref_mass
    
    # 1. Помилки відстеження (RMSE від уставки) - ✅ ЗАЛИШАЄТЬСЯ
    tracking_error_fe = np.sqrt(np.mean(error_fe**2))
    tracking_error_mass = np.sqrt(np.mean(error_mass**2))
    
    # 2. ISE (Integral of Squared Error) - ✅ ЗАЛИШАЄТЬСЯ
    ise_fe = np.sum(error_fe**2)
    ise_mass = np.sum(error_mass**2)
    
    # 3. IAE (Integral of Absolute Error) - ✅ ЗАЛИШАЄТЬСЯ
    iae_fe = np.sum(np.abs(error_fe))
    iae_mass = np.sum(np.abs(error_mass))
    
    # 4. Згладженість керування - ✅ ЗАЛИШАЄТЬСЯ
    control_changes = np.diff(control_values)
    control_smoothness = 1 / (1 + np.std(control_changes))
    
    # 5. Досягнення уставок з настроюваними толерантностями
    tolerance_fe = (tolerance_fe_percent / 100.0) * abs(ref_fe)
    tolerance_mass = (tolerance_mass_percent / 100.0) * abs(ref_mass)
    
    # ✅ ВИДАЛИТИ ТІЛЬКИ ЦІ 4 ЗАЙВІ PRINT-И (занадто багато тексту):
    # print(f"\n🎯 Розрахунок досягнення уставок:")
    # print(f"   ⚙️ Налаштування толерантності:")
    # print(f"      Fe: {tolerance_fe_percent}% від уставки = ±{tolerance_fe:.3f}")
    # print(f"      Mass: {tolerance_mass_percent}% від уставки = ±{tolerance_mass:.3f}")
    
    setpoint_achievement_fe = calculate_setpoint_achievement(fe_values, ref_fe, tolerance_fe)
    setpoint_achievement_mass = calculate_setpoint_achievement(mass_values, ref_mass, tolerance_mass)
    
    delta_u_max = params.get('delta_u_max', 1.0)
    aggressiveness_metrics = calculate_control_aggressiveness_metrics(control_values, delta_u_max)
    
    return {
        'tracking_error_fe': tracking_error_fe,
        'tracking_error_mass': tracking_error_mass,
        'ise_fe': ise_fe,
        'ise_mass': ise_mass,
        'iae_fe': iae_fe,
        'iae_mass': iae_mass,
        'control_smoothness': control_smoothness,
        'setpoint_achievement_fe': setpoint_achievement_fe,
        'setpoint_achievement_mass': setpoint_achievement_mass,
        **aggressiveness_metrics  # ✅ Додаємо всі метрики агресивності
    }

def calculate_control_aggressiveness_metrics(control_values: np.ndarray, 
                                           delta_u_max: float) -> Dict[str, float]:
    """
    Розраховує метрики агресивності керування на основі вашого прикладу
    """
    
    # Обчислюємо зміни керування
    delta_u = np.diff(control_values)
    abs_delta_u = np.abs(delta_u)
    nonzero_delta_u = abs_delta_u[abs_delta_u > 1e-8]
    
    # 1. Основні статистики
    mean_delta_u = np.mean(abs_delta_u) if len(abs_delta_u) > 0 else 0.0
    std_delta_u = np.std(delta_u) if len(delta_u) > 0 else 0.0
    max_abs_delta_u = np.max(abs_delta_u) if len(abs_delta_u) > 0 else 0.0
    mean_abs_nonzero_delta_u = np.mean(nonzero_delta_u) if len(nonzero_delta_u) > 0 else 0.0
    
    # 2. Енергія керування
    energy_u = np.sum(delta_u**2)
    
    # 3. Значні зміни (> 50% від max)
    significant_threshold = 0.5 * delta_u_max
    significant_magnitude_changes_count = np.sum(abs_delta_u > significant_threshold)
    significant_magnitude_changes_frequency = significant_magnitude_changes_count / len(delta_u) if len(delta_u) > 0 else 0.0
    
    # 4. Зміни напрямку
    directional_switch_count = 0
    for i in range(1, len(delta_u)):
        if np.sign(delta_u[i]) != np.sign(delta_u[i-1]) and abs(delta_u[i]) > 1e-8 and abs(delta_u[i-1]) > 1e-8:
            directional_switch_count += 1
    
    directional_switch_frequency = directional_switch_count / len(delta_u) if len(delta_u) > 0 else 0.0
    
    # 5. Використання максимуму
    percentage_of_max_delta_u_used = (mean_delta_u / delta_u_max * 100) if delta_u_max > 0 else 0.0
    num_steps_at_delta_u_max = np.sum(abs_delta_u >= 0.95 * delta_u_max)
    
    return {
        'control_aggressiveness': mean_abs_nonzero_delta_u,  # Використовуємо nonzero як у вас
        'control_variability': std_delta_u,
        'control_energy': energy_u,
        'control_stability_index': 1.0 - directional_switch_frequency,
        'control_utilization': percentage_of_max_delta_u_used,
        'significant_changes_frequency': significant_magnitude_changes_frequency,
        'significant_changes_count': float(significant_magnitude_changes_count),
        'max_control_change': max_abs_delta_u,
        'directional_switches_per_step': directional_switch_frequency,
        'directional_switches_count': float(directional_switch_count),
        'steps_at_max_delta_u': float(num_steps_at_delta_u_max)
    }

def calculate_setpoint_achievement(values: np.ndarray, setpoint: float, tolerance: float) -> float:
    """Розраховує відсоток часу, коли значення в межах толерантності від уставки"""
    within_tolerance = np.abs(values - setpoint) <= tolerance
    achievement_pct = np.mean(within_tolerance) * 100.0
    
    # ✅ ДОДАЄМО ДІАГНОСТИКУ
    # print(f"   🔍 Діагностика уставки:")
    # print(f"      Уставка: {setpoint:.2f}")
    # print(f"      Толерантність: ±{tolerance:.2f}")
    # print(f"      Діапазон допуску: [{setpoint-tolerance:.2f}, {setpoint+tolerance:.2f}]")
    # print(f"      Фактичний діапазон: [{np.min(values):.2f}, {np.max(values):.2f}]")
    # print(f"      Точок в допуску: {np.sum(within_tolerance)}/{len(values)}")
    print(f"      Уставка {setpoint:.1f} ±{tolerance:.2f}: {np.sum(within_tolerance)}/{len(values)} точок ({achievement_pct:.1f}%)")

    return achievement_pct

# =============================================================================
# === ФУНКЦІЇ ЗАГАЛЬНОГО ОЦІНЮВАННЯ ===
# =============================================================================

def calculate_overall_metrics(results_df: pd.DataFrame, params: Dict, 
                            model_metrics: Dict, control_metrics: Dict) -> Dict[str, float]:
    """Розраховує загальні метрики ефективності"""
    
    # 1. Стабільність процесу (обернена до коефіцієнта варіації)
    fe_values = results_df['conc_fe'].values
    mass_values = results_df['conc_mass'].values
    
    fe_cv = np.std(fe_values) / (np.mean(fe_values) + 1e-8)
    mass_cv = np.std(mass_values) / (np.mean(mass_values) + 1e-8)
    process_stability = 1 / (1 + (fe_cv + mass_cv) / 2)
    
    # 2. Загальний score (зважена комбінація метрик)
    # Нормалізуємо метрики до [0, 1]
    
    # Модель: R² вже в [0, 1], більше = краще
    model_score = (max(0, model_metrics['model_r2_fe']) + 
                   max(0, model_metrics['model_r2_mass'])) / 2
    
    # Керування: досягнення уставок в [0, 100], конвертуємо до [0, 1]
    control_score = (control_metrics['setpoint_achievement_fe'] + 
                     control_metrics['setpoint_achievement_mass']) / 200
    
    # Згладженість керування вже нормалізована
    smoothness_score = min(1.0, control_metrics['control_smoothness'])
    
    # Зважена комбінація
    overall_score = (0.4 * model_score +      # 40% - якість моделі
                     0.4 * control_score +    # 40% - досягнення уставок  
                     0.2 * smoothness_score   # 20% - згладженість
                    ) * 100  # Конвертуємо до [0, 100]
    
    return {
        'overall_score': overall_score,
        'process_stability': process_stability
    }

# =============================================================================
# === ГОЛОВНА ФУНКЦІЯ ОЦІНЮВАННЯ ===
# =============================================================================

def evaluate_simulation(results_df: pd.DataFrame, analysis_data: Dict, 
                       params: Dict) -> EvaluationResults:
    """
    Головна функція оцінювання ефективності симуляції
    
    Args:
        results_df: DataFrame з результатами симуляції
        analysis_data: Словник з додатковими даними аналізу
        params: Параметри симуляції
        
    Returns:
        EvaluationResults з усіма метриками
    """
    
    # Оцінка моделей
    model_metrics = evaluate_model_performance(results_df, analysis_data)
    
    # Оцінка керування
    control_metrics = evaluate_control_performance(results_df, params)
    
    # Загальні метрики
    overall_metrics = calculate_overall_metrics(results_df, params, 
                                               model_metrics, control_metrics)
    
    # Збираємо все разом
    return EvaluationResults(
        **model_metrics,
        **control_metrics,
        **overall_metrics
    )

# =============================================================================
# === ФУНКЦІЇ ВИВОДУ ТА ЗВІТНОСТІ ===
# =============================================================================

def print_evaluation_report(eval_results: EvaluationResults, detailed: bool = True):
    """
    Виводить звіт про оцінювання ефективності
    
    Args:
        eval_results: Результати оцінювання
        detailed: Чи виводити детальний звіт
    """
    
    print("🎯 ОЦІНКА ЕФЕКТИВНОСТІ MPC СИМУЛЯЦІЇ")
    print("=" * 50)
    
    # Загальна оцінка (завжди виводимо)
    print(f"⭐ ЗАГАЛЬНА ОЦІНКА: {eval_results.overall_score:.1f}/100")
    print(f"🔒 СТАБІЛЬНІСТЬ ПРОЦЕСУ: {eval_results.process_stability:.3f}")
    
    # ✅ ДОДАЄМО КЛАСИФІКАЦІЮ
    classification = get_mpc_quality_classification(eval_results.overall_score)
    print(f"📊 КЛАСИФІКАЦІЯ: {classification}")
    
    if detailed:
        print(f"\n📊 ЯКІСТЬ МОДЕЛЕЙ:")
        print(f"   • Fe RMSE: {eval_results.model_rmse_fe:.3f} (R² = {eval_results.model_r2_fe:.3f})")
        print(f"   • Mass RMSE: {eval_results.model_rmse_mass:.3f} (R² = {eval_results.model_r2_mass:.3f})")
        print(f"   • Систематична помилка Fe: {eval_results.model_bias_fe:+.3f}")
        print(f"   • Систематична помилка Mass: {eval_results.model_bias_mass:+.3f}")
        
        print(f"\n🎮 ЯКІСТЬ КЕРУВАННЯ:")
        print(f"   • Помилка відстеження Fe: {eval_results.tracking_error_fe:.3f}")
        print(f"   • Помилка відстеження Mass: {eval_results.tracking_error_mass:.3f}")
        print(f"   • ISE Fe: {eval_results.ise_fe:.1f}")
        print(f"   • ISE Mass: {eval_results.ise_mass:.1f}")
        print(f"   • IAE Fe: {eval_results.iae_fe:.1f}")
        print(f"   • IAE Mass: {eval_results.iae_mass:.1f}")
        print(f"   • Згладженість керування: {eval_results.control_smoothness:.3f}")
        print(f"   • Досягнення уставки Fe: {eval_results.setpoint_achievement_fe:.1f}%")
        print(f"   • Досягнення уставки Mass: {eval_results.setpoint_achievement_mass:.1f}%")

        print(f"\n🎛️ АГРЕСИВНІСТЬ КЕРУВАННЯ:")
        print(f"   • Середня зміна: {eval_results.control_aggressiveness:.3f}")
        print(f"   • Варіативність: {eval_results.control_variability:.3f}")
        print(f"   • Енергія керування: {eval_results.control_energy:.1f}")
        print(f"   • Індекс стабільності: {eval_results.control_stability_index:.3f}")
        print(f"   • Використання діапазону: {eval_results.control_utilization:.1f}%")
        print(f"   • Значні зміни: {eval_results.significant_changes_count:.0f} ({eval_results.significant_changes_frequency:.1%})")
        print(f"   • Зміни напрямку: {eval_results.directional_switches_count:.0f} ({eval_results.directional_switches_per_step:.1%})")
        print(f"   • Максимальна зміна: {eval_results.max_control_change:.3f}")
        print(f"   • Кроків на максимумі: {eval_results.steps_at_max_delta_u:.0f}")
        
        # ✅ ДОДАЄМО РЕКОМЕНДАЦІЇ
        recommendations = generate_recommendations(eval_results)
        if recommendations:
            print(f"\n💡 РЕКОМЕНДАЦІЇ:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

def get_mpc_quality_classification(score: float) -> str:
    """Класифікує якість MPC системи"""
    if score >= 80:
        return "Відмінно"
    elif score >= 65:
        return "Добре" 
    elif score >= 50:
        return "Задовільно"
    elif score >= 35:
        return "Потребує покращення"
    else:
        return "Незадовільно"

def generate_recommendations(eval_results: EvaluationResults) -> List[str]:
    """Генерує автоматичні рекомендації для покращення системи"""
    recommendations = []
    
    # Перевіряємо точність відстеження Mass
    if eval_results.tracking_error_mass > 2.0:
        recommendations.append("Покращити точність відслідковування Mass (помилка > 2.0)")
    
    # Перевіряємо досягнення уставок
    if eval_results.setpoint_achievement_fe < 70:
        recommendations.append("Покращити відстеження уставки Fe (< 70% в допуску)")
        
    if eval_results.setpoint_achievement_mass < 70:
        recommendations.append("Покращити відстеження уставки Mass (< 70% в допуску)")
    
    # Перевіряємо якість моделі
    if eval_results.model_r2_fe < 0.8:
        recommendations.append("Покращити якість моделі для Fe (R² < 0.8)")
        
    if eval_results.model_r2_mass < 0.8:
        recommendations.append("Покращити якість моделі для Mass (R² < 0.8)")
    
    # Перевіряємо згладженість керування
    if eval_results.control_smoothness < 0.5:
        recommendations.append("Зменшити коливання керуючого сигналу")
    
    # Позитивні відгуки
    if eval_results.control_smoothness > 0.8:
        recommendations.append("✅ Стабільне керування - добре налаштовано!")
        
    if eval_results.process_stability > 0.9:
        recommendations.append("✅ Висока стабільність процесу!")
        
    if eval_results.overall_score > 80:
        recommendations.append("✅ Відмінна загальна продуктивність!")
    
    if eval_results.control_stability_index < 0.6:
        recommendations.append("🔄 Занадто часті зміни напрямку - збільшити λ_obj")
        
    if eval_results.control_aggressiveness > 1.0:
        recommendations.append("⚡ Зменшити агресивність керування")
        
    if eval_results.control_utilization > 80:
        recommendations.append("📊 Контролер працює на межі - збільшити delta_u_max")
        
    if eval_results.significant_changes_frequency > 0.3:
        recommendations.append("📈 Занадто багато різких змін - розглянути фільтрацію")
    
    # Позитивні відгуки
    if eval_results.control_stability_index > 0.8:
        recommendations.append("✅ Стабільне керування без коливань!")
        
    return recommendations

def get_performance_summary(eval_results: EvaluationResults) -> str:
    """Повертає короткий текстовий опис ефективності"""
    score = eval_results.overall_score
    
    if score >= 90:
        return "🌟 Відмінно"
    elif score >= 80:
        return "✅ Добре"
    elif score >= 70:
        return "📈 Задовільно"
    elif score >= 60:
        return "⚠️ Потребує покращення"
    else:
        return "❌ Незадовільно"

# =============================================================================
# === ФУНКЦІЇ ПОРІВНЯННЯ ===
# =============================================================================

def compare_evaluations(evaluations: Dict[str, EvaluationResults], 
                       show_details: bool = True) -> None:
    """
    Порівнює результати кількох симуляцій
    
    Args:
        evaluations: Словник {назва_конфігурації: EvaluationResults}
        show_details: Чи показувати детальні метрики
    """
    
    print("\n🔍 ПОРІВНЯННЯ КОНФІГУРАЦІЙ")
    print("=" * 60)
    
    # Заголовок таблиці
    configs = list(evaluations.keys())
    print(f"{'Метрика':<25}", end="")
    for config in configs:
        print(f"{config:>15}", end="")
    print()
    print("-" * (25 + 15 * len(configs)))
    
    # Загальна оцінка
    print(f"{'Загальна оцінка':<25}", end="")
    for config in configs:
        score = evaluations[config].overall_score
        print(f"{score:>13.1f}/100", end="")
    print()
    
    if show_details:
        # Ключові метрики
        metrics_to_show = [
            ('Model R² Fe', 'model_r2_fe', '.3f'),
            ('Model R² Mass', 'model_r2_mass', '.3f'),
            ('ISE Fe', 'ise_fe', '.1f'),
            ('ISE Mass', 'ise_mass', '.1f'),
            ('IAE Fe', 'iae_fe', '.1f'),
            ('IAE Mass', 'iae_mass', '.1f'),
            ('Tracking Fe', 'setpoint_achievement_fe', '.1f'),
            ('Tracking Mass', 'setpoint_achievement_mass', '.1f'),
            ('Стабільність', 'process_stability', '.3f')
        ]
        
        for metric_name, attr_name, fmt in metrics_to_show:
            print(f"{metric_name:<25}", end="")
            for config in configs:
                value = getattr(evaluations[config], attr_name)
                if 'achievement' in attr_name:
                    print(f"{value:>{13}{fmt}}%", end="")
                else:
                    print(f"{value:>{15}{fmt}}", end="")
            print()
    
    # Рекомендація
    best_config = max(evaluations.keys(), 
                     key=lambda k: evaluations[k].overall_score)
    best_score = evaluations[best_config].overall_score
    
    print(f"\n💡 Рекомендація: '{best_config}' (оцінка: {best_score:.1f})")

# =============================================================================
# === ФУНКЦІЇ ВІЗУАЛІЗАЦІЇ ===
# =============================================================================

def create_evaluation_plots(results_df: pd.DataFrame, eval_results: EvaluationResults, 
                           params: Dict, save_path: Optional[str] = None):
    """
    Створює графіки для візуального аналізу ефективності
    
    Args:
        results_df: DataFrame з результатами симуляції
        eval_results: Результати оцінювання
        params: Параметри симуляції  
        save_path: Шлях для збереження (опціонально)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Оцінка ефективності MPC симуляції', fontsize=16, fontweight='bold')
    
    # 1. Відстеження уставок
    ax1 = axes[0, 0]
    time_steps = np.arange(len(results_df))
    
    ax1.plot(time_steps, results_df['conc_fe'], 'b-', label='Fe фактичне', alpha=0.8)
    ax1.axhline(y=params.get('ref_fe', 53.5), color='b', linestyle='--', 
                label=f"Fe уставка ({params.get('ref_fe', 53.5)})")
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_steps, results_df['conc_mass'], 'r-', label='Mass фактичне', alpha=0.8)
    ax1_twin.axhline(y=params.get('ref_mass', 57.0), color='r', linestyle='--',
                     label=f"Mass уставка ({params.get('ref_mass', 57.0)})")
    
    ax1.set_xlabel('Крок симуляції')
    ax1.set_ylabel('Fe концентрація, %', color='b')
    ax1_twin.set_ylabel('Mass потік, т/г', color='r')
    ax1.set_title('Відстеження уставок')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Керуючий сигнал
    ax2 = axes[0, 1]
    ax2.plot(time_steps, results_df['solid_feed_percent'], 'g-', linewidth=1.5)
    ax2.set_xlabel('Крок симуляції')
    ax2.set_ylabel('Solid feed, %')
    ax2.set_title(f'Згладженість керування: {eval_results.control_smoothness:.3f}')
    ax2.grid(True, alpha=0.3)
    
    # 3. Розподіл помилок відстеження
    ax3 = axes[1, 0]
    fe_errors = results_df['conc_fe'] - params.get('ref_fe', 53.5)
    mass_errors = results_df['conc_mass'] - params.get('ref_mass', 57.0)
    
    ax3.hist(fe_errors, bins=20, alpha=0.7, label='Fe помилки', color='blue')
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Помилка відстеження')
    ax3.set_ylabel('Частота')
    ax3.set_title('Розподіл помилок Fe')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Підсумкові метрики
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Текстовий звіт без емодзі для matplotlib
    summary_text = f"""
ПІДСУМОК ОЦІНКИ

Загальна оцінка: {eval_results.overall_score:.1f}/100
Статус: {get_performance_summary(eval_results).replace('🌟', '').replace('✅', '').replace('📈', '').replace('⚠️', '').replace('❌', '').strip()}

Модель:
  R² Fe: {eval_results.model_r2_fe:.3f}
  R² Mass: {eval_results.model_r2_mass:.3f}

Керування:
  Досягнення Fe: {eval_results.setpoint_achievement_fe:.1f}%
  Досягнення Mass: {eval_results.setpoint_achievement_mass:.1f}%
  
Стабільність: {eval_results.process_stability:.3f}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='sans-serif',  # Змінено з 'monospace'
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Графіки збережено: {save_path}")
    
    plt.show()

# =============================================================================
# === ДОПОМІЖНІ ФУНКЦІЇ ===
# =============================================================================

def validate_evaluation_data(results_df: pd.DataFrame, analysis_data: Dict, 
                            params: Dict) -> bool:
    """Перевіряє чи є всі необхідні дані для оцінювання"""
    
    required_columns = ['conc_fe', 'conc_mass', 'solid_feed_percent']
    missing_columns = [col for col in required_columns if col not in results_df.columns]
    
    if missing_columns:
        print(f"❌ Відсутні колонки в results_df: {missing_columns}")
        return False
    
    if len(results_df) == 0:
        print("❌ results_df порожній")
        return False
    
    required_params = ['ref_fe', 'ref_mass']
    missing_params = [param for param in required_params if param not in params]
    
    if missing_params:
        print(f"⚠️ Відсутні параметри (використовуємо значення за замовчуванням): {missing_params}")
    
    return True

if __name__ == "__main__":
    # Приклад використання (для тестування)
    print("evaluation_simple.py - модуль готовий до використання!")
    print("Використання:")
    print("  from evaluation_simple import evaluate_simulation, print_evaluation_report")
    print("  eval_results = evaluate_simulation(results_df, analysis_data, params)")
    print("  print_evaluation_report(eval_results)")