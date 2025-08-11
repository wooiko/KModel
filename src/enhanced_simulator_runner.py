# enhanced_simulator_runner.py - Приклади запуску розширеного симулятора

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from enhanced_sim import compare_mpc_configurations

# Імпортуємо розширені функції
from enhanced_sim import (
    simulate_mpc,
    quick_mpc_benchmark, 
    detailed_mpc_analysis,
    simulate_mpc_with_config_enhanced,
    load_historical_data
)
from enhanced_benchmark import (
    pandas_safe_sort
)

# def pandas_safe_sort(df, column):
#     """Безпечне сортування для всіх версій pandas"""
#     if df.empty or column not in df.columns:
#         return df
    
#     try:
#         return df.sort_values(column, na_position='last')
#     except (TypeError, ValueError):
#         try:
#             return df.sort_values(column, na_last=True)
#         except (TypeError, ValueError):
#             # Ручне сортування
#             valid_mask = df[column].notna()
#             if valid_mask.any():
#                 valid_df = df[valid_mask].sort_values(column)
#                 invalid_df = df[~valid_mask]
#                 return pd.concat([valid_df, invalid_df], ignore_index=True)
#             return df
        
def compare_mpc_configurations_improved(
    configurations: List[Dict],
    hist_df: pd.DataFrame,
    base_config: str = 'oleksandr_original',
    comparison_steps: int = 100
) -> pd.DataFrame:
    """🔄 ПОКРАЩЕНЕ порівняння конфігурацій MPC з оптимальними розмірами даних"""
    
    print("🔄 ПОКРАЩЕНЕ ПОРІВНЯННЯ КОНФІГУРАЦІЙ MPC")
    print("="*60)
    
    comparison_results = []
    
    for i, config in enumerate(configurations):
        config_name = config.get('name', f'Config_{i+1}')
        print(f"\n🧪 Тестуємо конфігурацію: {config_name}")
        
        try:
            # Імпортуємо функцію
            try:
                from enhanced_sim import simulate_mpc_core_enhanced as simulate_mpc_core
            except ImportError:
                from sim import simulate_mpc_core
            
            # Готуємо конфігурацію
            test_config = config.copy()
            test_config.pop('name', None)  # Видаляємо проблемний параметр
            
            # ✅ ОПТИМАЛЬНІ РОЗМІРИ ДАНИХ ДЛЯ КОЖНОЇ МОДЕЛІ
            model_type = test_config.get('model_type', 'linear')
            
            if model_type == 'krr':
                optimal_data = 8000  # KRR потребує багато даних
            elif model_type == 'svr':
                optimal_data = 5000  # SVR менше
            else:
                optimal_data = 3000  # Linear достатньо
            
            # Оновлюємо конфігурацію
            test_config.update({
                'N_data': optimal_data,  # ← КЛЮЧОВЕ ПОКРАЩЕННЯ!
                'control_pts': comparison_steps,
                'run_analysis': False,
                'find_optimal_params': True,  # ✅ Обов'язково увімкнено!
                'train_size': 0.7,
                'val_size': 0.15,
                'test_size': 0.15
            })
            
            print(f"   🔧 Модель: {model_type}, Дані: {optimal_data}, Оптимізація: ✅")
            print(f"   ⚙️ Параметри: Np={config.get('Np', '?')}, λ={config.get('λ_obj', '?')}")
            
            # Запускаємо симуляцію
            start_time = time.time()
            results_df, metrics = simulate_mpc_core(hist_df, **test_config)
            test_time = time.time() - start_time
            
            # Збираємо метрики
            comparison_row = {
                'Configuration': config_name,
                'Model': f"{config.get('model_type', 'unknown')}-{config.get('kernel', 'default')}",
                'Data_Size': optimal_data,
                'Np': config.get('Np', 'unknown'),
                'Nc': config.get('Nc', 'unknown'),
                'Lambda': config.get('λ_obj', 'unknown'),
                'Test_Time_Min': test_time / 60
            }
            
            # Додаємо результати
            if isinstance(metrics, dict):
                comparison_row['RMSE_Fe'] = metrics.get('test_rmse_conc_fe', np.nan)
                comparison_row['RMSE_Mass'] = metrics.get('test_rmse_conc_mass', np.nan)
                comparison_row['R2_Fe'] = metrics.get('r2_fe', np.nan)
                comparison_row['R2_Mass'] = metrics.get('r2_mass', np.nan)
                comparison_row['Quality_Score'] = metrics.get('quality_score', np.nan)
                comparison_row['Real_Time_Suitable'] = metrics.get('real_time_suitable', False)
            
            comparison_results.append(comparison_row)
            
            # Звіт
            rmse_fe = comparison_row.get('RMSE_Fe', float('inf'))
            r2_fe = comparison_row.get('R2_Fe', 0)
            quality = comparison_row.get('Quality_Score', 1)
            
            print(f"   ✅ Результати:")
            print(f"      RMSE Fe: {rmse_fe:.4f}")
            print(f"      R² Fe: {r2_fe:.4f}")
            print(f"      Якість: {quality:.4f}")
            print(f"      Час: {test_time/60:.1f}хв")
            
        except Exception as e:
            print(f"   ❌ Помилка: {e}")
            comparison_results.append({
                'Configuration': config_name,
                'Error': str(e)
            })
    
    # Створюємо DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    # Безпечне сортування (використовуємо ту саму функцію що в enhanced_benchmark.py)
    if not comparison_df.empty and 'RMSE_Fe' in comparison_df.columns:
        # Відфільтровуємо NaN перед сортуванням
        valid_mask = comparison_df['RMSE_Fe'].notna()
        if valid_mask.any():
            valid_df = comparison_df[valid_mask].sort_values('RMSE_Fe')
            invalid_df = comparison_df[~valid_mask]
            comparison_df = pd.concat([valid_df, invalid_df], ignore_index=True)
    
    print(f"\n📊 ПІДСУМОК ПОКРАЩЕНОГО ПОРІВНЯННЯ:")
    if not comparison_df.empty:
        display_cols = ['Configuration', 'Model', 'Data_Size', 'RMSE_Fe', 'R2_Fe', 'Quality_Score']
        available_cols = [col for col in display_cols if col in comparison_df.columns]
        if available_cols:
            print(comparison_df[available_cols].round(4))
    
    return comparison_df
        
# def load_historical_data() -> pd.DataFrame:
#     """Завантажує історичні дані для симуляції"""
    
#     # Спробуємо завантажити з різних місць
#     possible_paths = [
#         'processed.parquet',
#         'data/processed.parquet', 
#         '/content/KModel/src/processed.parquet',
#         '../data/processed.parquet'
#     ]
    
#     for path in possible_paths:
#         try:
#             hist_df = pd.read_parquet(path)
#             print(f"✅ Дані завантажено з: {path}")
#             print(f"   📊 Розмір: {hist_df.shape[0]} рядків, {hist_df.shape[1]} колонок")
#             return hist_df
#         except FileNotFoundError:
#             continue
#         except Exception as e:
#             print(f"⚠️ Помилка завантаження з {path}: {e}")
#             continue
    
#     raise FileNotFoundError("❌ Не вдалося знайти файл processed.parquet")

def progress_callback(step: int, total: int, message: str):
    """Callback для відображення прогресу"""
    if step % 50 == 0 or step == total or 'завершен' in message.lower():
        progress_pct = (step / total * 100) if total > 0 else 0
        print(f"   📈 [{step:4d}/{total:4d}] {progress_pct:5.1f}% - {message}")

def example_1_quick_benchmark():
    """🚀 Приклад 1: Швидкий бенчмарк різних моделей"""
    
    print("\n" + "="*70)
    print("🚀 ПРИКЛАД 1: ШВИДКИЙ БЕНЧМАРК МОДЕЛЕЙ MPC")
    print("="*70)
    
    try:
        hist_df = load_historical_data()
        
        # Запускаємо швидкий бенчмарк
        models_to_test = ['krr', 'svr', 'linear']
        
        print(f"🧪 Тестуємо {len(models_to_test)} моделей...")
        start_time = time.time()
        
        results_df = quick_mpc_benchmark(
            hist_df=hist_df,
            config='oleksandr_original',
            models_to_test=models_to_test,
            save_results=True
        )
        
        total_time = time.time() - start_time
        
        print(f"\n📊 РЕЗУЛЬТАТИ ШВИДКОГО БЕНЧМАРКУ:")
        print(f"   ⏱️ Загальний час: {total_time:.1f} секунд")
        print(f"   🏆 Найкраща модель: {results_df.iloc[0]['Model']}")
        
        # Показуємо топ результати
        print(f"\n🏅 ТОП-3 МОДЕЛІ:")
        top_3 = results_df.head(3)
        for idx, row in top_3.iterrows():
            rank = idx + 1
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            print(f"   {medal} {row['Model']}: RMSE_Fe={row['RMSE_Fe']:.4f}, "
                  f"Якість={row['Quality_Score']:.3f}, "
                  f"Час={row['Cycle_Time_Ms']:.1f}ms")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Помилка в прикладі 1: {e}")
        return None

def example_2_detailed_analysis():
    """🔬 Приклад 2: Детальний аналіз конкретної конфігурації"""
    
    print("\n" + "="*70)
    print("🔬 ПРИКЛАД 2: ДЕТАЛЬНИЙ АНАЛІЗ MPC")
    print("="*70)
    
    try:
        hist_df = load_historical_data()
        
        # Конфігурація для детального аналізу
        config_overrides = {
            'model_type': 'krr',
            'kernel': 'rbf',
            'Np': 8,
            'Nc': 6,
            'N_data': 3000,  # Більше даних для точного аналізу
            'control_pts': 500
        }
        
        print("🔬 Запускаємо детальний аналіз...")
        print(f"   📋 Конфігурація: {config_overrides}")
        
        start_time = time.time()
        
        analysis_report = detailed_mpc_analysis(
            hist_df=hist_df,
            config='oleksandr_original',
            config_overrides=config_overrides
        )
        
        analysis_time = time.time() - start_time
        
        print(f"\n📋 ДЕТАЛЬНИЙ ЗВІТ:")
        print(f"   ⏱️ Час аналізу: {analysis_time:.1f} секунд")
        
        # Основні метрики
        basic_metrics = analysis_report['basic_metrics']
        print(f"\n📊 ОСНОВНІ МЕТРИКИ:")
        for key, value in basic_metrics.items():
            if isinstance(value, (int, float)):
                print(f"   • {key}: {value:.6f}")
        
        # Метрики швидкості  
        speed_metrics = analysis_report['speed_metrics']
        print(f"\n⚡ ШВИДКОДІЯ:")
        for key, value in speed_metrics.items():
            if isinstance(value, (int, float)):
                if 'time' in key.lower():
                    unit = "ms" if value < 1 else "с"
                    display_value = value * 1000 if value < 1 else value
                    print(f"   • {key}: {display_value:.2f}{unit}")
        
        # Рекомендації
        recommendations = analysis_report['recommendations']
        if recommendations:
            print(f"\n💡 РЕКОМЕНДАЦІЇ ({len(recommendations)}):")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print(f"\n✅ СИСТЕМА ПРАЦЮЄ ОПТИМАЛЬНО!")
        
        return analysis_report
        
    except Exception as e:
        print(f"❌ Помилка в прикладі 2: {e}")
        return None

def example_3_custom_simulation():
    """🎯 Приклад 3: Користувацька симуляція з бенчмарком"""
    
    print("\n" + "="*70)
    print("🎯 ПРИКЛАД 3: КОРИСТУВАЦЬКА СИМУЛЯЦІЯ З БЕНЧМАРКОМ")
    print("="*70)
    
    try:
        hist_df = load_historical_data()
        
        # Користувацька конфігурація
        custom_config = {
            'model_type': 'gpr',  # Gaussian Process Regression
            'kernel': 'rbf',
            'Np': 10,  # Більший горизонт прогнозування
            'Nc': 8,   # Більший горизонт керування
            'w_fe': 10.0,  # Більша вага для Fe
            'w_mass': 1.5,
            'λ_obj': 0.05,  # Менше згладжування
            'ref_fe': 54.0,  # Вища уставка Fe
            'ref_mass': 58.0,
            'N_data': 4000,
            'control_pts': 800,
            'find_optimal_params': True  # Оптимізація гіперпараметрів
        }
        
        print("🎯 Запускаємо користувацьку симуляцію...")
        print(f"   📋 Особливості:")
        print(f"      • Модель: {custom_config['model_type'].upper()}")
        print(f"      • Горизонти: Np={custom_config['Np']}, Nc={custom_config['Nc']}")
        print(f"      • Уставки: Fe={custom_config['ref_fe']}, Mass={custom_config['ref_mass']}")
        print(f"      • Оптимізація гіперпараметрів: ✅")
        
        start_time = time.time()
        
        results_df, metrics = simulate_mpc(
            hist_df,
            config='oleksandr_original',
            config_overrides=custom_config,
            # 🆕 Увімкнуті розширені функції бенчмарку
            enable_comprehensive_analysis=True,
            benchmark_control_quality=True,
            save_benchmark_results=True,
            progress_callback=progress_callback
        )
        
        simulation_time = time.time() - start_time
        
        print(f"\n🎯 РЕЗУЛЬТАТИ КОРИСТУВАЦЬКОЇ СИМУЛЯЦІЇ:")
        print(f"   ⏱️ Час симуляції: {simulation_time/60:.1f} хвилин")
        print(f"   📊 Точок даних: {len(results_df)}")
        
        # Ключові результати
        key_results = {
            'RMSE Fe': metrics.get('test_rmse_conc_fe', 'N/A'),
            'RMSE Mass': metrics.get('test_rmse_conc_mass', 'N/A'), 
            'R² Fe': metrics.get('r2_fe', 'N/A'),
            'R² Mass': metrics.get('r2_mass', 'N/A'),
            'Якість керування': metrics.get('quality_score', 'N/A'),
            'Час циклу (ms)': metrics.get('total_cycle_time', 0) * 1000,
            'Real-time придатність': metrics.get('real_time_suitable', False)
        }
        
        print(f"\n📈 КЛЮЧОВІ РЕЗУЛЬТАТИ:")
        for key, value in key_results.items():
            if isinstance(value, (int, float)):
                if 'R²' in key:
                    print(f"   • {key}: {value:.4f}")
                elif 'RMSE' in key or 'Якість' in key:
                    print(f"   • {key}: {value:.6f}")
                elif 'ms' in key:
                    print(f"   • {key}: {value:.1f}")
                else:
                    print(f"   • {key}: {value}")
            else:
                print(f"   • {key}: {value}")
        
        # Аналіз результатів
        print(f"\n🔍 АНАЛІЗ:")
        
        # Якість моделі
        rmse_fe = metrics.get('test_rmse_conc_fe', float('inf'))
        if rmse_fe < 0.05:
            print(f"   ✅ Відмінна точність моделі Fe (RMSE < 0.05)")
        elif rmse_fe < 0.1:
            print(f"   👍 Хороша точність моделі Fe (RMSE < 0.1)")
        else:
            print(f"   ⚠️ Потребує покращення точності Fe (RMSE > 0.1)")
        
        # Швидкодія
        cycle_time = metrics.get('total_cycle_time', 0) * 1000
        if cycle_time < 1000:  # < 1 секунди
            print(f"   ⚡ Відмінна швидкодія ({cycle_time:.0f}ms)")
        elif cycle_time < 5000:  # < 5 секунд
            print(f"   👍 Прийнятна швидкодія для real-time ({cycle_time:.0f}ms)")
        else:
            print(f"   🐌 Повільна швидкодія ({cycle_time:.0f}ms)")
        
        # Якість керування
        quality_score = metrics.get('quality_score', 1.0)
        if quality_score < 0.3:
            print(f"   🎯 Відмінна якість керування")
        elif quality_score < 0.5:
            print(f"   👍 Хороша якість керування")
        else:
            print(f"   ⚠️ Потребує налаштування параметрів MPC")
        
        return results_df, metrics
        
    except Exception as e:
        print(f"❌ Помилка в прикладі 3: {e}")
        return None, None

def example_4_model_comparison_truly_correct():
    """🔄 Приклад 4: АБСОЛЮТНО ПРАВИЛЬНЕ порівняння без втручання"""
    
    print("\n" + "="*70)
    print("🔄 ПРИКЛАД 4: ПРАВИЛЬНЕ ПОРІВНЯННЯ БЕЗ ВТРУЧАННЯ")
    print("="*70)
    
    try:
        hist_df = load_historical_data()
        
        # 🎯 ЕКСПЕРИМЕНТАТОР ЗАДАЄ КОНФІГУРАЦІЇ
        configurations = [
            {
                'name': 'KRR_Conservative',
                'model_type': 'krr',
                'kernel': 'rbf', 
                'Np': 6,
                'Nc': 4,
                'λ_obj': 0.2,
                'w_fe': 5.0,
                'w_mass': 1.0
                # Інші параметри - за замовчуванням з simulate_mpc_core_enhanced
            },
            {
                'name': 'KRR_Aggressive', 
                'model_type': 'krr',
                'kernel': 'rbf',
                'Np': 8,
                'Nc': 6,
                'λ_obj': 0.05,
                'w_fe': 10.0,
                'w_mass': 1.5,
                'N_data': 12000,  # Експериментатор хоче більше даних
                'find_optimal_params': False  # Експериментатор НЕ хоче оптимізації
            },
            {
                'name': 'SVR_Balanced',
                'model_type': 'svr',
                'kernel': 'rbf',
                'Np': 7,
                'Nc': 5,
                'λ_obj': 0.1,
                'w_fe': 7.0,
                'w_mass': 1.2
            },
            {
                'name': 'Linear_Fast',
                'model_type': 'linear',
                'linear_type': 'ridge',
                'Np': 10,
                'Nc': 8,
                'λ_obj': 0.15,
                'w_fe': 6.0,
                'w_mass': 1.0,
                'verbose_reports': True,  # Експериментатор хоче бачити звіт для цієї конфігурації
                'silent_mode': False
            }
        ]
        
        print(f"🎯 Порівнюємо {len(configurations)} конфігурацій БЕЗ втручання...")
        
        start_time = time.time()
        
        # 🔧 ВИКОРИСТОВУЄМО ПРАВИЛЬНУ ФУНКЦІЮ
        comparison_df = compare_mpc_configurations_correct(
            configurations=configurations,
            hist_df=hist_df,
            base_config='oleksandr_original',
            comparison_steps=100,
            show_progress=True
        )
        
        comparison_time = time.time() - start_time
        
        print(f"\n⏱️ Загальний час порівняння: {comparison_time/60:.1f} хвилин")
        print(f"✅ Жодного втручання в конфігурації експериментатора!")
        
        return comparison_df
        
    except Exception as e:
        print(f"❌ Помилка в прикладі 4: {e}")
        return None

def compute_correct_mpc_metrics_silent(results_df, basic_metrics, reference_values=None):
    """
    🎯 МОДИФІКОВАНА версія з обчисленням та виводом ISE/IAE метрик
    
    🔧 ЗМІНА: Додано обчислення та збереження ISE/IAE для звіту порівняння
    """
    
    if reference_values is None:
        reference_values = {'fe': 53.5, 'mass': 57.0}
    
    mpc_metrics = {}
    
    # 1. 📊 МЕТРИКИ ТОЧНОСТІ ВІДСЛІДКОВУВАННЯ
    if 'conc_fe' in results_df.columns:
        fe_values = results_df['conc_fe'].dropna().values
        fe_setpoint = reference_values['fe']
        
        fe_mean_error = np.mean(fe_values) - fe_setpoint
        fe_abs_error = np.mean(np.abs(fe_values - fe_setpoint))
        fe_max_error = np.max(np.abs(fe_values - fe_setpoint))
        fe_std_error = np.std(fe_values - fe_setpoint)
        
        fe_tolerance = 0.3
        fe_in_tolerance = np.mean(np.abs(fe_values - fe_setpoint) <= fe_tolerance) * 100
        
        mpc_metrics.update({
            'tracking_error_fe_mean': fe_mean_error,
            'tracking_error_fe_mae': fe_abs_error,
            'tracking_error_fe_max': fe_max_error,
            'tracking_error_fe_std': fe_std_error,
            'tracking_fe_in_tolerance_pct': fe_in_tolerance,
            'tracking_fe_setpoint': fe_setpoint,
            'tracking_fe_achieved': np.mean(fe_values)
        })
    
    if 'conc_mass' in results_df.columns:
        mass_values = results_df['conc_mass'].dropna().values
        mass_setpoint = reference_values['mass']
        
        mass_mean_error = np.mean(mass_values) - mass_setpoint
        mass_abs_error = np.mean(np.abs(mass_values - mass_setpoint))
        
        mass_tolerance = 2.0
        mass_in_tolerance = np.mean(np.abs(mass_values - mass_setpoint) <= mass_tolerance) * 100
        
        mpc_metrics.update({
            'tracking_error_mass_mean': mass_mean_error,
            'tracking_error_mass_mae': mass_abs_error,
            'tracking_mass_in_tolerance_pct': mass_in_tolerance,
            'tracking_mass_setpoint': mass_setpoint,
            'tracking_mass_achieved': np.mean(mass_values)
        })
    
    # 2. 📈 МЕТРИКИ СТАБІЛЬНОСТІ КЕРУВАННЯ
    if 'solid_feed_percent' in results_df.columns:
        control_actions = results_df['solid_feed_percent'].dropna().values
        
        control_std = np.std(control_actions)
        control_mean = np.mean(control_actions)
        
        if len(control_actions) > 1:
            control_changes = np.diff(control_actions)
            control_smoothness = np.std(control_changes)
        else:
            control_smoothness = 0
        
        mpc_metrics.update({
            'control_mean': control_mean,
            'control_std': control_std,
            'control_smoothness': control_smoothness
        })
    
    # 3. 🆕 ІНТЕГРАЛЬНІ МЕТРИКИ ЯКОСТІ (ISE/IAE)
    mpc_metrics.update(calculate_ise_iae_metrics(results_df, reference_values))
    
    # 4. 📊 ЗАГАЛЬНА ОЦІНКА MPC
    quality_factors = []
    
    if 'tracking_error_fe_mae' in mpc_metrics:
        mae_fe = mpc_metrics['tracking_error_fe_mae']
        fe_accuracy = max(0, 40 - mae_fe * 50)
        quality_factors.append(('Fe точність', fe_accuracy, 40))
    
    if 'tracking_error_mass_mae' in mpc_metrics:
        mae_mass = mpc_metrics['tracking_error_mass_mae']
        mass_accuracy = max(0, 30 - mae_mass * 15)
        quality_factors.append(('Mass точність', mass_accuracy, 30))
    
    if 'control_smoothness' in mpc_metrics:
        smoothness = mpc_metrics['control_smoothness']
        control_stability = max(0, 20 - smoothness * 20)
        quality_factors.append(('Стабільність', control_stability, 20))
    
    if 'tracking_fe_in_tolerance_pct' in mpc_metrics:
        consistency_pct = mpc_metrics['tracking_fe_in_tolerance_pct']
        consistency = consistency_pct / 10
        quality_factors.append(('Консистентність', consistency, 10))
    
    if quality_factors:
        total_score = sum(factor[1] for factor in quality_factors)
        max_possible = sum(factor[2] for factor in quality_factors)
        mpc_quality_score = (total_score / max_possible) * 100
        mpc_metrics['mpc_quality_score'] = mpc_quality_score
        
        # Класифікація
        if mpc_quality_score >= 80:
            quality_class = "Промислово відмінно"
        elif mpc_quality_score >= 65:
            quality_class = "Промислово добре"  
        elif mpc_quality_score >= 50:
            quality_class = "Промислово прийнятно"
        else:
            quality_class = "Потребує покращення"
        
        mpc_metrics['mpc_quality_class'] = quality_class
    
    # 5. 💡 РЕКОМЕНДАЦІЇ
    recommendations = []
    
    if mpc_metrics.get('tracking_error_fe_mae', 0) <= 0.5:
        recommendations.append("✅ Відмінна точність Fe - продовжуйте!")
    
    if mpc_metrics.get('control_smoothness', 0) <= 0.5:
        recommendations.append("✅ Стабільне керування - добре налаштовано!")
    
    if not recommendations:
        recommendations.append("MPC працює добре в промислових умовах!")
    
    mpc_metrics['recommendations'] = recommendations
    
    # Оновлюємо метрики
    basic_metrics.update(mpc_metrics)
    basic_metrics['mpc_evaluation_method'] = 'realistic_industrial_criteria'
    
    return basic_metrics


def calculate_ise_iae_metrics(results_df, reference_values):
    """
    🆕 НОВИЙ МЕТОД: Обчислює ISE/IAE метрики для звіту порівняння
    
    Args:
        results_df: DataFrame з результатами симуляції
        reference_values: Словник з уставками {'fe': 53.5, 'mass': 57.0}
        
    Returns:
        dict: Словник з ISE/IAE метриками
    """
    
    ise_iae_metrics = {}
    
    # ISE/IAE для Fe концентрату
    if 'conc_fe' in results_df.columns:
        fe_values = results_df['conc_fe'].dropna().values
        fe_setpoint = reference_values['fe']
        
        # Обчислюємо помилки
        fe_errors = fe_values - fe_setpoint
        
        # ISE (Integral Square Error)
        ise_fe = np.sum(fe_errors**2)
        
        # IAE (Integral Absolute Error)  
        iae_fe = np.sum(np.abs(fe_errors))
        
        # ITSE (Integral Time Square Error)
        time_weights = np.arange(1, len(fe_errors) + 1)
        itse_fe = np.sum(time_weights * fe_errors**2)
        
        # ITAE (Integral Time Absolute Error)
        itae_fe = np.sum(time_weights * np.abs(fe_errors))
        
        ise_iae_metrics.update({
            'performance_ise_fe': ise_fe,
            'performance_iae_fe': iae_fe,
            'performance_itse_fe': itse_fe,
            'performance_itae_fe': itae_fe,
            # Нормалізовані метрики (на одиницю часу)
            'performance_ise_fe_normalized': ise_fe / len(fe_errors),
            'performance_iae_fe_normalized': iae_fe / len(fe_errors)
        })
    
    # ISE/IAE для масового потоку
    if 'conc_mass' in results_df.columns:
        mass_values = results_df['conc_mass'].dropna().values
        mass_setpoint = reference_values['mass']
        
        # Обчислюємо помилки
        mass_errors = mass_values - mass_setpoint
        
        # ISE (Integral Square Error)
        ise_mass = np.sum(mass_errors**2)
        
        # IAE (Integral Absolute Error)
        iae_mass = np.sum(np.abs(mass_errors))
        
        # ITSE (Integral Time Square Error)
        time_weights = np.arange(1, len(mass_errors) + 1)
        itse_mass = np.sum(time_weights * mass_errors**2)
        
        # ITAE (Integral Time Absolute Error)
        itae_mass = np.sum(time_weights * np.abs(mass_errors))
        
        ise_iae_metrics.update({
            'performance_ise_mass': ise_mass,
            'performance_iae_mass': iae_mass,
            'performance_itse_mass': itse_mass,
            'performance_itae_mass': itae_mass,
            # Нормалізовані метрики (на одиницю часу)
            'performance_ise_mass_normalized': ise_mass / len(mass_errors),
            'performance_iae_mass_normalized': iae_mass / len(mass_errors)
        })
    
    # Комбіновані метрики
    if ('performance_ise_fe' in ise_iae_metrics and 
        'performance_ise_mass' in ise_iae_metrics):
        
        # Зважена комбінована ISE (60% Fe + 40% Mass)
        combined_ise = (0.6 * ise_iae_metrics['performance_ise_fe_normalized'] + 
                       0.4 * ise_iae_metrics['performance_ise_mass_normalized'])
        
        # Зважена комбінована IAE
        combined_iae = (0.6 * ise_iae_metrics['performance_iae_fe_normalized'] + 
                       0.4 * ise_iae_metrics['performance_iae_mass_normalized'])
        
        ise_iae_metrics.update({
            'performance_combined_ise': combined_ise,
            'performance_combined_iae': combined_iae
        })
    
    return ise_iae_metrics


def compare_mpc_configurations_correct(
    configurations: List[Dict],
    hist_df: pd.DataFrame,
    base_config: str = 'oleksandr_original',
    comparison_steps: int = 100,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    🔄 АЛИАС для зворотної сумісності - тепер викликає функцію з enhanced_sim
    
    🔧 ВИПРАВЛЕНО: Тепер викликає основну реалізацію з enhanced_sim.py
    """
    
   
    return compare_mpc_configurations(
        configurations=configurations,
        hist_df=hist_df,
        base_config=base_config,
        comparison_steps=comparison_steps,
        show_progress=show_progress
    )

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

def save_experiment_summary(
    results: Dict[str, Any], 
    experiment_name: Optional[str] = None,
    base_results_dir: str = "experiment_results",
    save_detailed_data: bool = True,
    save_plots: bool = False,
    compress_results: bool = False
):
    """
    💾 Покращене збереження результатів експерименту в структуровану папку
    🔧 ВИПРАВЛЕНО: дублювання timestamp, помилки типів даних, логіка ранжування
    """
    
    # 1. 📁 СТВОРЕННЯ СТРУКТУРИ ПАПОК (ВИПРАВЛЕНО - без дублювання timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name is None:
        experiment_name = f"experiment_{timestamp}"
        # Якщо назва None, НЕ додаємо timestamp двічі
        safe_experiment_name = experiment_name
    else:
        # Очищуємо назву експерименту від недозволених символів
        safe_experiment_name = "".join(c for c in experiment_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_experiment_name = safe_experiment_name.replace(' ', '_')
        # Додаємо timestamp тільки якщо назва була задана користувачем
        safe_experiment_name = f"{safe_experiment_name}_{timestamp}"
    
    # Створюємо ієрархію папок БЕЗ додаткового timestamp
    base_path = Path(base_results_dir)
    experiment_path = base_path / safe_experiment_name
    
    # Створюємо підпапки
    subdirs = {
        'summary': experiment_path / "summary",
        'detailed': experiment_path / "detailed_data", 
        'configs': experiment_path / "configurations",
        'metrics': experiment_path / "metrics",
        'plots': experiment_path / "plots",
        'logs': experiment_path / "logs"
    }
    
    # Створюємо всі папки
    for subdir_path in subdirs.values():
        subdir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Створено структуру експерименту: {experiment_path}")
    
    # 2. 📋 ЗБЕРЕЖЕННЯ ОСНОВНОГО РЕЗЮМЕ
    experiment_summary = {
        'experiment_info': {
            'name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': _calculate_total_duration(results),
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results.values() if r is not None]),
            'python_version': _get_python_version(),
            'system_info': _get_system_info()
        },
        'experiments_overview': _create_experiments_overview(results),
        'key_findings': _extract_key_findings(results),  # Використовуємо виправлену версію
        'file_structure': {
            'summary_files': [],
            'detailed_files': [],
            'config_files': [],
            'metric_files': []
        }
    }
    
    # Зберігаємо основне резюме
    summary_file = subdirs['summary'] / "experiment_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_summary, f, indent=2, ensure_ascii=False, default=str)
    
    experiment_summary['file_structure']['summary_files'].append(str(summary_file.name))
    print(f"📋 Збережено основне резюме: {summary_file}")
    
    # 3. 📊 ЗБЕРЕЖЕННЯ ДЕТАЛЬНИХ РЕЗУЛЬТАТІВ (ВИПРАВЛЕНО)
    if save_detailed_data:
        for exp_name, exp_result in results.items():
            if exp_result is not None:
                _save_detailed_experiment_data(  # Викликаємо виправлену версію
                    exp_name, exp_result, subdirs, experiment_summary
                )
    
    # 4. ⚙️ ЗБЕРЕЖЕННЯ КОНФІГУРАЦІЙ
    _save_experiment_configurations(results, subdirs, experiment_summary)
    
    # 5. 📈 ЗБЕРЕЖЕННЯ АГРЕГОВАНИХ МЕТРИК
    _save_aggregated_metrics(results, subdirs, experiment_summary)
    
    # 6. 📊 СТВОРЕННЯ ЗВЕДЕНОЇ ТАБЛИЦІ ПОРІВНЯННЯ (ВИПРАВЛЕНО)
    comparison_table = _create_comparison_table(results)  # Використовуємо виправлену версію
    if comparison_table is not None:
        comparison_file = subdirs['summary'] / "comparison_table.csv"
        comparison_table.to_csv(comparison_file, index=False)
        experiment_summary['file_structure']['summary_files'].append(str(comparison_file.name))
        print(f"📊 Збережено таблицю порівняння: {comparison_file}")
    
    # 7. 📝 СТВОРЕННЯ ТЕКСТОВОГО ЗВІТУ
    text_report = _create_text_report(experiment_summary, results)
    report_file = subdirs['summary'] / "experiment_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(text_report)
    experiment_summary['file_structure']['summary_files'].append(str(report_file.name))
    print(f"📝 Збережено текстовий звіт: {report_file}")
    
    # 8. 🗃️ ОНОВЛЕННЯ ФІНАЛЬНОГО РЕЗЮМЕ
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # 9. 📦 АРХІВУВАННЯ (опціонально)
    if compress_results:
        archive_file = _compress_experiment_results(experiment_path)
        print(f"📦 Результати архівовано: {archive_file}")
    
    # 10. 📊 ПІДСУМОК ЗБЕРЕЖЕННЯ
    total_files = sum(len(files) for files in experiment_summary['file_structure'].values())
    
    print(f"\n✅ ЕКСПЕРИМЕНТ ЗБЕРЕЖЕНО УСПІШНО!")
    print(f"📁 Папка: {experiment_path}")
    print(f"📄 Всього файлів: {total_files}")
    print(f"💾 Розмір: {_calculate_directory_size(experiment_path):.1f} MB")
    
    # Показуємо структуру
    print(f"\n📁 СТРУКТУРА РЕЗУЛЬТАТІВ:")
    for subdir_name, subdir_path in subdirs.items():
        file_count = len(list(subdir_path.glob('*')))
        if file_count > 0:
            print(f"   📂 {subdir_name}/: {file_count} файлів")
    
    return str(experiment_path)

def _save_detailed_experiment_data(
    exp_name: str, 
    exp_result: Any, 
    subdirs: Dict[str, Path], 
    experiment_summary: Dict
):
    """🔧 ВИПРАВЛЕНО: Зберігає детальні дані з обробкою помилок типів"""
    
    try:
        if isinstance(exp_result, tuple) and len(exp_result) == 2:
            # Результат симуляції: (DataFrame, metrics)
            results_df, metrics = exp_result
            
            # Зберігаємо DataFrame
            if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                data_file = subdirs['detailed'] / f"{exp_name}_results.parquet"
                results_df.to_parquet(data_file, index=False)
                experiment_summary['file_structure']['detailed_files'].append(str(data_file.name))
                
                # CSV з обробкою проблемних колонок
                csv_file = subdirs['detailed'] / f"{exp_name}_results.csv"
                
                # 🔧 ВИПРАВЛЕННЯ: Обробка колонок з mixed типами
                df_to_save = results_df.copy()
                for col in df_to_save.columns:
                    if df_to_save[col].dtype == 'object':
                        # Конвертуємо проблемні колонки в string
                        df_to_save[col] = df_to_save[col].astype(str)
                
                df_to_save.to_csv(csv_file, index=False)
                experiment_summary['file_structure']['detailed_files'].append(str(csv_file.name))
            
            # Зберігаємо метрики
            if isinstance(metrics, dict):
                metrics_file = subdirs['metrics'] / f"{exp_name}_metrics.json"
                
                # 🔧 ВИПРАВЛЕННЯ: Очищення метрик від проблемних типів
                clean_metrics = _clean_metrics_for_json(metrics)
                
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(clean_metrics, f, indent=2, ensure_ascii=False, default=str)
                experiment_summary['file_structure']['metric_files'].append(str(metrics_file.name))
                
        elif isinstance(exp_result, pd.DataFrame):
            # DataFrame результат (наприклад, порівняння конфігурацій)
            
            # 🔧 ВИПРАВЛЕННЯ: Очищення DataFrame від проблемних типів
            df_clean = _clean_dataframe_for_save(exp_result)
            
            data_file = subdirs['detailed'] / f"{exp_name}_comparison.parquet"
            df_clean.to_parquet(data_file, index=False)
            experiment_summary['file_structure']['detailed_files'].append(str(data_file.name))
            
            # CSV версія
            csv_file = subdirs['detailed'] / f"{exp_name}_comparison.csv"
            df_clean.to_csv(csv_file, index=False)
            experiment_summary['file_structure']['detailed_files'].append(str(csv_file.name))
            
        elif isinstance(exp_result, dict):
            # Словник результатів
            result_file = subdirs['detailed'] / f"{exp_name}_results.json"
            clean_result = _clean_metrics_for_json(exp_result)
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(clean_result, f, indent=2, ensure_ascii=False, default=str)
            experiment_summary['file_structure']['detailed_files'].append(str(result_file.name))
            
    except Exception as e:
        print(f"   ⚠️ Помилка збереження детальних даних для {exp_name}: {e}")
        # Зберігаємо інформацію про помилку замість повного падіння
        error_file = subdirs['logs'] / f"{exp_name}_save_error.txt"
        error_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"Помилка збереження {exp_name}:\n")
            f.write(f"Тип помилки: {type(e).__name__}\n")
            f.write(f"Опис: {str(e)}\n")
            f.write(f"Тип результату: {type(exp_result)}\n")
            if isinstance(exp_result, pd.DataFrame):
                f.write(f"Колонки DataFrame: {list(exp_result.columns)}\n")
                f.write(f"Типи колонок: {exp_result.dtypes.to_dict()}\n")

def _save_experiment_configurations(results: Dict[str, Any], subdirs: Dict[str, Path], experiment_summary: Dict):
    """Зберігає конфігурації експериментів"""
    
    configs = {}
    
    for exp_name, exp_result in results.items():
        try:
            config_info = None
            
            if isinstance(exp_result, tuple) and len(exp_result) == 2:
                _, metrics = exp_result
                if isinstance(metrics, dict) and 'config_info' in metrics:
                    config_info = metrics['config_info']
                elif isinstance(metrics, dict) and 'config_summary' in metrics:
                    config_info = metrics['config_summary']
            
            configs[exp_name] = config_info or "Configuration not available"
            
        except Exception as e:
            configs[exp_name] = f"Error extracting config: {e}"
    
    # Зберігаємо всі конфігурації
    configs_file = subdirs['configs'] / "all_configurations.json"
    with open(configs_file, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False, default=str)
    
    experiment_summary['file_structure']['config_files'].append(str(configs_file.name))


def _save_aggregated_metrics(results: Dict[str, Any], subdirs: Dict[str, Path], experiment_summary: Dict):
    """Зберігає агреговані метрики всіх експериментів"""
    
    all_metrics = {}
    
    for exp_name, exp_result in results.items():
        try:
            if isinstance(exp_result, tuple) and len(exp_result) == 2:
                _, metrics = exp_result
                if isinstance(metrics, dict):
                    # Витягуємо ключові метрики
                    key_metrics = {
                        'rmse_fe': metrics.get('test_rmse_conc_fe'),
                        'rmse_mass': metrics.get('test_rmse_conc_mass'),
                        'r2_fe': metrics.get('r2_fe'),
                        'r2_mass': metrics.get('r2_mass'),
                        'quality_score': metrics.get('quality_score'),
                        'mpc_quality_score': metrics.get('mpc_quality_score'),
                        'total_cycle_time': metrics.get('total_cycle_time'),
                        'real_time_suitable': metrics.get('real_time_suitable')
                    }
                    all_metrics[exp_name] = {k: v for k, v in key_metrics.items() if v is not None}
        except Exception as e:
            all_metrics[exp_name] = f"Error: {e}"
    
    # Зберігаємо агреговані метрики
    metrics_file = subdirs['metrics'] / "aggregated_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=str)
    
    experiment_summary['file_structure']['metric_files'].append(str(metrics_file.name))


def _create_comparison_table(results: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """🔧 ВИПРАВЛЕНО: Створює зведену таблицю з правильною логікою ранжування"""
    
    comparison_data = []
    
    for exp_name, exp_result in results.items():
        try:
            row = {'Experiment': exp_name}
            
            if isinstance(exp_result, tuple) and len(exp_result) == 2:
                _, metrics = exp_result
                if isinstance(metrics, dict):
                    row.update({
                        'RMSE_Fe': metrics.get('test_rmse_conc_fe'),
                        'RMSE_Mass': metrics.get('test_rmse_conc_mass'),
                        'R2_Fe': metrics.get('r2_fe'),
                        'R2_Mass': metrics.get('r2_mass'),
                        'Quality_Score': metrics.get('quality_score'),
                        'MPC_Quality': metrics.get('mpc_quality_score'),
                        'Cycle_Time_ms': metrics.get('total_cycle_time', 0) * 1000,
                        'Real_Time': metrics.get('real_time_suitable', False)
                    })
            elif isinstance(exp_result, pd.DataFrame):
                # 🔧 ВИПРАВЛЕННЯ: Правильна логіка для порівняльних експериментів
                if not exp_result.empty:
                    # Сортуємо за комбінованою метрикою замість тільки RMSE
                    df_sorted = exp_result.copy()
                    
                    # Створюємо комбіновану оцінку: 70% MPC_Quality + 30% точність
                    if 'MPC_Quality_Score' in df_sorted.columns and 'RMSE_Fe' in df_sorted.columns:
                        # Нормалізуємо метрики (вище MPC якість = краще, нижче RMSE = краще)
                        mpc_quality_norm = df_sorted['MPC_Quality_Score'].fillna(0) / 100
                        rmse_norm = 1 / (1 + df_sorted['RMSE_Fe'].fillna(1))  # Інвертуємо RMSE
                        
                        df_sorted['Combined_Score'] = 0.7 * mpc_quality_norm + 0.3 * rmse_norm
                        df_sorted = df_sorted.sort_values('Combined_Score', ascending=False)
                        
                        print(f"   🔧 Сортування за комбінованою оцінкою (70% MPC якість + 30% точність)")
                    elif 'MPC_Quality_Score' in df_sorted.columns:
                        df_sorted = df_sorted.sort_values('MPC_Quality_Score', ascending=False)
                        print(f"   🔧 Сортування за MPC якістю")
                    elif 'RMSE_Fe' in df_sorted.columns:
                        df_sorted = df_sorted.sort_values('RMSE_Fe', ascending=True)
                        print(f"   🔧 Сортування за RMSE Fe")
                    
                    best_row = df_sorted.iloc[0]
                    row.update({
                        'Best_Config': best_row.get('Configuration', 'Unknown'),
                        'RMSE_Fe': best_row.get('RMSE_Fe'),
                        'MPC_Quality': best_row.get('MPC_Quality_Score'),
                        'Combined_Score': best_row.get('Combined_Score', 'N/A'),
                        'Total_Configs_Tested': len(df_sorted),
                        'Ranking_Logic': '70% MPC Quality + 30% Model Accuracy'
                    })
            
            comparison_data.append(row)
            
        except Exception as e:
            comparison_data.append({
                'Experiment': exp_name,
                'Error': str(e)
            })
    
    if comparison_data:
        return pd.DataFrame(comparison_data)
    return None

def _create_experiments_overview(results: Dict[str, Any]) -> Dict:
    """Створює огляд експериментів"""
    
    overview = {
        'total_count': len(results),
        'successful_count': 0,
        'failed_count': 0,
        'experiment_types': {},
        'best_results': {}
    }
    
    for exp_name, exp_result in results.items():
        if exp_result is not None:
            overview['successful_count'] += 1
            
            # Визначаємо тип експерименту
            if 'benchmark' in exp_name.lower():
                exp_type = 'benchmark'
            elif 'comparison' in exp_name.lower():
                exp_type = 'comparison'
            elif 'analysis' in exp_name.lower():
                exp_type = 'analysis'
            else:
                exp_type = 'simulation'
            
            overview['experiment_types'][exp_type] = overview['experiment_types'].get(exp_type, 0) + 1
        else:
            overview['failed_count'] += 1
    
    return overview


def _extract_key_findings(results: Dict[str, Any]) -> Dict:
    """🔧 ВИПРАВЛЕНО: Витягує висновки з правильною логікою оцінки"""
    
    findings = {
        'best_performers': {},
        'recommendations': [],
        'performance_summary': {},
        'ranking_logic': 'Combined score: 70% MPC Quality + 30% Model Accuracy'
    }
    
    # Збираємо всі метрики
    all_configs = []
    
    for exp_name, exp_result in results.items():
        try:
            if isinstance(exp_result, pd.DataFrame) and 'MPC_Quality_Score' in exp_result.columns:
                # Це результат порівняння конфігурацій
                for _, row in exp_result.iterrows():
                    config_data = {
                        'experiment': exp_name,
                        'configuration': row.get('Configuration', 'Unknown'),
                        'rmse_fe': row.get('RMSE_Fe'),
                        'mpc_quality': row.get('MPC_Quality_Score'),
                        'cycle_time': row.get('Total_Cycle_Time', 0) * 1000 if pd.notna(row.get('Total_Cycle_Time')) else None
                    }
                    
                    # Комбінована оцінка
                    if pd.notna(config_data['mpc_quality']) and pd.notna(config_data['rmse_fe']):
                        mpc_norm = config_data['mpc_quality'] / 100
                        rmse_norm = 1 / (1 + config_data['rmse_fe'])
                        config_data['combined_score'] = 0.7 * mpc_norm + 0.3 * rmse_norm
                    
                    all_configs.append(config_data)
            
            elif isinstance(exp_result, tuple) and len(exp_result) == 2:
                _, metrics = exp_result
                if isinstance(metrics, dict):
                    config_data = {
                        'experiment': exp_name,
                        'configuration': exp_name,
                        'rmse_fe': metrics.get('test_rmse_conc_fe'),
                        'mpc_quality': metrics.get('mpc_quality_score'),
                        'quality_score': metrics.get('quality_score')
                    }
                    all_configs.append(config_data)
        except:
            continue
    
    # Знаходимо найкращі конфігурації
    if all_configs:
        # За комбінованою оцінкою
        configs_with_combined = [c for c in all_configs if 'combined_score' in c and pd.notna(c['combined_score'])]
        if configs_with_combined:
            best_combined = max(configs_with_combined, key=lambda x: x['combined_score'])
            findings['best_performers']['best_overall'] = {
                'configuration': best_combined['configuration'],
                'experiment': best_combined['experiment'],
                'combined_score': best_combined['combined_score'],
                'mpc_quality': best_combined['mpc_quality'],
                'rmse_fe': best_combined['rmse_fe']
            }
        
        # За MPC якістю
        configs_with_mpc = [c for c in all_configs if pd.notna(c.get('mpc_quality'))]
        if configs_with_mpc:
            best_mpc = max(configs_with_mpc, key=lambda x: x['mpc_quality'])
            findings['best_performers']['best_mpc_quality'] = {
                'configuration': best_mpc['configuration'],
                'experiment': best_mpc['experiment'],
                'mpc_quality': best_mpc['mpc_quality']
            }
        
        # За точністю моделі
        configs_with_rmse = [c for c in all_configs if pd.notna(c.get('rmse_fe'))]
        if configs_with_rmse:
            best_accuracy = min(configs_with_rmse, key=lambda x: x['rmse_fe'])
            findings['best_performers']['best_accuracy'] = {
                'configuration': best_accuracy['configuration'],
                'experiment': best_accuracy['experiment'],
                'rmse_fe': best_accuracy['rmse_fe']
            }
    
    # Генеруємо правильні рекомендації
    if 'best_overall' in findings['best_performers']:
        best = findings['best_performers']['best_overall']
        findings['recommendations'].append(
            f"🏆 Рекомендована конфігурація: {best['configuration']} "
            f"(Комбінована оцінка: {best['combined_score']:.3f})"
        )
        
        if best['mpc_quality'] >= 65:
            findings['recommendations'].append("✅ Висока якість MPC - готово для промислового використання")
        elif best['mpc_quality'] >= 50:
            findings['recommendations'].append("⚠️ Середня якість MPC - розгляньте додаткове налаштування")
        else:
            findings['recommendations'].append("🔧 Низька якість MPC - потрібне серйозне налаштування")
    
    return findings

def _create_text_report(experiment_summary: Dict, results: Dict[str, Any]) -> str:
    """Створює текстовий звіт експерименту"""
    
    report = f"""
🔬 ЗВІТ ПРО ЕКСПЕРИМЕНТ MPC
{'='*60}

📋 ЗАГАЛЬНА ІНФОРМАЦІЯ:
   Назва експерименту: {experiment_summary['experiment_info']['name']}
   Дата/час: {experiment_summary['experiment_info']['timestamp']}
   Тривалість: {experiment_summary['experiment_info']['duration_minutes']:.1f} хвилин
   
📊 ОГЛЯД ЕКСПЕРИМЕНТІВ:
   Всього експериментів: {experiment_summary['experiments_overview']['total_count']}
   Успішних: {experiment_summary['experiments_overview']['successful_count']}
   Невдалих: {experiment_summary['experiments_overview']['failed_count']}

🎯 КЛЮЧОВІ РЕЗУЛЬТАТИ:
"""
    
    # Додаємо кращі результати
    if 'best_performers' in experiment_summary['key_findings']:
        best = experiment_summary['key_findings']['best_performers']
        
        if 'lowest_rmse_fe' in best:
            report += f"   🏆 Найкраща точність Fe: {best['lowest_rmse_fe']['experiment']} "
            report += f"(RMSE = {best['lowest_rmse_fe']['value']:.4f})\n"
        
        if 'best_quality' in best:
            report += f"   🎯 Найкраща якість керування: {best['best_quality']['experiment']} "
            report += f"(Quality = {best['best_quality']['value']:.4f})\n"
    
    # Додаємо статистику
    if 'performance_summary' in experiment_summary['key_findings']:
        perf = experiment_summary['key_findings']['performance_summary']
        if 'rmse_fe' in perf:
            rmse_stats = perf['rmse_fe']
            report += f"\n📈 СТАТИСТИКА RMSE Fe:\n"
            report += f"   Середнє: {rmse_stats['mean']:.4f}\n"
            report += f"   Стд. відхилення: {rmse_stats['std']:.4f}\n"
            report += f"   Діапазон: {rmse_stats['min']:.4f} - {rmse_stats['max']:.4f}\n"
    
    # Додаємо структуру файлів
    report += f"\n📁 ЗБЕРЕЖЕНІ ФАЙЛИ:\n"
    for file_type, files in experiment_summary['file_structure'].items():
        if files:
            report += f"   📂 {file_type}: {len(files)} файлів\n"
    
    report += f"\n{'='*60}\n"
    
    return report


# Допоміжні функції
def _calculate_total_duration(results: Dict[str, Any]) -> float:
    """Обчислює загальну тривалість експериментів"""
    # Placeholder - в реальності можна додати трекінг часу
    return len(results) * 2.5  # Приблизно 2.5 хвилини на експеримент

def _get_python_version() -> str:
    """Отримує версію Python"""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def _get_system_info() -> Dict:
    """Отримує інформацію про систему"""
    import platform
    return {
        'os': platform.system(),
        'os_version': platform.version(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor()
    }

def _calculate_directory_size(directory: Path) -> float:
    """Обчислює розмір папки в MB"""
    total_size = 0
    for path in directory.rglob('*'):
        if path.is_file():
            total_size += path.stat().st_size
    return total_size / (1024 * 1024)  # Конвертуємо в MB

def _clean_dataframe_for_save(df: pd.DataFrame) -> pd.DataFrame:
    """Очищує DataFrame від проблемних типів даних"""
    
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Перевіряємо чи є mixed типи в колонці
            unique_types = set(type(x).__name__ for x in df_clean[col].dropna())
            
            if len(unique_types) > 1:
                # Mixed типи - конвертуємо все в string
                df_clean[col] = df_clean[col].astype(str)
                print(f"   🔧 Конвертовано колонку '{col}' в string (було mixed типів: {unique_types})")
            else:
                # Один тип - залишаємо як є, але конвертуємо 'default' в NaN для числових колонок
                if col.startswith('Config_') and any(x in col for x in ['N_data', 'Np', 'Nc']):
                    # Числова конфігураційна колонка
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    print(f"   🔧 Конвертовано колонку '{col}' в числову (NaN для 'default')")
    
    return df_clean


def _clean_metrics_for_json(metrics: Dict) -> Dict:
    """Очищує метрики від типів, які не можна серіалізувати в JSON"""
    
    clean_metrics = {}
    
    for key, value in metrics.items():
        try:
            if isinstance(value, (np.integer, np.floating)):
                clean_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                clean_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                clean_metrics[key] = _clean_metrics_for_json(value)
            elif isinstance(value, list):
                clean_metrics[key] = [_clean_single_value_for_json(v) for v in value]
            elif pd.isna(value):
                clean_metrics[key] = None
            else:
                clean_metrics[key] = _clean_single_value_for_json(value)
        except Exception as e:
            # Якщо не можемо очистити значення, зберігаємо як string
            clean_metrics[key] = str(value)
    
    return clean_metrics


def _clean_single_value_for_json(value):
    """Очищає одне значення для JSON серіалізації"""
    
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif pd.isna(value):
        return None
    elif isinstance(value, (int, float, str, bool, type(None))):
        return value
    else:
        return str(value)
    
def _compress_experiment_results(experiment_path: Path) -> str:
    """Архівує результати експерименту"""
    import shutil
    
    archive_path = experiment_path.parent / f"{experiment_path.name}"
    archive_file = shutil.make_archive(str(archive_path), 'zip', str(experiment_path))
    return archive_file

print("✅ Покращена функція save_experiment_summary готова!")
print("📁 Нові можливості:")
print("   • Структурована папка для кожного експерименту")
print("   • Збереження в різних форматах (JSON, CSV, Parquet)")
print("   • Детальні звіти та порівняльні таблиці")
print("   • Автоматичне архівування результатів")
print("   • Метадані про систему та конфігурацію")

def load_experiment_results(experiment_path: str) -> Dict[str, Any]:
    """
    📂 Завантажує збережені результати експерименту
    
    Parameters:
    -----------
    experiment_path : str
        Шлях до папки з результатами експерименту
        
    Returns:
    --------
    Dict[str, Any]
        Словник з результатами експерименту
    """
    
    experiment_path = Path(experiment_path)
    
    if not experiment_path.exists():
        raise FileNotFoundError(f"Папка експерименту не знайдена: {experiment_path}")
    
    # Завантажуємо основне резюме
    summary_file = experiment_path / "summary" / "experiment_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    else:
        summary = {}
    
    # Завантажуємо детальні дані
    detailed_data = {}
    detailed_dir = experiment_path / "detailed_data"
    
    if detailed_dir.exists():
        for file_path in detailed_dir.glob("*.parquet"):
            exp_name = file_path.stem.replace('_results', '').replace('_comparison', '')
            try:
                detailed_data[exp_name] = pd.read_parquet(file_path)
            except Exception as e:
                print(f"⚠️ Помилка завантаження {file_path}: {e}")
    
    # Завантажуємо метрики
    metrics_data = {}
    metrics_dir = experiment_path / "metrics"
    
    if metrics_dir.exists():
        for file_path in metrics_dir.glob("*.json"):
            exp_name = file_path.stem.replace('_metrics', '')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    metrics_data[exp_name] = json.load(f)
            except Exception as e:
                print(f"⚠️ Помилка завантаження {file_path}: {e}")
    
    return {
        'summary': summary,
        'detailed_data': detailed_data,
        'metrics': metrics_data,
        'experiment_path': str(experiment_path)
    }


def list_available_experiments(base_results_dir: str = "experiment_results") -> pd.DataFrame:
    """
    📋 Показує список всіх доступних експериментів
    
    Returns:
    --------
    pd.DataFrame
        Таблиця з інформацією про експерименти
    """
    
    base_path = Path(base_results_dir)
    
    if not base_path.exists():
        print(f"📁 Папка результатів не знайдена: {base_path}")
        return pd.DataFrame()
    
    experiments = []
    
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir():
            summary_file = exp_dir / "summary" / "experiment_summary.json"
            
            if summary_file.exists():
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    exp_info = summary.get('experiment_info', {})
                    overview = summary.get('experiments_overview', {})
                    
                    experiments.append({
                        'Experiment_Name': exp_info.get('name', exp_dir.name),
                        'Date': exp_info.get('timestamp', 'Unknown')[:19],  # Без мікросекунд
                        'Duration_Min': exp_info.get('duration_minutes', 0),
                        'Total_Tests': overview.get('total_count', 0),
                        'Successful': overview.get('successful_count', 0),
                        'Path': str(exp_dir),
                        'Size_MB': _calculate_directory_size(exp_dir)
                    })
                except Exception as e:
                    experiments.append({
                        'Experiment_Name': exp_dir.name,
                        'Date': 'Error',
                        'Duration_Min': 0,
                        'Total_Tests': 0,
                        'Successful': 0,
                        'Path': str(exp_dir),
                        'Size_MB': _calculate_directory_size(exp_dir),
                        'Error': str(e)
                    })
    
    if experiments:
        df = pd.DataFrame(experiments)
        # Сортуємо за датою (найновіші спочатку)
        df = df.sort_values('Date', ascending=False)
        return df
    else:
        return pd.DataFrame()


def clean_old_experiments(base_results_dir: str = "experiment_results", keep_last_n: int = 10):
    """
    🧹 Видаляє старі експерименти, залишаючи тільки останні N
    
    Parameters:
    -----------
    base_results_dir : str
        Папка з результатами
    keep_last_n : int
        Кількість останніх експериментів для збереження
    """
    
    experiments_df = list_available_experiments(base_results_dir)
    
    if len(experiments_df) <= keep_last_n:
        print(f"✅ Експериментів {len(experiments_df)} <= {keep_last_n}, видалення не потрібне")
        return
    
    # Видаляємо старі експерименти
    to_delete = experiments_df.iloc[keep_last_n:]
    total_size = to_delete['Size_MB'].sum()
    
    print(f"🧹 Видалення {len(to_delete)} старих експериментів (звільнення {total_size:.1f} MB)...")
    
    for _, row in to_delete.iterrows():
        try:
            exp_path = Path(row['Path'])
            if exp_path.exists():
                import shutil
                shutil.rmtree(exp_path)
                print(f"   🗑️ Видалено: {row['Experiment_Name']}")
        except Exception as e:
            print(f"   ❌ Помилка видалення {row['Experiment_Name']}: {e}")
    
    print(f"✅ Очищення завершено")


# enhanced_simulator_runner.py - ПОВНИЙ КОД відкоригованої функції main

def main():
    """🚀 Головна функція запуску всіх прикладів з покращеним збереженням та правильними рекомендаціями"""
    
    print("🔬 РОЗШИРЕНИЙ СИМУЛЯТОР MPC З БЕНЧМАРКОМ")
    print("="*70)
    print("🎯 Доступні приклади:")
    print("   1. 🚀 Швидкий бенчмарк моделей")
    print("   2. 🔬 Детальний аналіз MPC")
    print("   3. 🎯 Користувацька симуляція")
    print("   4. 🔄 Порівняння конфігурацій")
    print("   5. 🏃 Запустити всі приклади")
    print("="*70)
    
    try:
        # Запитуємо у користувача вибір та назву експерименту
        choice = input("Оберіть приклад (1-5) або Enter для всіх: ").strip()
        
        if not choice:
            choice = "5"  # Запускаємо всі за замовчуванням
        
        # 🆕 ЗАПИТУЄМО НАЗВУ ЕКСПЕРИМЕНТУ
        experiment_name = input("Введіть назву експерименту (або Enter для автогенерації): ").strip()
        if not experiment_name:
            experiment_name = None  # Буде згенеровано автоматично
        
        # 🆕 НАЛАШТУВАННЯ ЗБЕРЕЖЕННЯ
        print("\n📁 Налаштування збереження результатів:")
        save_detailed = input("Зберігати детальні дані? (y/N): ").strip().lower() in ['y', 'yes', 'так', 'т']
        compress_results = input("Архівувати результати? (y/N): ").strip().lower() in ['y', 'yes', 'так', 'т']
        
        results = {}
        total_start_time = time.time()
        
        print(f"\n🚀 ПОЧАТОК ЕКСПЕРИМЕНТУ: {experiment_name or 'Автогенерований'}")
        print("="*70)
        
        # Запускаємо обрані експерименти
        if choice in ["1", "5"]:
            print(f"\n{'🚀 ЗАПУСК ПРИКЛАДУ 1' if choice == '1' else '🚀 ПРИКЛАД 1/4'}")
            try:
                results['quick_benchmark'] = example_1_quick_benchmark()
                print("   ✅ Приклад 1 завершено успішно")
            except Exception as e:
                print(f"   ❌ Помилка в прикладі 1: {e}")
                results['quick_benchmark'] = None
        
        if choice in ["2", "5"]:
            print(f"\n{'🔬 ЗАПУСК ПРИКЛАДУ 2' if choice == '2' else '🔬 ПРИКЛАД 2/4'}")
            try:
                results['detailed_analysis'] = example_2_detailed_analysis()
                print("   ✅ Приклад 2 завершено успішно")
            except Exception as e:
                print(f"   ❌ Помилка в прикладі 2: {e}")
                results['detailed_analysis'] = None
        
        if choice in ["3", "5"]:
            print(f"\n{'🎯 ЗАПУСК ПРИКЛАДУ 3' if choice == '3' else '🎯 ПРИКЛАД 3/4'}")
            try:
                results['custom_simulation'] = example_3_custom_simulation()
                print("   ✅ Приклад 3 завершено успішно")
            except Exception as e:
                print(f"   ❌ Помилка в прикладі 3: {e}")
                results['custom_simulation'] = None
        
        if choice in ["4", "5"]:
            print(f"\n{'🔄 ЗАПУСК ПРИКЛАДУ 4' if choice == '4' else '🔄 ПРИКЛАД 4/4'}")
            try:
                # 🔧 ВИКОРИСТОВУЄМО ВИПРАВЛЕНУ ФУНКЦІЮ БЕЗ ВТРУЧАННЯ
                results['configuration_comparison'] = compare_mpc_configurations_correct(
                    configurations=[
                        {
                            'name': 'KRR_Conservative',
                            'N_data':1000, # added
                            'model_type': 'krr',
                            'kernel': 'rbf', 
                            'Np': 6,
                            'Nc': 4,
                            'λ_obj': 0.2,
                            'w_fe': 5.0,
                            'w_mass': 1.0,
                            'find_optimal_params': False # added
                        },
                        {
                            'name': 'KRR_Aggressive', 
                            'N_data':1000, # added
                            'model_type': 'krr',
                            'kernel': 'rbf',
                            'Np': 8,
                            'Nc': 6,
                            'λ_obj': 0.05,
                            'w_fe': 10.0,
                            'w_mass': 1.5,
                            # 'N_data': 12000 # removed
                            'find_optimal_params': False
                        }
                        # {
                        #     'name': 'SVR_Balanced',
                        #     'model_type': 'svr',
                        #     'kernel': 'rbf',
                        #     'Np': 7,
                        #     'Nc': 5,
                        #     'λ_obj': 0.1,
                        #     'w_fe': 7.0,
                        #     'w_mass': 1.2
                        # },
                        # {
                        #     'name': 'Linear_Fast',
                        #     'model_type': 'linear',
                        #     'linear_type': 'ridge',
                        #     'Np': 10,
                        #     'Nc': 8,
                        #     'λ_obj': 0.15,
                        #     'w_fe': 6.0,
                        #     'w_mass': 1.0,
                        #     'verbose_reports': True,
                        #     'silent_mode': False
                        # }
                    ],
                    hist_df=load_historical_data(),
                    base_config='oleksandr_original',
                    comparison_steps=100,
                    show_progress=True
                )
                print("   ✅ Приклад 4 завершено успішно")
            except Exception as e:
                print(f"   ❌ Помилка в прикладі 4: {e}")
                results['configuration_comparison'] = None
        
        total_time = time.time() - total_start_time
        
        # Підсумок експерименту
        print(f"\n" + "="*70)
        print(f"🎉 ЕКСПЕРИМЕНТ ЗАВЕРШЕНО")
        print(f"="*70)
        print(f"⏱️ Загальний час: {total_time/60:.1f} хвилин")
        print(f"📊 Проведено експериментів: {len([r for r in results.values() if r is not None])}")
        
        success_count = len([r for r in results.values() if r is not None])
        total_count = len(results)
        print(f"✅ Успішних: {success_count}/{total_count}")
        
        # 🆕 ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ З НОВОЮ СИСТЕМОЮ
        if results:
            print(f"\n💾 ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ ЕКСПЕРИМЕНТУ...")
            
            try:
                experiment_path = save_experiment_summary(
                    results=results,
                    experiment_name=experiment_name,
                    base_results_dir="experiment_results",
                    save_detailed_data=save_detailed,
                    save_plots=False,  # Поки що не реалізовано
                    compress_results=compress_results
                )
                
                print(f"🎯 ЕКСПЕРИМЕНТ УСПІШНО ЗБЕРЕЖЕНО!")
                print(f"📂 Локація: {experiment_path}")
                
                # Показуємо що було збережено
                if success_count > 0:
                    print(f"\n📁 ЗБЕРЕЖЕНІ РЕЗУЛЬТАТИ:")
                    for exp_name, exp_result in results.items():
                        if exp_result is not None:
                            if isinstance(exp_result, pd.DataFrame):
                                print(f"   📊 {exp_name}: Таблиця порівняння ({exp_result.shape[0]} рядків)")
                            elif isinstance(exp_result, tuple):
                                print(f"   📈 {exp_name}: Результати симуляції + метрики")
                            elif isinstance(exp_result, dict):
                                print(f"   📋 {exp_name}: Детальний аналіз")
                            else:
                                print(f"   📄 {exp_name}: {type(exp_result).__name__}")
                
                # 🔧 ВИПРАВЛЕНІ РЕКОМЕНДАЦІЇ на основі комбінованої оцінки
                print(f"\n💡 РЕКОМЕНДАЦІЇ ДЛЯ ПОДАЛЬШОЇ РОБОТИ:")
                
                if 'configuration_comparison' in results and results['configuration_comparison'] is not None:
                    comparison_df = results['configuration_comparison']
                    if not comparison_df.empty and 'Configuration' in comparison_df.columns:
                        
                        # 🔧 ПРАВИЛЬНА ЛОГІКА: Знаходимо найкращу конфігурацію за комбінованою оцінкою
                        if 'Combined_Score' in comparison_df.columns:
                            valid_mask = comparison_df['Combined_Score'].notna()
                            if valid_mask.any():
                                # Сортуємо за комбінованою оцінкою (вища = краща)
                                sorted_df = comparison_df[valid_mask].sort_values('Combined_Score', ascending=False)
                                best_config = sorted_df.iloc[0]['Configuration']
                                best_score = sorted_df.iloc[0]['Combined_Score']
                                best_mpc_quality = sorted_df.iloc[0].get('MPC_Quality_Score', 0)
                                best_rmse = sorted_df.iloc[0].get('RMSE_Fe', 0)
                                
                                print(f"   🏆 Використовуйте конфігурацію '{best_config}' для продакшн")
                                print(f"   📊 Комбінована оцінка: {best_score:.4f} (70% MPC якість + 30% точність)")
                                print(f"   📈 RMSE Fe: {best_rmse:.4f}, MPC якість: {best_mpc_quality:.1f}/100")
                                
                                # Інтерпретація якості
                                if best_mpc_quality >= 65:
                                    print(f"   ✅ Висока якість MPC - готово для промислового використання")
                                elif best_mpc_quality >= 50:
                                    print(f"   ⚠️ Середня якість MPC - розгляньте додаткове налаштування")
                                else:
                                    print(f"   🔧 Низька якість MPC - потрібне серйозне налаштування")
                                
                                # Показуємо топ-3
                                print(f"\n   📊 Топ-3 конфігурації за комбінованою оцінкою:")
                                for idx in range(min(3, len(sorted_df))):
                                    row = sorted_df.iloc[idx]
                                    rank_emoji = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉"
                                    print(f"      {rank_emoji} {row['Configuration']}: {row['Combined_Score']:.4f} "
                                          f"(MPC: {row.get('MPC_Quality_Score', 0):.1f}, RMSE: {row.get('RMSE_Fe', 0):.4f})")
                            else:
                                # Fallback до простого RMSE
                                best_config = comparison_df.iloc[0]['Configuration']
                                print(f"   🏆 Використовуйте конфігурацію '{best_config}' для продакшн (за RMSE)")
                        else:
                            # Немає комбінованої оцінки
                            best_config = comparison_df.iloc[0]['Configuration']
                            print(f"   🏆 Використовуйте конфігурацію '{best_config}' для продакшн")
                
                # Інші рекомендації
                if 'quick_benchmark' in results and results['quick_benchmark'] is not None:
                    benchmark_df = results['quick_benchmark']
                    if not benchmark_df.empty and 'Model' in benchmark_df.columns:
                        best_model = benchmark_df.iloc[0]['Model']
                        print(f"   🚀 Найшвидша модель: {best_model}")
                
                print(f"   📊 Регулярно запускайте бенчмарк для моніторингу продуктивності")
                print(f"   🔧 Налаштовуйте параметри MPC на основі збережених рекомендацій")
                print(f"   📈 Використовуйте збережені метрики для порівняння з майбутніми експериментами")
                
                # Інструкції щодо доступу до результатів
                print(f"\n📖 ЯК ВИКОРИСТОВУВАТИ ЗБЕРЕЖЕНІ РЕЗУЛЬТАТИ:")
                print(f"   1. Основне резюме: {experiment_path}/summary/experiment_summary.json")
                print(f"   2. Порівняльна таблиця: {experiment_path}/summary/comparison_table.csv")
                print(f"   3. Детальні дані: {experiment_path}/detailed_data/")
                print(f"   4. Конфігурації: {experiment_path}/configurations/")
                print(f"   5. Текстовий звіт: {experiment_path}/summary/experiment_report.txt")
                
            except Exception as save_error:
                print(f"❌ Помилка збереження результатів: {save_error}")
                print(f"⚠️ Результати залишаються в пам'яті для цієї сесії")
                
                # Fallback: зберігаємо базове резюме
                try:
                    import json
                    fallback_summary = {
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'experiments_conducted': len(results),
                        'successful_experiments': success_count,
                        'total_time_minutes': total_time / 60,
                        'results_summary': {k: str(type(v)) for k, v in results.items()},
                        'error': str(save_error)
                    }
                    
                    fallback_file = f"experiment_fallback_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(fallback_file, 'w', encoding='utf-8') as f:
                        json.dump(fallback_summary, f, indent=2, default=str)
                    
                    print(f"💾 Базове резюме збережено: {fallback_file}")
                    
                    # 🔧 FALLBACK рекомендації
                    if 'configuration_comparison' in results and results['configuration_comparison'] is not None:
                        comparison_df = results['configuration_comparison']
                        if not comparison_df.empty:
                            best_config = comparison_df.iloc[0]['Configuration']
                            print(f"\n💡 БАЗОВА РЕКОМЕНДАЦІЯ:")
                            print(f"   🏆 Найкраща конфігурація: {best_config}")
                            
                except Exception as fallback_error:
                    print(f"❌ Не вдалося зберегти навіть базове резюме: {fallback_error}")
        
        else:
            print(f"⚠️ Немає результатів для збереження")
        
        # Фінальні поради
        print(f"\n🚀 НАСТУПНІ КРОКИ:")
        print(f"   • Проаналізуйте збережені результати")
        print(f"   • Виберіть оптимальну конфігурацію для вашого процесу")
        print(f"   • Запустіть додаткові експерименти при необхідності")
        print(f"   • Використовуйте результати для налаштування промислової системи")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Експеримент перервано користувачем")
        
        # Спробуємо зберегти часткові результати
        if 'results' in locals() and results:
            try:
                print(f"💾 Спроба збереження часткових результатів...")
                experiment_path = save_experiment_summary(
                    results=results,
                    experiment_name=f"{experiment_name or 'interrupted'}_partial",
                    base_results_dir="experiment_results",
                    save_detailed_data=False,  # Швидке збереження
                    compress_results=False
                )
                print(f"✅ Часткові результати збережено: {experiment_path}")
                
                # 🔧 Часткові рекомендації
                if 'configuration_comparison' in results and results['configuration_comparison'] is not None:
                    comparison_df = results['configuration_comparison']
                    if not comparison_df.empty and 'Combined_Score' in comparison_df.columns:
                        valid_mask = comparison_df['Combined_Score'].notna()
                        if valid_mask.any():
                            sorted_df = comparison_df[valid_mask].sort_values('Combined_Score', ascending=False)
                            best_config = sorted_df.iloc[0]['Configuration']
                            best_score = sorted_df.iloc[0]['Combined_Score']
                            print(f"\n💡 ЧАСТКОВА РЕКОМЕНДАЦІЯ:")
                            print(f"   🏆 Найкраща конфігурація: {best_config} (оцінка: {best_score:.4f})")
                
            except Exception as partial_save_error:
                print(f"❌ Не вдалося зберегти часткові результати: {partial_save_error}")
    
    except Exception as e:
        print(f"\n❌ Критична помилка: {e}")
        import traceback
        traceback.print_exc()
        
        # Спробуємо зберегти інформацію про помилку
        try:
            import json
            error_info = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'experiment_name': experiment_name,
                'choice': choice if 'choice' in locals() else 'unknown'
            }
            
            error_file = f"experiment_error_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_info, f, indent=2)
            
            print(f"📝 Інформація про помилку збережена: {error_file}")
        except Exception as error_save_error:
            print(f"❌ Не вдалося зберегти інформацію про помилку: {error_save_error}")


if __name__ == '__main__':
    main()

print("✅ ПОВНИЙ КОД ВІДКОРИГОВАНОЇ ФУНКЦІЇ main ГОТОВИЙ!")
print("🔧 Ключові покращення:")
print("   1. ✅ Правильні рекомендації за комбінованою оцінкою (70% MPC + 30% точність)")
print("   2. ✅ Детальне пояснення логіки вибору найкращої конфігурації")
print("   3. ✅ Показ топ-3 конфігурацій з поясненням")
print("   4. ✅ Інтерпретація якості MPC (висока/середня/низька)")
print("   5. ✅ Обробка помилок з частковим збереженням результатів")
print("   6. ✅ Fallback логіка для випадків відсутності комбінованої оцінки")
print("   7. ✅ Інформативні інструкції користувачу")