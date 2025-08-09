# enhanced_simulator_runner.py - Приклади запуску розширеного симулятора

import pandas as pd
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Імпортуємо розширені функції
from enhanced_sim import (
    simulate_mpc,
    quick_mpc_benchmark, 
    detailed_mpc_analysis,
    compare_mpc_configurations,
    simulate_mpc_with_config_enhanced
)

def load_historical_data() -> pd.DataFrame:
    """Завантажує історичні дані для симуляції"""
    
    # Спробуємо завантажити з різних місць
    possible_paths = [
        'processed.parquet',
        'data/processed.parquet', 
        '/content/KModel/src/processed.parquet',
        '../data/processed.parquet'
    ]
    
    for path in possible_paths:
        try:
            hist_df = pd.read_parquet(path)
            print(f"✅ Дані завантажено з: {path}")
            print(f"   📊 Розмір: {hist_df.shape[0]} рядків, {hist_df.shape[1]} колонок")
            return hist_df
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"⚠️ Помилка завантаження з {path}: {e}")
            continue
    
    raise FileNotFoundError("❌ Не вдалося знайти файл processed.parquet")

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

def example_4_model_comparison():
    """🔄 Приклад 4: Порівняння різних конфігурацій"""
    
    print("\n" + "="*70)
    print("🔄 ПРИКЛАД 4: ПОРІВНЯННЯ КОНФІГУРАЦІЙ MPC")
    print("="*70)
    
    try:
        hist_df = load_historical_data()
        
        # Визначаємо конфігурації для порівняння
        configurations = [
            {
                'name': 'KRR_Conservative',
                'model_type': 'krr',
                'kernel': 'rbf', 
                'Np': 6,
                'Nc': 4,
                'λ_obj': 0.2,  # Більше згладжування
                'w_fe': 5.0,
                'w_mass': 1.0
            },
            {
                'name': 'KRR_Aggressive', 
                'model_type': 'krr',
                'kernel': 'rbf',
                'Np': 8,
                'Nc': 6,
                'λ_obj': 0.05,  # Менше згладжування
                'w_fe': 10.0,   # Більша вага Fe
                'w_mass': 1.5
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
                'Np': 10,  # Можемо дозволити більший горизонт
                'Nc': 8,
                'λ_obj': 0.15,
                'w_fe': 6.0,
                'w_mass': 1.0
            }
        ]
        
        print(f"🔄 Порівнюємо {len(configurations)} конфігурацій...")
        
        # Показуємо конфігурації
        for i, config in enumerate(configurations, 1):
            print(f"   {i}. {config['name']}: {config['model_type']} "
                  f"(Np={config['Np']}, λ={config['λ_obj']})")
        
        start_time = time.time()
        
        comparison_df = compare_mpc_configurations(
            configurations=configurations,
            hist_df=hist_df,
            base_config='oleksandr_original',
            comparison_steps=100  # Коротші тести для швидкості
        )
        
        comparison_time = time.time() - start_time
        
        print(f"\n🔄 РЕЗУЛЬТАТИ ПОРІВНЯННЯ:")
        print(f"   ⏱️ Час порівняння: {comparison_time:.1f} секунд")
        
        # Показуємо результати
        if not comparison_df.empty:
            print(f"\n🏆 РЕЙТИНГ КОНФІГУРАЦІЙ:")
            
            # Сортуємо за якістю або RMSE
            if 'Quality_Score' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('Quality_Score', na_position=True)
                sort_metric = 'Quality_Score'
            elif 'RMSE_Fe' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('RMSE_Fe', na_position=True)
                sort_metric = 'RMSE_Fe'
            else:
                sort_metric = 'Configuration'
            
            for idx, row in comparison_df.iterrows():
                rank = idx + 1
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                
                config_name = row['Configuration']
                rmse_fe = row.get('RMSE_Fe', 'N/A')
                quality = row.get('Quality_Score', 'N/A')
                cycle_time = row.get('Total_Cycle_Time', 'N/A')
                
                print(f"   {medal} {config_name}:")
                if isinstance(rmse_fe, (int, float)):
                    print(f"       RMSE Fe: {rmse_fe:.4f}")
                if isinstance(quality, (int, float)):
                    print(f"       Якість: {quality:.3f}")
                if isinstance(cycle_time, (int, float)):
                    print(f"       Час циклу: {cycle_time*1000:.1f}ms")
            
            # Рекомендації
            print(f"\n💡 РЕКОМЕНДАЦІЇ:")
            best_config = comparison_df.iloc[0]
            print(f"   🏆 Найкраща конфігурація: {best_config['Configuration']}")
            
            if isinstance(best_config.get('RMSE_Fe'), (int, float)):
                if best_config['RMSE_Fe'] < 0.05:
                    print(f"   ✅ Відмінна точність - рекомендується для продакшн")
                else:
                    print(f"   ⚠️ Розгляньте додаткове налаштування")
        
        return comparison_df
        
    except Exception as e:
        print(f"❌ Помилка в прикладі 4: {e}")
        return None

def save_experiment_summary(results: Dict[str, Any]):
    """💾 Зберігає підсумок експерименту"""
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_summary_{timestamp}.json"
    
    # Підготовка даних для збереження
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'experiments_conducted': len(results),
        'results': {}
    }
    
    for exp_name, exp_result in results.items():
        if exp_result is not None:
            if isinstance(exp_result, pd.DataFrame):
                # DataFrame -> dict
                summary['results'][exp_name] = {
                    'type': 'dataframe',
                    'shape': exp_result.shape,
                    'columns': list(exp_result.columns),
                    'data_sample': exp_result.head().to_dict() if not exp_result.empty else {}
                }
            elif isinstance(exp_result, dict):
                # Dict -> відфільтровані дані
                summary['results'][exp_name] = {
                    'type': 'dict',
                    'keys': list(exp_result.keys()),
                    'sample_data': {k: str(v)[:100] for k, v in exp_result.items() if k != 'comprehensive_analysis'}
                }
            else:
                summary['results'][exp_name] = {
                    'type': str(type(exp_result)),
                    'value': str(exp_result)[:200]
                }
        else:
            summary['results'][exp_name] = {
                'type': 'none',
                'status': 'failed'
            }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"💾 Підсумок експерименту збережено: {filename}")
    except Exception as e:
        print(f"⚠️ Помилка збереження підсумку: {e}")

def main():
    """🚀 Головна функція запуску всіх прикладів"""
    
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
        # Можна вибрати конкретний приклад
        choice = input("Оберіть приклад (1-5) або Enter для всіх: ").strip()
        
        if not choice:
            choice = "5"  # Запускаємо всі за замовчуванням
        
        results = {}
        total_start_time = time.time()
        
        if choice in ["1", "5"]:
            print(f"\n{'🚀 ЗАПУСК ПРИКЛАДУ 1' if choice == '1' else '🚀 ПРИКЛАД 1/4'}")
            results['quick_benchmark'] = example_1_quick_benchmark()
        
        if choice in ["2", "5"]:
            print(f"\n{'🔬 ЗАПУСК ПРИКЛАДУ 2' if choice == '2' else '🔬 ПРИКЛАД 2/4'}")
            results['detailed_analysis'] = example_2_detailed_analysis()
        
        if choice in ["3", "5"]:
            print(f"\n{'🎯 ЗАПУСК ПРИКЛАДУ 3' if choice == '3' else '🎯 ПРИКЛАД 3/4'}")
            results['custom_simulation'] = example_3_custom_simulation()
        
        if choice in ["4", "5"]:
            print(f"\n{'🔄 ЗАПУСК ПРИКЛАДУ 4' if choice == '4' else '🔄 ПРИКЛАД 4/4'}")
            results['configuration_comparison'] = example_4_model_comparison()
        
        total_time = time.time() - total_start_time
        
        # Підсумок
        print(f"\n" + "="*70)
        print(f"🎉 ЕКСПЕРИМЕНТИ ЗАВЕРШЕНО")
        print(f"="*70)
        print(f"⏱️ Загальний час: {total_time/60:.1f} хвилин")
        print(f"📊 Проведено експериментів: {len([r for r in results.values() if r is not None])}")
        
        success_count = len([r for r in results.values() if r is not None])
        total_count = len(results)
        print(f"✅ Успішних: {success_count}/{total_count}")
        
        if success_count > 0:
            print(f"\n📁 Згенеровані файли:")
            print(f"   • CSV файли з результатами бенчмарку")
            print(f"   • JSON файли з метриками")
            print(f"   • Конфігурації для відтворення результатів")
        
        # Зберігаємо підсумок
        if results:
            save_experiment_summary(results)
        
        print(f"\n💡 ПОРАДИ:")
        print(f"   • Для продакшн використовуйте конфігурацію з найкращими результатами")
        print(f"   • Регулярно запускайте бенчмарк для моніторингу продуктивності")
        print(f"   • Налаштовуйте параметри MPC на основі рекомендацій")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Експеримент перервано користувачем")
    except Exception as e:
        print(f"\n❌ Критична помилка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()