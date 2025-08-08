import numpy as np  
import pandas as pd  
import time  
import json  
import os  
from datetime import datetime  

import traceback  

from typing import Callable, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from collections import deque

from data_gen import StatefulDataGenerator
from model import KernelModel
from objectives import MaxIronMassTrackingObjective
from mpc import MPCController
# from conf_manager import MPCConfigManager
from utils import (
    run_post_simulation_analysis_enhanced,  diagnose_mpc_behavior, diagnose_ekf_detailed
)
from ekf import ExtendedKalmanFilter
from anomaly_detector import SignalAnomalyDetector
from maf import MovingAverageFilter
from benchmark import benchmark_model_training, benchmark_mpc_solve_time
from conf_manager import config_manager
from typing import Optional, List


def run_model_comparison_experiment(  
    hist_df: pd.DataFrame,  
    base_config: str = 'oleksandr_original',  
    n_repeats: int = 5,  
    results_dir: str = 'model_comparison_experiment',  
    save_individual: bool = True  
) -> Dict[str, Dict]:  
    """  
    Запуск експерименту порівняння моделей з повтореннями  
    
    Returns:  
        Dict з результатами всіх експериментів  
    """  
    
    # Конфігурація експерименту  
    models_config = {  
        'KRR_RBF': {  
            'model_type': 'krr',  
            'kernel': 'rbf',  
            'find_optimal_params': True  
        },  
        'SVR_RBF': {  
            'model_type': 'svr',   
            'kernel': 'rbf',  
            'find_optimal_params': True  
        },  
        'GPR_RBF': {  
            'model_type': 'gpr',  
            'kernel': 'rbf',   
            'find_optimal_params': True  
        },  
        'LINEAR_RIDGE': {  
            'model_type': 'linear',  
            'linear_type': 'ridge',  
            'find_optimal_params': True  
        }  
    }  
    
    # Створюємо директорію  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
    full_results_dir = f"{results_dir}_{timestamp}"  
    os.makedirs(full_results_dir, exist_ok=True)  
    
    print("🔬 ПОЧАТОК ЕКСПЕРИМЕНТУ ПОРІВНЯННЯ МОДЕЛЕЙ")  
    print("="*70)  
    print(f"📊 Моделей: {len(models_config)}")  
    print(f"🔄 Повторень: {n_repeats}")  
    print(f"📁 Директорія результатів: {full_results_dir}")  
    print(f"⚡ Загалом запусків: {len(models_config) * n_repeats}")  
    print("="*70)  
    
    all_results = {}  
    experiment_summary = []  
    
    total_runs = len(models_config) * n_repeats  
    current_run = 0  
    
    for model_name, model_params in models_config.items():  
        print(f"\n🧪 МОДЕЛЬ: {model_name}")  
        print(f"   Параметри: {model_params}")  
        
        model_results = {  
            'runs': {},  
            'summary_stats': {},  
            'model_config': model_params  
        }  
        
        run_metrics = []  
        
        for repeat in range(n_repeats):  
            current_run += 1  
            run_id = f"{model_name}_run_{repeat+1}"  
            
            print(f"\n   🎯 Запуск {repeat+1}/{n_repeats} ({current_run}/{total_runs})")  
            
            try:  
                # Змінюємо seed для кожного повтору  
                run_start = time.time()  
                
                results_df, metrics = simulate_mpc(  
                    hist_df,  
                    config=base_config,  
                    config_overrides=model_params,  
                    seed=repeat * 42,  # Різні seeds для повторюваності  
                    run_analysis=False  
                )  
                
                run_time = time.time() - run_start  
                
                # Додаємо метаінформацію  
                results_df['experiment_model'] = model_name  
                results_df['experiment_run'] = repeat + 1  
                results_df['experiment_run_id'] = run_id  
                
                metrics['experiment_info'] = {  
                    'model_name': model_name,  
                    'run_number': repeat + 1,  
                    'run_id': run_id,  
                    'seed_used': repeat * 42,  
                    'run_time_seconds': run_time  
                }  
                
                # Зберігаємо індивідуальні результати  
                if save_individual:  
                    results_df.to_parquet(f"{full_results_dir}/{run_id}_results.parquet")  
                    
                    with open(f"{full_results_dir}/{run_id}_metrics.json", 'w') as f:  
                        json_metrics = {}  
                        for key, value in metrics.items():  
                            try:  
                                json.dumps(value)  # Перевіряємо JSON serializable  
                                json_metrics[key] = value  
                            except:  
                                json_metrics[key] = str(value)  
                        json.dump(json_metrics, f, indent=2)  
                
                # Збираємо метрики для статистики  
                run_summary = {  
                    'run_id': run_id,  
                    'model': model_name,  
                    'run_number': repeat + 1,  
                    'run_time': run_time  
                }  
                
                # Додаємо ключові метрики  
                key_metrics = ['rmse_fe', 'rmse_mass', 'mae_fe', 'mae_mass', 'r2_fe', 'r2_mass']  
                for metric in key_metrics:  
                    if metric in metrics:  
                        run_summary[metric] = metrics[metric]  
                
                run_metrics.append(run_summary)  
                model_results['runs'][run_id] = (results_df, metrics)  
                experiment_summary.append(run_summary)  
                
                print(f"      ✅ Завершено за {run_time:.1f}с")  
                if 'rmse_fe' in metrics:  
                    print(f"      📊 RMSE: Fe={metrics['rmse_fe']:.4f}, Mass={metrics.get('rmse_mass', 'N/A'):.4f}")  
                
            except Exception as e:  
                print(f"      ❌ ПОМИЛКА: {e}")  
                run_summary = {  
                    'run_id': run_id,  
                    'model': model_name,  
                    'run_number': repeat + 1,  
                    'error': str(e)  
                }  
                experiment_summary.append(run_summary)  
                continue  
        
        # Статистика по моделі  
        if run_metrics:  
            metrics_df = pd.DataFrame(run_metrics)  
            
            model_stats = {}  
            for metric in ['rmse_fe', 'rmse_mass', 'r2_fe', 'r2_mass', 'run_time']:  
                if metric in metrics_df.columns:  
                    model_stats[f'{metric}_mean'] = metrics_df[metric].mean()  
                    model_stats[f'{metric}_std'] = metrics_df[metric].std()  
                    model_stats[f'{metric}_min'] = metrics_df[metric].min()  
                    model_stats[f'{metric}_max'] = metrics_df[metric].max()  
            
            model_results['summary_stats'] = model_stats  
            
            print(f"   📈 СТАТИСТИКА {model_name}:")  
            print(f"      RMSE Fe: {model_stats.get('rmse_fe_mean', 0):.4f} ± {model_stats.get('rmse_fe_std', 0):.4f}")  
            print(f"      RMSE Mass: {model_stats.get('rmse_mass_mean', 0):.4f} ± {model_stats.get('rmse_mass_std', 0):.4f}")  
            print(f"      R² Fe: {model_stats.get('r2_fe_mean', 0):.4f} ± {model_stats.get('r2_fe_std', 0):.4f}")  
            print(f"      Час: {model_stats.get('run_time_mean', 0):.1f}с ± {model_stats.get('run_time_std', 0):.1f}с")  
        
        all_results[model_name] = model_results  
    
    # Зберігаємо загальну статистику  
    summary_df = pd.DataFrame(experiment_summary)  
    summary_df.to_csv(f"{full_results_dir}/experiment_summary.csv", index=False)  
    
    # Створюємо порівняльну таблицю  
    comparison_table = create_model_comparison_table(all_results)  
    comparison_table.to_csv(f"{full_results_dir}/model_comparison.csv", index=False)  
    
    print("\n" + "="*70)  
    print("🏁 ЕКСПЕРИМЕНТ ЗАВЕРШЕНО")  
    print(f"📁 Результати збережено в: {full_results_dir}")  
    print("="*70)  
    
    return all_results, comparison_table  


def create_model_comparison_table(all_results: Dict) -> pd.DataFrame:  
    """Створює таблицю порівняння моделей"""  
    
    comparison_data = []  
    
    for model_name, model_data in all_results.items():  
        if 'summary_stats' in model_data and model_data['summary_stats']:  
            stats = model_data['summary_stats']  
            
            row = {  
                'Model': model_name,  
                'RMSE_Fe_Mean': stats.get('rmse_fe_mean', np.nan),  
                'RMSE_Fe_Std': stats.get('rmse_fe_std', np.nan),  
                'RMSE_Mass_Mean': stats.get('rmse_mass_mean', np.nan),  
                'RMSE_Mass_Std': stats.get('rmse_mass_std', np.nan),  
                'R2_Fe_Mean': stats.get('r2_fe_mean', np.nan),  
                'R2_Fe_Std': stats.get('r2_fe_std', np.nan),  
                'R2_Mass_Mean': stats.get('r2_mass_mean', np.nan),  
                'R2_Mass_Std': stats.get('r2_mass_std', np.nan),  
                'Runtime_Mean': stats.get('run_time_mean', np.nan),  
                'Runtime_Std': stats.get('run_time_std', np.nan)  
            }  
            comparison_data.append(row)  
    
    comparison_df = pd.DataFrame(comparison_data)  
    
    # Сортуємо по RMSE Fe (найкращий зверху)  
    if 'RMSE_Fe_Mean' in comparison_df.columns:  
        comparison_df = comparison_df.sort_values('RMSE_Fe_Mean')  
    
    return comparison_df  


def analyze_experiment_results(results_dir: str) -> None:  
    """Детальний аналіз результатів експерименту"""  
    
    print("📊 АНАЛІЗ РЕЗУЛЬТАТІВ ЕКСПЕРИМЕНТУ")  
    print("="*70)  
    
    # Завантажуємо порівняльну таблицю  
    comparison_file = f"{results_dir}/model_comparison.csv"  
    if os.path.exists(comparison_file):  
        comparison_df = pd.read_csv(comparison_file)  
        
        print("🏆 РЕЙТИНГ МОДЕЛЕЙ (по RMSE Fe):")  
        print(comparison_df[['Model', 'RMSE_Fe_Mean', 'RMSE_Fe_Std', 'R2_Fe_Mean']].round(4))  
        
        print("\n📈 СТАТИСТИЧНА ЗНАЧУЩІСТЬ:")  
        # Простий аналіз статистичної значущості  
        best_model = comparison_df.iloc[0]  
        print(f"🥇 Найкращий: {best_model['Model']}")  
        print(f"   RMSE Fe: {best_model['RMSE_Fe_Mean']:.4f} ± {best_model['RMSE_Fe_Std']:.4f}")  
        print(f"   R² Fe: {best_model['R2_Fe_Mean']:.4f} ± {best_model['R2_Fe_Std']:.4f}")  
        
        if len(comparison_df) > 1:  
            second_model = comparison_df.iloc[1]  
            rmse_diff = second_model['RMSE_Fe_Mean'] - best_model['RMSE_Fe_Mean']  
            combined_std = np.sqrt(best_model['RMSE_Fe_Std']**2 + second_model['RMSE_Fe_Std']**2)  
            
            print(f"\n🥈 Другий: {second_model['Model']}")  
            print(f"   Різниця RMSE: +{rmse_diff:.4f}")  
            print(f"   Статистична значущість: {'Так' if rmse_diff > 2*combined_std else 'Сумнівно'}")  
    
    # Завантажуємо детальну статистику  
    summary_file = f"{results_dir}/experiment_summary.csv"  
    if os.path.exists(summary_file):  
        summary_df = pd.read_csv(summary_file)  
        
        print(f"\n📋 ДЕТАЛІ ЕКСПЕРИМЕНТУ:")  
        print(f"   Загалом запусків: {len(summary_df)}")  
        print(f"   Успішних: {len(summary_df[~summary_df.get('error', pd.Series()).notna()])}")  
        
        # Статистика по моделях  
        if 'model' in summary_df.columns:  
            model_stats = summary_df.groupby('model').agg({  
                'rmse_fe': ['count', 'mean', 'std'],  
                'run_time': ['mean', 'std']  
            }).round(4)  
            print(f"\n📊 ДЕТАЛЬНА СТАТИСТИКА ПО МОДЕЛЯХ:")  
            print(model_stats)  


# 🚀 ЗАПУСК ЕКСПЕРИМЕНТУ  
def main_experiment():  
    """Головна функція експерименту"""  
    
    # Завантажуємо дані  
    try:  
        hist_df = pd.read_parquet('processed.parquet')  
        print(f"✅ Дані завантажено: {hist_df.shape}")  
    except FileNotFoundError:  
        try:  
            hist_df = pd.read_parquet('/content/KModel/src/processed.parquet')  
            print(f"✅ Дані завантажено з /content/: {hist_df.shape}")  
        except Exception as e:  
            print(f"❌ Помилка завантаження даних: {e}")  
            return  
    
    # Callback для прогресу (показуємо тільки ключові моменти)  
    def progress_callback(step, total, msg):  
        if step % 100 == 0 or step == total or 'завершен' in msg.lower():  
            print(f"      [{step}/{total}] {msg}")  
    
    # Запускаємо експеримент  
    print("🚀 ПОЧИНАЄМО ЕКСПЕРИМЕНТ...")  
    start_time = time.time()  
    
    all_results, comparison_table = run_model_comparison_experiment(  
        hist_df,  
        base_config='oleksandr_original',  
        n_repeats=5,  
        results_dir='model_comparison_experiment'  
    )  
    
    total_time = time.time() - start_time  
    
    print(f"\n⏱️ ЗАГАЛЬНИЙ ЧАС ЕКСПЕРИМЕНТУ: {total_time/60:.1f} хв")  
    print(f"📊 РЕЗУЛЬТАТИ:")  
    print(comparison_table.round(4))  
    
    # Визначаємо директорію результатів  
    # Визначаємо директорію результатів  
    results_dirs = [d for d in os.listdir('.') if d.startswith('model_comparison_experiment_')]  
    if results_dirs:  
        latest_dir = max(results_dirs)  # Останній за алфавітом (найновіший timestamp)  
        print(f"\n🔍 Детальний аналіз результатів...")  
        analyze_experiment_results(latest_dir)  
        
        # Створюємо візуалізацію результатів  
        create_experiment_visualizations(latest_dir)  
    
    return all_results, comparison_table  


def create_experiment_visualizations(results_dir: str) -> None:  
    """Створює графіки для аналізу експерименту"""  
    
    try:  
        import matplotlib.pyplot as plt  
        import seaborn as sns  
        
        print("\n📈 СТВОРЕННЯ ВІЗУАЛІЗАЦІЙ...")  
        
        # Завантажуємо дані  
        comparison_df = pd.read_csv(f"{results_dir}/model_comparison.csv")  
        summary_df = pd.read_csv(f"{results_dir}/experiment_summary.csv")  
        
        # Налаштування стилю  
        plt.style.use('default')  
        sns.set_palette("husl")  
        
        # Створюємо фігуру з підграфіками  
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))  
        fig.suptitle('🔬 Результати експерименту порівняння моделей MPC', fontsize=16, fontweight='bold')  
        
        # 1. Bar plot RMSE Fe  
        ax1 = axes[0, 0]  
        bars1 = ax1.bar(comparison_df['Model'], comparison_df['RMSE_Fe_Mean'],   
                       yerr=comparison_df['RMSE_Fe_Std'], capsize=5, alpha=0.8)  
        ax1.set_title('🎯 RMSE Fe (нижче = краще)', fontweight='bold')  
        ax1.set_ylabel('RMSE Fe')  
        ax1.tick_params(axis='x', rotation=45)  
        ax1.grid(True, alpha=0.3)  
        
        # Підсвічуємо найкращий результат  
        min_idx = comparison_df['RMSE_Fe_Mean'].idxmin()  
        bars1[min_idx].set_color('gold')  
        bars1[min_idx].set_edgecolor('orange')  
        bars1[min_idx].set_linewidth(2)  
        
        # 2. Bar plot RMSE Mass  
        ax2 = axes[0, 1]  
        bars2 = ax2.bar(comparison_df['Model'], comparison_df['RMSE_Mass_Mean'],  
                       yerr=comparison_df['RMSE_Mass_Std'], capsize=5, alpha=0.8)  
        ax2.set_title('⚖️ RMSE Mass (нижче = краще)', fontweight='bold')  
        ax2.set_ylabel('RMSE Mass')  
        ax2.tick_params(axis='x', rotation=45)  
        ax2.grid(True, alpha=0.3)  
        
        # 3. R² Fe  
        ax3 = axes[0, 2]  
        bars3 = ax3.bar(comparison_df['Model'], comparison_df['R2_Fe_Mean'],  
                       yerr=comparison_df['R2_Fe_Std'], capsize=5, alpha=0.8)  
        ax3.set_title('📈 R² Fe (вище = краще)', fontweight='bold')  
        ax3.set_ylabel('R² Fe')  
        ax3.tick_params(axis='x', rotation=45)  
        ax3.grid(True, alpha=0.3)  
        ax3.set_ylim(0, 1)  
        
        # Підсвічуємо найкращий R²  
        max_r2_idx = comparison_df['R2_Fe_Mean'].idxmax()  
        bars3[max_r2_idx].set_color('gold')  
        bars3[max_r2_idx].set_edgecolor('orange')  
        bars3[max_r2_idx].set_linewidth(2)  
        
        # 4. Box plot RMSE по моделях (якщо є детальні дані)  
        ax4 = axes[1, 0]  
        if 'rmse_fe' in summary_df.columns and 'model' in summary_df.columns:  
            summary_df.boxplot(column='rmse_fe', by='model', ax=ax4)  
            ax4.set_title('📦 Розподіл RMSE Fe по моделях', fontweight='bold')  
            ax4.set_xlabel('Модель')  
            ax4.set_ylabel('RMSE Fe')  
        else:  
            ax4.text(0.5, 0.5, 'Детальні дані\nнедоступні', ha='center', va='center',   
                    transform=ax4.transAxes, fontsize=12)  
            ax4.set_title('📦 Розподіл RMSE Fe')  
        
        # 5. Час виконання  
        ax5 = axes[1, 1]  
        bars5 = ax5.bar(comparison_df['Model'], comparison_df['Runtime_Mean'],  
                       yerr=comparison_df['Runtime_Std'], capsize=5, alpha=0.8, color='lightcoral')  
        ax5.set_title('⏱️ Час виконання (сек)', fontweight='bold')  
        ax5.set_ylabel('Час (сек)')  
        ax5.tick_params(axis='x', rotation=45)  
        ax5.grid(True, alpha=0.3)  
        
        # 6. Scatter RMSE vs R²  
        ax6 = axes[1, 2]  
        scatter = ax6.scatter(comparison_df['RMSE_Fe_Mean'], comparison_df['R2_Fe_Mean'],  
                             s=100, alpha=0.7, c=range(len(comparison_df)), cmap='viridis')  
        ax6.set_xlabel('RMSE Fe')  
        ax6.set_ylabel('R² Fe')  
        ax6.set_title('🎯 RMSE vs R² (ліво-верх = краще)', fontweight='bold')  
        ax6.grid(True, alpha=0.3)  
        
        # Додаємо підписи точок  
        for i, model in enumerate(comparison_df['Model']):  
            ax6.annotate(model,   
                        (comparison_df.iloc[i]['RMSE_Fe_Mean'], comparison_df.iloc[i]['R2_Fe_Mean']),  
                        xytext=(5, 5), textcoords='offset points', fontsize=8)  
        
        plt.tight_layout()  
        plt.savefig(f"{results_dir}/experiment_analysis.png", dpi=300, bbox_inches='tight')  
        plt.savefig(f"{results_dir}/experiment_analysis.pdf", bbox_inches='tight')  
        
        print(f"   ✅ Графіки збережено: {results_dir}/experiment_analysis.png")  
        
        # Додатковий графік: детальний аналіз кожної моделі  
        if 'rmse_fe' in summary_df.columns:  
            create_detailed_model_analysis(summary_df, results_dir)  
        
        plt.show()  
        
    except ImportError:  
        print("   ⚠️ matplotlib/seaborn недоступні для візуалізації")  
    except Exception as e:  
        print(f"   ❌ Помилка створення графіків: {e}")  


def create_detailed_model_analysis(summary_df: pd.DataFrame, results_dir: str) -> None:  
    """Детальний аналіз результатів по моделях"""  
    
    try:  
        import matplotlib.pyplot as plt  
        import seaborn as sns  
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))  
        fig.suptitle('🔍 Детальний аналіз стабільності моделей', fontsize=14, fontweight='bold')  
        
        # 1. Violin plot RMSE Fe  
        ax1 = axes[0, 0]  
        if len(summary_df['model'].unique()) > 1:  
            sns.violinplot(data=summary_df, x='model', y='rmse_fe', ax=ax1)  
            ax1.set_title('🎻 Розподіл RMSE Fe')  
            ax1.tick_params(axis='x', rotation=45)  
        
        # 2. Violin plot R² Fe  
        ax2 = axes[0, 1]  
        if 'r2_fe' in summary_df.columns:  
            sns.violinplot(data=summary_df, x='model', y='r2_fe', ax=ax2)  
            ax2.set_title('🎻 Розподіл R² Fe')  
            ax2.tick_params(axis='x', rotation=45)  
        
        # 3. Кореляція метрик  
        ax3 = axes[1, 0]  
        if 'r2_fe' in summary_df.columns:  
            ax3.scatter(summary_df['rmse_fe'], summary_df['r2_fe'], alpha=0.6)  
            ax3.set_xlabel('RMSE Fe')  
            ax3.set_ylabel('R² Fe')  
            ax3.set_title('📊 Кореляція RMSE vs R²')  
            ax3.grid(True, alpha=0.3)  
        
        # 4. Стабільність по запускам  
        ax4 = axes[1, 1]  
        if 'run_number' in summary_df.columns:  
            for model in summary_df['model'].unique():  
                model_data = summary_df[summary_df['model'] == model]  
                ax4.plot(model_data['run_number'], model_data['rmse_fe'],   
                        'o-', label=model, alpha=0.7)  
            ax4.set_xlabel('Номер запуску')  
            ax4.set_ylabel('RMSE Fe')  
            ax4.set_title('📈 Стабільність по запускам')  
            ax4.legend()  
            ax4.grid(True, alpha=0.3)  
        
        plt.tight_layout()  
        plt.savefig(f"{results_dir}/detailed_analysis.png", dpi=300, bbox_inches='tight')  
        print(f"   ✅ Детальний аналіз збережено: {results_dir}/detailed_analysis.png")  
        
    except Exception as e:  
        print(f"   ❌ Помилка детального аналізу: {e}")  


def statistical_significance_test(results_dir: str) -> None:  
    """Статистичний тест значущості різниць між моделями"""  
    
    try:  
        from scipy import stats  
        
        print("\n🧮 СТАТИСТИЧНИЙ АНАЛІЗ ЗНАЧУЩОСТІ")  
        print("="*50)  
        
        summary_df = pd.read_csv(f"{results_dir}/experiment_summary.csv")  
        
        if 'rmse_fe' not in summary_df.columns or 'model' not in summary_df.columns:  
            print("❌ Недостатньо даних для статистичного тесту")  
            return  
        
        models = summary_df['model'].unique()  
        if len(models) < 2:  
            print("❌ Потрібно принаймні 2 моделі для порівняння")  
            return  
        
        # Парні t-тести між моделями  
        print("🔬 ПАРНІ T-ТЕСТИ (RMSE Fe):")  
        results_matrix = []  
        
        for i, model1 in enumerate(models):  
            row = []  
            for j, model2 in enumerate(models):  
                if i == j:  
                    row.append("-")  
                else:  
                    data1 = summary_df[summary_df['model'] == model1]['rmse_fe']  
                    data2 = summary_df[summary_df['model'] == model2]['rmse_fe']  
                    
                    if len(data1) >= 2 and len(data2) >= 2:  
                        t_stat, p_value = stats.ttest_ind(data1, data2)  
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"  
                        row.append(f"{p_value:.4f} {significance}")  
                    else:  
                        row.append("N/A")  
            results_matrix.append(row)  
        
        # Створюємо таблицю результатів  
        results_df = pd.DataFrame(results_matrix, index=models, columns=models)  
        print(results_df)  
        
        print("\n📊 Позначення: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")  
        
        # Зберігаємо статистичний аналіз  
        results_df.to_csv(f"{results_dir}/statistical_analysis.csv")  
        
    except ImportError:  
        print("⚠️ scipy недоступна для статистичного аналізу")  
    except Exception as e:  
        print(f"❌ Помилка статистичного аналізу: {e}")  


# 🎯 ГОЛОВНА ФУНКЦІЯ ЗАПУСКУ  
if __name__ == '__main__':  
    print("🚀 ЗАПУСК ЕКСПЕРИМЕНТУ ПОРІВНЯННЯ МОДЕЛЕЙ MPC")  
    print("="*70)  
    
    # Запускаємо експеримент  
    try:  
        all_results, comparison_table = main_experiment()  
        
        # Знаходимо директорію з результатами  
        results_dirs = [d for d in os.listdir('.') if d.startswith('model_comparison_experiment_')]  
        if results_dirs:  
            latest_dir = max(results_dirs)  
            
            print(f"\n📊 ФІНАЛЬНІ РЕЗУЛЬТАТИ:")  
            print("="*70)  
            
            # Показуємо порівняльну таблицю  
            print("🏆 РЕЙТИНГ МОДЕЛЕЙ:")  
            print(comparison_table[['Model', 'RMSE_Fe_Mean', 'RMSE_Fe_Std', 'R2_Fe_Mean', 'Runtime_Mean']].round(4))  
            
            # Статистичний аналіз  
            statistical_significance_test(latest_dir)  
            
            # Рекомендації  
            print(f"\n💡 РЕКОМЕНДАЦІЇ:")  
            best_model = comparison_table.iloc[0]['Model']  
            best_rmse = comparison_table.iloc[0]['RMSE_Fe_Mean']  
            best_r2 = comparison_table.iloc[0]['R2_Fe_Mean']  
            
            print(f"🥇 Найкращий результат: {best_model}")  
            print(f"   RMSE Fe: {best_rmse:.4f}")  
            print(f"   R² Fe: {best_r2:.4f}")  
            
            if len(comparison_table) > 1:  
                second_best = comparison_table.iloc[1]  
                improvement = ((second_best['RMSE_Fe_Mean'] - best_rmse) / second_best['RMSE_Fe_Mean'] * 100)  
                print(f"   Покращення відносно другого місця: {improvement:.2f}%")  
            
            print(f"\n📁 Всі результати збережено в: {latest_dir}")  
            print(f"📊 Файли:")  
            print(f"   - model_comparison.csv (порівняльна таблиця)")  
            print(f"   - experiment_summary.csv (детальні результати)")  
            print(f"   - experiment_analysis.png (графіки)")  
            print(f"   - statistical_analysis.csv (статистичний аналіз)")  
            print(f"   - individual *_results.parquet (результати кожного запуску)")  
            
    except Exception as e:  
        print(f"❌ КРИТИЧНА ПОМИЛКА: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n🎉 ЕКСПЕРИМЕНТ ЗАВЕРШЕНО!")
    print("="*70)


# 🔧 ДОДАТКОВІ УТИЛІТИ ДЛЯ АНАЛІЗУ РЕЗУЛЬТАТІВ

def load_and_compare_experiments(experiment_dirs: List[str]) -> pd.DataFrame:
    """Порівнює результати кількох експериментів"""
    
    print("🔍 ПОРІВНЯННЯ ЕКСПЕРИМЕНТІВ")
    print("="*50)
    
    all_comparisons = []
    
    for exp_dir in experiment_dirs:
        try:
            comparison_file = f"{exp_dir}/model_comparison.csv"
            if os.path.exists(comparison_file):
                df = pd.read_csv(comparison_file)
                df['Experiment'] = exp_dir
                all_comparisons.append(df)
                print(f"✅ Завантажено: {exp_dir}")
            else:
                print(f"❌ Не знайдено: {comparison_file}")
        except Exception as e:
            print(f"❌ Помилка завантаження {exp_dir}: {e}")
    
    if all_comparisons:
        combined_df = pd.concat(all_comparisons, ignore_index=True)
        
        # Створюємо зведену таблицю
        pivot_table = combined_df.pivot_table(
            index='Model',
            columns='Experiment', 
            values='RMSE_Fe_Mean',
            aggfunc='mean'
        )
        
        print(f"\n📊 ЗВЕДЕНА ТАБЛИЦЯ RMSE Fe:")
        print(pivot_table.round(4))
        
        return combined_df
    else:
        print("❌ Немає даних для порівняння")
        return pd.DataFrame()


def generate_experiment_report(results_dir: str, output_file: str = None) -> str:
    """Генерує детальний звіт про експеримент"""
    
    if output_file is None:
        output_file = f"{results_dir}/experiment_report.md"
    
    try:
        comparison_df = pd.read_csv(f"{results_dir}/model_comparison.csv")
        summary_df = pd.read_csv(f"{results_dir}/experiment_summary.csv")
        
        report = f"""# 🔬 Звіт про експеримент порівняння моделей MPC

## 📊 Загальна інформація
- **Дата експерименту:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Кількість моделей:** {len(comparison_df)}
- **Повторень на модель:** {summary_df['model'].value_counts().iloc[0] if len(summary_df) > 0 else 'N/A'}
- **Загалом запусків:** {len(summary_df)}

## 🏆 Результати

### Рейтинг моделей (по RMSE Fe):
"""
        
        for idx, row in comparison_df.iterrows():
            rank = idx + 1
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            
            report += f"""
{medal} **{row['Model']}**
- RMSE Fe: {row['RMSE_Fe_Mean']:.4f} ± {row['RMSE_Fe_Std']:.4f}
- RMSE Mass: {row['RMSE_Mass_Mean']:.4f} ± {row['RMSE_Mass_Std']:.4f}
- R² Fe: {row['R2_Fe_Mean']:.4f} ± {row['R2_Fe_Std']:.4f}
- R² Mass: {row['R2_Mass_Mean']:.4f} ± {row['R2_Mass_Std']:.4f}
- Час виконання: {row['Runtime_Mean']:.1f}с ± {row['Runtime_Std']:.1f}с
"""
        
        # Аналіз стабільності
        report += f"""
## 📈 Аналіз стабільності

### Коефіцієнти варіації (CV = std/mean):
"""
        
        for idx, row in comparison_df.iterrows():
            cv_rmse = (row['RMSE_Fe_Std'] / row['RMSE_Fe_Mean']) * 100
            cv_r2 = (row['R2_Fe_Std'] / row['R2_Fe_Mean']) * 100 if row['R2_Fe_Mean'] > 0 else float('inf')
            
            stability = "Дуже стабільна" if cv_rmse < 5 else "Стабільна" if cv_rmse < 10 else "Нестабільна"
            
            report += f"- **{row['Model']}**: CV_RMSE = {cv_rmse:.2f}% ({stability})\n"
        
        # Рекомендації
        best_model = comparison_df.iloc[0]
        report += f"""
## 💡 Рекомендації

### 🎯 Найкращий результат: {best_model['Model']}
- Найнижчий RMSE Fe: {best_model['RMSE_Fe_Mean']:.4f}
- Високий R²: {best_model['R2_Fe_Mean']:.4f}
- {'Швидкий' if best_model['Runtime_Mean'] < 60 else 'Повільний'} час виконання: {best_model['Runtime_Mean']:.1f}с

### 📊 Порівняння з іншими моделями:
"""
        
        if len(comparison_df) > 1:
            for idx in range(1, min(4, len(comparison_df))):  # Топ-3 після найкращої
                model = comparison_df.iloc[idx]
                improvement = ((model['RMSE_Fe_Mean'] - best_model['RMSE_Fe_Mean']) / model['RMSE_Fe_Mean'] * 100)
                report += f"- Краще за {model['Model']} на {improvement:.2f}%\n"
        
        # Технічні деталі
        report += f"""
## 🔧 Технічні деталі

### Конфігурація експерименту:
- Базова конфігурація: `oleksandr_original`
- Seeds використані: 0, 42, 84, 126, 168 (для відтворюваності)
- Аналіз відключений для швидкості

### Файли результатів:
- `model_comparison.csv` - порівняльна таблиця
- `experiment_summary.csv` - детальні результати всіх запусків
- `experiment_analysis.png` - візуалізація результатів
- `statistical_analysis.csv` - статистичний аналіз значущості
- `*_results.parquet` - результати окремих симуляцій
- `*_metrics.json` - метрики окремих симуляцій

## 📝 Висновки

1. **Найкращий вибір:** {best_model['Model']} показує найкращі результати по точності
2. **Стабільність:** Всі моделі показують {'стабільні' if comparison_df['RMSE_Fe_Std'].max() < 0.1 else 'варіативні'} результати
3. **Швидкодія:** {'GPR та SVR' if any('GPR' in m or 'SVR' in m for m in comparison_df['Model']) else 'Лінійні моделі'} потребують більше часу для навчання
4. **Рекомендація:** Використовувати {best_model['Model']} для продакшн системи

---
*Звіт згенерований автоматично {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        # Зберігаємо звіт
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Звіт збережено: {output_file}")
        return report
        
    except Exception as e:
        print(f"❌ Помилка генерації звіту: {e}")
        return ""


# 🎯 ШВИДКИЙ ЗАПУСК (ДЛЯ ТЕСТУВАННЯ)
def quick_test_experiment():
    """Швидкий тест експерименту з меншою кількістю даних"""
    
    print("⚡ ШВИДКИЙ ТЕСТ ЕКСПЕРИМЕНТУ")
    
    try:
        hist_df = pd.read_parquet('processed.parquet')
    except:
        try:
            hist_df = pd.read_parquet('/content/KModel/src/processed.parquet')
        except Exception as e:
            print(f"❌ Не можу завантажити дані: {e}")
            return
    
    # Тестуємо тільки 2 моделі по 2 рази з меншими даними
    models_config = {
        'KRR_RBF_TEST': {
            'model_type': 'krr',
            'kernel': 'rbf',
            'N_data': 2000,
            'control_pts': 300,
            'find_optimal_params': False
        },
        'LINEAR_TEST': {
            'model_type': 'linear',
            'linear_type': 'ridge',
            'N_data': 2000,
            'control_pts': 300,
            'find_optimal_params': False
        }
    }
    
    results = {}
    for model_name, params in models_config.items():
        print(f"\n🧪 Тестуємо {model_name}...")
        
        try:
            results_df, metrics = simulate_mpc(
                hist_df,
                config='oleksandr_original',
                config_overrides=params,
                run_analysis=False
            )
            
            results[model_name] = {
                'rmse_fe': metrics.get('rmse_fe', 0),
                'rmse_mass': metrics.get('rmse_mass', 0),
                'r2_fe': metrics.get('r2_fe', 0)
            }
            
            print(f"   ✅ RMSE Fe: {metrics.get('rmse_fe', 0):.4f}")
            
        except Exception as e:
            print(f"   ❌ Помилка: {e}")
    
    print(f"\n📊 РЕЗУЛЬТАТИ ШВИДКОГО ТЕСТУ:")
    for model, metrics in results.items():
        print(f"   {model}: RMSE_Fe={metrics['rmse_fe']:.4f}, R²={metrics['r2_fe']:.4f}")
    
    return results


# 🎯 ТОЧКА ВХОДУ
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        # Швидкий тест
        quick_test_experiment()
    else:
        # Повний експеримент
        main_experiment()

print("✅ Код експерименту готовий!")
print("\n🚀 ДЛЯ ЗАПУСКУ:")
print("   python experiment.py          # Повний експеримент (5 моделей × 5 повторів)")
print("   python experiment.py quick    # Швидкий тест (2 моделі × 1 повтор)")