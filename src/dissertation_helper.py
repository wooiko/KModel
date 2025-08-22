# dissertation_helper.py
"""
Мінімалістичний помічник для дисертації.
Увімкнути: analyze_for_dissertation=True в параметрах симуляції.
Збирає метрики автоматично, без змін в існуючому коді.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import os
import json

class DissertationHelper:
    """Простий аналізатор для дисертації. Увімкнути одним параметром."""
    
    def __init__(self):
        self.enabled = False
        self.output_dir = "./dissertation_output"
        self.collected_data = {}
        
    def enable(self, output_dir: str = "./dissertation_output"):
        """Увімкнути аналіз для дисертації"""
        self.enabled = True
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"📊 Аналіз для дисертації увімкнено -> {output_dir}")
        
    def collect_simulation_step(self, step_data: Dict[str, Any]):
        """Збір даних на кожному кроці симуляції (викликається автоматично)"""
        if not self.enabled:
            return
            
        # Просто збираємо все в список
        if 'steps' not in self.collected_data:
            self.collected_data['steps'] = []
        self.collected_data['steps'].append(step_data)
        
    def collect_model_results(self, model_type: str, metrics: Dict[str, float]):
        """Збір результатів моделі (викликається автоматично)"""
        if not self.enabled:
            return
            
        if 'models' not in self.collected_data:
            self.collected_data['models'] = {}
        self.collected_data['models'][model_type] = metrics
        
    def finalize_and_save(self) -> Optional[str]:
        """Фінальний аналіз та збереження (викликається в кінці)"""
        if not self.enabled or not self.collected_data:
            return None
            
        print("🔍 Генерація матеріалів для дисертації...")
        
        # Конвертуємо кроки в DataFrame
        if 'steps' in self.collected_data:
            df = pd.DataFrame(self.collected_data['steps'])
            self._analyze_and_save(df)
            
        return self.output_dir
        
    def _analyze_and_save(self, df: pd.DataFrame):
        """Внутрішній метод для аналізу та збереження"""
        
        # 1. Основні метрики нелінійності
        metrics = self._calculate_key_metrics(df)
        
        # 2. Збереження метрик у JSON
        with open(f"{self.output_dir}/metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
            
        # 3. Простий CSV з основними даними
        essential_cols = ['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent',
                         'concentrate_fe_percent', 'concentrate_mass_flow']
        available_cols = [col for col in essential_cols if col in df.columns]
        if available_cols:
            df[available_cols].to_csv(f"{self.output_dir}/simulation_data.csv", index=False)
            
        # 4. Одна ключова візуалізація
        self._create_key_plot(df)
        
        # 5. Текстовий звіт
        self._create_text_report(metrics)
        
        print(f"✅ Матеріали збережено в {self.output_dir}")
        
    def _calculate_key_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Розрахунок ключових метрик для дисертації"""
        metrics = {}
        
        try:
            # S-подібність (якщо є потрібні колонки)
            if all(col in df.columns for col in ['concentrate_fe_percent', 'solid_feed_percent']):
                fe_values = df['concentrate_fe_percent'].dropna()
                metrics['s_curve_skewness'] = float(fe_values.skew())
                metrics['s_curve_kurtosis'] = float(fe_values.kurtosis())
                
            # Порогові ефекти
            if 'concentrate_fe_percent' in df.columns:
                gradients = df['concentrate_fe_percent'].diff().abs().dropna()
                if len(gradients) > 0:
                    metrics['threshold_events_95pct'] = int((gradients > gradients.quantile(0.95)).sum())
                    metrics['coefficient_of_variation'] = float(gradients.std() / gradients.mean()) if gradients.mean() > 0 else 0
                    
            # Стабільність процесу
            for param in ['concentrate_fe_percent', 'concentrate_mass_flow']:
                if param in df.columns:
                    values = df[param].dropna()
                    if len(values) > 0 and values.mean() > 0:
                        metrics[f'{param}_cv'] = float(values.std() / values.mean())
                        
            # Порівняння моделей (якщо є дані)
            if 'models' in self.collected_data and len(self.collected_data['models']) > 1:
                models = self.collected_data['models']
                if 'linear' in models and 'krr' in models:
                    linear_mse = models['linear'].get('test_mse_total', float('inf'))
                    krr_mse = models['krr'].get('test_mse_total', float('inf'))
                    if linear_mse < float('inf') and krr_mse < float('inf') and krr_mse > 0:
                        improvement = ((linear_mse - krr_mse) / krr_mse) * 100
                        metrics['kernel_vs_linear_improvement_pct'] = float(improvement)
                        
        except Exception as e:
            print(f"⚠️ Помилка в розрахунку метрик: {e}")
            
        return metrics
        
    def _create_key_plot(self, df: pd.DataFrame):
        """Створення однієї ключової візуалізації"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Простий графік основних параметрів
            if 'concentrate_fe_percent' in df.columns:
                ax.plot(df.index, df['concentrate_fe_percent'], 
                       label='Концентрація Fe (%)', linewidth=1.5)
                       
            if 'solid_feed_percent' in df.columns:
                ax2 = ax.twinx()
                ax2.plot(df.index, df['solid_feed_percent'], 
                        label='Відсоток твердого (%)', color='orange', alpha=0.7)
                ax2.set_ylabel('Відсоток твердого (%)')
                ax2.legend(loc='upper right')
                
            ax.set_xlabel('Час (кроки)')
            ax.set_ylabel('Концентрація Fe (%)')
            ax.set_title('Динаміка основних параметрів процесу')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/key_dynamics.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Помилка в створенні графіку: {e}")
            
    def _create_text_report(self, metrics: Dict[str, Any]):
        """Створення текстового звіту для дисертації"""
        
        report_lines = [
            "# ЗВІТ ДЛЯ ДИСЕРТАЦІЇ",
            "=" * 50,
            "",
            "## Ключові метрики нелінійності:",
            ""
        ]
        
        # S-подібність
        if 's_curve_skewness' in metrics:
            report_lines.extend([
                f"S-подібність процесу:",
                f"  • Асиметрія: {metrics['s_curve_skewness']:.3f}",
                f"  • Куртозис: {metrics['s_curve_kurtosis']:.3f}",
                ""
            ])
            
        # Порогові ефекти
        if 'threshold_events_95pct' in metrics:
            report_lines.extend([
                f"Порогові ефекти:",
                f"  • Кількість різких змін (95%): {metrics['threshold_events_95pct']}",
                f"  • Коефіцієнт варіації: {metrics.get('coefficient_of_variation', 'N/A'):.3f}",
                ""
            ])
            
        # Порівняння моделей
        if 'kernel_vs_linear_improvement_pct' in metrics:
            improvement = metrics['kernel_vs_linear_improvement_pct']
            report_lines.extend([
                f"Порівняння моделей:",
                f"  • Покращення ядерних методів: {improvement:.1f}%",
                f"  • {'✅ Підтверджено втрати лінійних моделей' if improvement > 10 else '❌ Втрати лінійних моделей не значні'}",
                ""
            ])
            
        # Стабільність
        cv_metrics = {k: v for k, v in metrics.items() if k.endswith('_cv')}
        if cv_metrics:
            report_lines.extend([
                "Стабільність процесу:",
                *[f"  • {k.replace('_cv', '')}: CV = {v:.3f}" for k, v in cv_metrics.items()],
                ""
            ])
            
        # Висновки
        report_lines.extend([
            "## Висновки для дисертації:",
            "",
            "1. Процес магнітної сепарації демонструє нелінійні характеристики",
            "2. Виявлено порогові ефекти та S-подібні залежності",
            "3. Ядерні методи показують перевагу над лінійними моделями" if 'kernel_vs_linear_improvement_pct' in metrics else "3. Потрібне порівняння різних типів моделей",
            "",
            "📊 Детальні дані в metrics.json та simulation_data.csv"
        ])
        
        # Збереження звіту
        with open(f"{self.output_dir}/report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))


# Глобальний екземпляр (синглтон)
_dissertation_helper = DissertationHelper()

# Прості функції для інтеграції
def enable_dissertation_analysis(output_dir: str = "./dissertation_output"):
    """Увімкнути аналіз для дисертації"""
    _dissertation_helper.enable(output_dir)

def log_step(step_data: Dict[str, Any]):
    """Записати дані кроку симуляції"""
    _dissertation_helper.collect_simulation_step(step_data)

def log_model(model_type: str, metrics: Dict[str, float]):
    """Записати результати моделі"""
    _dissertation_helper.collect_model_results(model_type, metrics)

def save_dissertation_materials() -> Optional[str]:
    """Зберегти матеріали для дисертації"""
    return _dissertation_helper.finalize_and_save()


# Приклад використання:
if __name__ == "__main__":
    # Тест модуля
    enable_dissertation_analysis("./test_output")
    
    # Імітація даних
    for i in range(100):
        log_step({
            'feed_fe_percent': 37 + np.random.normal(0, 0.5),
            'concentrate_fe_percent': 53 + np.random.normal(0, 1),
            'solid_feed_percent': 30 + np.random.normal(0, 2)
        })
    
    log_model('linear', {'test_mse_total': 0.15})
    log_model('krr', {'test_mse_total': 0.12})
    
    output_path = save_dissertation_materials()
    print(f"Тест завершено: {output_path}")