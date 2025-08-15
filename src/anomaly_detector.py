# anomaly_detector.py - Покращені існуючі класи

import numpy as np
from collections import deque
from typing import Deque, Dict, Iterable, Optional, Tuple

class SignalAnomalyDetector:
    """
    ПОКРАЩЕНИЙ робастний фільтр для одновимірного сигналу.
    Зберігає всі існуючі методи, але з покращеною внутрішньою логікою.
    """
    def __init__(self,
                 window: int = 21,               
                 spike_z: float = 4.0,           
                 drop_rel: float = 0.20,         
                 freeze_len: int = 5,            
                 eps: float = 1e-9,
                 enabled: bool = True,
                 # ✅ НОВІ ПАРАМЕТРИ (опціональні, зворотно сумісні)
                 signal_type: str = 'auto',      # 'auto', 'composition', 'flow'
                 adaptive_sensitivity: bool = True,
                 correlation_check: bool = True,
                 quality_monitoring: bool = True):
        
        if window % 2 == 0:
            window += 1
        self.window = window
        self.spike_z = spike_z
        self.drop_rel = drop_rel
        self.freeze_len = freeze_len
        self.eps = eps
        self.enabled = enabled
        
        # ✅ НОВІ АТРИБУТИ для покращеної функціональності
        self.signal_type = signal_type
        self.adaptive_sensitivity = adaptive_sensitivity
        self.correlation_check = correlation_check
        self.quality_monitoring = quality_monitoring

        # Існуючі атрибути
        self.buf: Deque[float] = deque(maxlen=window)
        self.last_good: float | None = None
        self.freeze_cnt = 0
        
        # ✅ НОВІ АТРИБУТИ для покращень
        self.clean_buf: Deque[float] = deque(maxlen=window)
        self.current_spike_z = spike_z
        self.current_drop_rel = drop_rel
        
        # Автоматичне визначення типу сигналу
        self._auto_signal_type_detector = {'values': [], 'samples_needed': 50}
        
        # Статистика якості
        self.quality_metrics = {
            'total_processed': 0,
            'spikes_detected': 0,
            'drops_detected': 0,
            'freezes_detected': 0,
            'corrections_made': 0,
            'false_positives_estimated': 0
        } if quality_monitoring else None

    def _robust_stats(self) -> tuple[float, float]:
        """ПОКРАЩЕНА версія робастних статистик."""
        if len(self.buf) < 3:
            return 0.0, 1.0
            
        arr = np.asarray(self.buf)
        
        # ✅ ПОКРАЩЕННЯ: Використовуємо тримедіан для більшої стабільності
        if len(arr) >= 5:
            q1, median, q3 = np.percentile(arr, [25, 50, 75])
            tri_median = (q1 + 2*median + q3) / 4
        else:
            tri_median = np.median(arr)
        
        # ✅ ПОКРАЩЕННЯ: Покращена оцінка MAD
        mad = np.median(np.abs(arr - tri_median))
        if mad < self.eps:
            mad = np.std(arr) * 0.6745  # Normalized std як fallback
            
        return tri_median, max(mad, self.eps)

    def _auto_detect_signal_type(self, x: float):
        """Автоматичне визначення типу сигналу на основі характеристик."""
        if self.signal_type != 'auto':
            return
            
        detector = self._auto_signal_type_detector
        detector['values'].append(x)
        
        if len(detector['values']) >= detector['samples_needed']:
            values = np.array(detector['values'])
            
            # Аналіз характеристик сигналу
            mean_val = np.mean(values)
            cv = np.std(values) / (np.abs(mean_val) + self.eps)
            
            # Евристики для визначення типу
            if 20 <= mean_val <= 80 and cv < 0.3:  # Схоже на відсоток складу
                self.signal_type = 'composition'
                self.current_spike_z = self.spike_z * 0.7  # Більш чутливий
                self.current_drop_rel = self.drop_rel * 0.7
            elif mean_val > 50 and cv > 0.2:  # Схоже на потік
                self.signal_type = 'flow'
                self.current_spike_z = self.spike_z * 1.2  # Менш чутливий
                self.current_drop_rel = self.drop_rel * 1.2
            else:
                self.signal_type = 'process'  # Загальний процесний сигнал
                
            # Очищаємо детектор після визначення
            detector['values'].clear()

    def _update_adaptive_thresholds(self):
        """НОВА функція адаптивного оновлення порогів."""
        if not self.adaptive_sensitivity or not self.quality_metrics:
            return
            
        if self.quality_metrics['total_processed'] < 20:
            return
            
        # Розрахунок частоти корекцій
        correction_rate = self.quality_metrics['corrections_made'] / self.quality_metrics['total_processed']
        
        # Адаптація порогів
        if correction_rate > 0.3:  # Занадто багато корекцій
            self.current_spike_z = min(self.current_spike_z * 1.1, self.spike_z * 2)
            self.current_drop_rel = min(self.current_drop_rel * 1.1, self.drop_rel * 2)
        elif correction_rate < 0.05:  # Замало корекцій
            self.current_spike_z = max(self.current_spike_z * 0.95, self.spike_z * 0.5)
            self.current_drop_rel = max(self.current_drop_rel * 0.95, self.drop_rel * 0.5)

    def _validate_correction(self, original: float, corrected: float) -> bool:
        """НОВА функція валідації корекції."""
        if len(self.clean_buf) < 3:
            return True
            
        # Перевіряємо тренд
        recent_clean = list(self.clean_buf)[-3:]
        if len(recent_clean) >= 2:
            trend = np.diff(recent_clean)
            if len(trend) >= 1:
                expected_next = recent_clean[-1] + np.mean(trend)
                correction_error = abs(corrected - expected_next)
                original_error = abs(original - expected_next)
                
                # Корекція має покращувати відповідність тренду
                return correction_error <= original_error * 1.3
                
        return True

    def update(self, x: float, correlation_signal: Optional[float] = None) -> float:
        """
        ПОКРАЩЕНА версія update з збереженням оригінальної сигнатури.
        
        ✅ Новий опціональний параметр correlation_signal для перехресної валідації
        """
        if not self.enabled:
            self.last_good = x
            return x

        # ✅ НОВЕ: Оновлення статистики
        if self.quality_metrics:
            self.quality_metrics['total_processed'] += 1

        # ✅ НОВЕ: Автовизначення типу сигналу
        self._auto_detect_signal_type(x)

        self.buf.append(x)
        if len(self.buf) < 3:
            self.last_good = x
            self.clean_buf.append(x)
            return x

        med, mad = self._robust_stats()
        x_corrected = x
        correction_made = False

        # 1. freeze: n разів підряд ≈ однаково (покращена логіка)
        if len(self.buf) >= self.freeze_len:
            recent_range = np.ptp(list(self.buf)[-self.freeze_len:])
            if recent_range < self.eps:
                self.freeze_cnt += 1
            else:
                self.freeze_cnt = 0

            if self.freeze_cnt >= self.freeze_len:
                if self.last_good is not None:
                    # ✅ ПОКРАЩЕННЯ: Додаємо невеликий шум для уникнення точного повторення
                    noise = np.random.normal(0, mad * 0.01) if mad > 0 else 0
                    x_corrected = self.last_good + noise
                    correction_made = True
                    if self.quality_metrics:
                        self.quality_metrics['freezes_detected'] += 1

        # 2. spike (robust-z) з адаптивним порогом
        if not correction_made:
            z = abs(x - med) / mad
            if z > self.current_spike_z:
                x_corrected = med
                correction_made = True
                if self.quality_metrics:
                    self.quality_metrics['spikes_detected'] += 1

        # 3. drop (раптове падіння) з адаптивним порогом
        if not correction_made and abs(med) > self.eps:
            drop_ratio = (med - x) / abs(med)
            if drop_ratio > self.current_drop_rel:
                # ✅ ПОКРАЩЕННЯ: Часткова корекція замість повної
                x_corrected = med - (med * self.current_drop_rel * 0.5)
                correction_made = True
                if self.quality_metrics:
                    self.quality_metrics['drops_detected'] += 1

        # ✅ НОВЕ: Валідація корекції
        if correction_made:
            if not self._validate_correction(x, x_corrected):
                x_corrected = x
                correction_made = False
                if self.quality_metrics:
                    self.quality_metrics['false_positives_estimated'] += 1
            else:
                if self.quality_metrics:
                    self.quality_metrics['corrections_made'] += 1

        # ✅ НОВЕ: Перехресна валідація з корельованим сигналом
        if (correction_made and correlation_signal is not None and 
            self.correlation_check and len(self.buf) >= 5):
            
            # Простий приклад перехресної валідації
            signal_change = abs(x - x_corrected) / (abs(x) + self.eps)
            if signal_change > 0.5:  # Велика корекція
                # Перевіряємо, чи корельований сигнал теж змінився
                if hasattr(self, '_last_correlation_signal'):
                    corr_change = abs(correlation_signal - self._last_correlation_signal) / (abs(self._last_correlation_signal) + self.eps)
                    if corr_change < 0.05:  # Корельований сигнал стабільний
                        # Зменшуємо корекцію
                        x_corrected = x * 0.7 + x_corrected * 0.3
            
            self._last_correlation_signal = correlation_signal

        # ✅ НОВЕ: Оновлення адаптивних порогів
        if self.quality_metrics and self.quality_metrics['total_processed'] % 20 == 0:
            self._update_adaptive_thresholds()

        self.last_good = x_corrected
        self.clean_buf.append(x_corrected)
        
        return x_corrected

    # ✅ НОВІ МЕТОДИ (не порушують існуючий API)
    def get_quality_report(self) -> Dict:
        """Новий метод для отримання звіту про якість роботи."""
        if not self.quality_metrics:
            return {'quality_monitoring': False}
            
        total = max(1, self.quality_metrics['total_processed'])
        return {
            **self.quality_metrics,
            'correction_rate': self.quality_metrics['corrections_made'] / total,
            'spike_rate': self.quality_metrics['spikes_detected'] / total,
            'drop_rate': self.quality_metrics['drops_detected'] / total,
            'freeze_rate': self.quality_metrics['freezes_detected'] / total,
            'false_positive_rate': self.quality_metrics['false_positives_estimated'] / total,
            'current_spike_threshold': self.current_spike_z,
            'current_drop_threshold': self.current_drop_rel,
            'detected_signal_type': self.signal_type,
            'adaptive_mode': self.adaptive_sensitivity
        }

    def reset_quality_metrics(self):
        """Новий метод для скидання статистики якості."""
        if self.quality_metrics:
            self.quality_metrics = {key: 0 for key in self.quality_metrics}

    def get_adaptive_parameters(self) -> Dict:
        """Новий метод для отримання поточних адаптивних параметрів."""
        return {
            'current_spike_z': self.current_spike_z,
            'current_drop_rel': self.current_drop_rel,
            'base_spike_z': self.spike_z,
            'base_drop_rel': self.drop_rel,
            'signal_type': self.signal_type,
            'adaptation_active': self.adaptive_sensitivity
        }


class MultiSignalDetector:
    """
    ПОКРАЩЕНА версія мульти-сигнального детектора.
    Зберігає всі існуючі методи та сигнатури.
    """
    def __init__(self, columns: Iterable[str], **det_kwargs):
        # ✅ ПОКРАЩЕННЯ: Створюємо детектори з покращеною логікою
        self.detectors: Dict[str, SignalAnomalyDetector] = {}
        self.columns = list(columns)
        
        # ✅ НОВЕ: Зберігаємо конфігурацію для аналізу
        self.base_config = det_kwargs
        
        for c in self.columns:
            # ✅ ПОКРАЩЕННЯ: Автоматичне визначення типу сигналу
            signal_config = det_kwargs.copy()
            
            # Евристики для різних типів сигналів
            if 'fe' in c.lower() and 'percent' in c.lower():
                signal_config['signal_type'] = 'composition'
            elif 'flow' in c.lower() or 'mass' in c.lower():
                signal_config['signal_type'] = 'flow'
            else:
                signal_config['signal_type'] = 'auto'
                
            self.detectors[c] = SignalAnomalyDetector(**signal_config)

    def clean_row(self, row: dict) -> dict:
        """
        ПОКРАЩЕНА версія clean_row з перехресною валідацією.
        Зберігає оригінальну сигнатуру.
        """
        corr = {}
        
        # Спочатку обробляємо кожен сигнал
        for c, det in self.detectors.items():
            if c in row:
                # ✅ ПОКРАЩЕННЯ: Передаємо корельовані сигнали
                correlation_signal = None
                if 'fe' in c.lower() and any('flow' in other.lower() for other in row.keys()):
                    # Знаходимо потоковий сигнал для Fe
                    flow_cols = [k for k in row.keys() if 'flow' in k.lower()]
                    if flow_cols:
                        correlation_signal = row[flow_cols[0]]
                elif 'flow' in c.lower() and any('fe' in other.lower() for other in row.keys()):
                    # Знаходимо Fe сигнал для потоку
                    fe_cols = [k for k in row.keys() if 'fe' in k.lower()]
                    if fe_cols:
                        correlation_signal = row[fe_cols[0]]
                
                corr[c] = det.update(row[c], correlation_signal=correlation_signal)
            else:
                corr[c] = 0.0  # Значення за замовчуванням
        
        # ✅ НОВЕ: Пост-обробна перехресна валідація
        corr = self._cross_validate_signals(row, corr)
        
        return corr

    def _cross_validate_signals(self, original: dict, cleaned: dict) -> dict:
        """НОВА функція перехресної валідації між сигналами."""
        
        # Знаходимо пари для валідації
        fe_cols = [c for c in cleaned.keys() if 'fe' in c.lower()]
        flow_cols = [c for c in cleaned.keys() if 'flow' in c.lower()]
        
        if not fe_cols or not flow_cols:
            return cleaned
            
        # Валідуємо найбільш важливі пари
        fe_col = fe_cols[0]
        flow_col = flow_cols[0]
        
        if fe_col in original and flow_col in original:
            # Обчислюємо корекції
            fe_correction = abs(cleaned[fe_col] - original[fe_col]) / (abs(original[fe_col]) + 1e-6)
            flow_correction = abs(cleaned[flow_col] - original[flow_col]) / (abs(original[flow_col]) + 1e-6)
            
            # Якщо одна корекція значно більша за іншу - це підозріло
            if fe_correction > 0.15 and flow_correction < 0.05:
                # Перевіряємо логічність корекції Fe
                if original[flow_col] > 80 and fe_correction > 0.1:  # Високий потік, велика корекція Fe
                    cleaned[fe_col] = original[fe_col] * 0.8 + cleaned[fe_col] * 0.2  # Часткове повернення
                    
            elif flow_correction > 0.2 and fe_correction < 0.05:
                # Перевіряємо логічність корекції потоку
                if original[fe_col] < 35 and cleaned[flow_col] > original[flow_col] * 1.3:  # Низький Fe, збільшення потоку
                    cleaned[flow_col] = original[flow_col] * 0.7 + cleaned[flow_col] * 0.3  # Часткове повернення
        
        return cleaned

    def clean_dataframe(self, df, in_place=False):
        """Існуючий метод без змін сигнатури, але з покращеною логікою."""
        if not in_place:
            df = df.copy()
            
        # ✅ ПОКРАЩЕННЯ: Обробляємо по рядках для перехресної валідації
        for idx in range(len(df)):
            row_dict = {c: df.loc[idx, c] for c in self.detectors.keys() if c in df.columns}
            cleaned_row = self.clean_row(row_dict)
            
            for c in cleaned_row:
                if c in df.columns:
                    df.loc[idx, c] = cleaned_row[c]
                    
        return df

    # ✅ НОВІ МЕТОДИ (не порушують існуючий API)
    def get_comprehensive_report(self) -> Dict:
        """Новий метод для отримання комплексного звіту."""
        reports = {}
        for signal_name, detector in self.detectors.items():
            reports[signal_name] = detector.get_quality_report()
        
        # Загальна статистика
        total_corrections = sum(r.get('corrections_made', 0) for r in reports.values())
        total_processed = sum(r.get('total_processed', 0) for r in reports.values())
        
        return {
            'individual_reports': reports,
            'overall_correction_rate': total_corrections / max(1, total_processed),
            'total_corrections': total_corrections,
            'total_processed': total_processed,
            'signal_types_detected': {name: r.get('detected_signal_type', 'unknown') 
                                    for name, r in reports.items()}
        }

    def reset_all_quality_metrics(self):
        """Новий метод для скидання всієї статистики."""
        for detector in self.detectors.values():
            detector.reset_quality_metrics()

    def get_adaptation_status(self) -> Dict:
        """Новий метод для отримання статусу адаптації."""
        status = {}
        for name, detector in self.detectors.items():
            status[name] = detector.get_adaptive_parameters()
        return status


# ===============================================
# ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ ІНТЕГРАЦІЇ
# ===============================================

def create_enhanced_anomaly_detectors(anomaly_params: dict, columns: list = None) -> MultiSignalDetector:
    """
    Функція для створення покращених детекторів з існуючими параметрами.
    
    ✅ Повністю зворотно сумісна з існуючим кодом
    """
    if columns is None:
        columns = ['feed_fe_percent', 'ore_mass_flow']
    
    # Додаємо покращені параметри до існуючих
    enhanced_params = anomaly_params.copy()
    
    # Автоматично активуємо покращення, якщо не вказано інше
    if 'adaptive_sensitivity' not in enhanced_params:
        enhanced_params['adaptive_sensitivity'] = True
    if 'correlation_check' not in enhanced_params:
        enhanced_params['correlation_check'] = True
    if 'quality_monitoring' not in enhanced_params:
        enhanced_params['quality_monitoring'] = True
    
    return MultiSignalDetector(columns, **enhanced_params)

def monitor_anomaly_quality(detector: MultiSignalDetector, step: int, report_interval: int = 50):
    """
    Функція для моніторингу якості детекції (можна додати в існуючий цикл).
    """
    if step % report_interval != 0 or step == 0:
        return
        
    report = detector.get_comprehensive_report()
    
    print(f"\n📊 ЗВІТ ПРО ДЕТЕКЦІЮ АНОМАЛІЙ (крок {step}):")
    print(f"   🎯 Загальна частота корекцій: {report['overall_correction_rate']:.1%}")
    
    for signal, stats in report['individual_reports'].items():
        if stats.get('quality_monitoring', True):
            print(f"   📈 {signal}:")
            print(f"      • Корекції: {stats['correction_rate']:.1%}")
            print(f"      • Тип сигналу: {stats.get('detected_signal_type', 'auto')}")
            print(f"      • Адаптивний поріг: {stats['current_spike_threshold']:.2f}")
            
            if stats['correction_rate'] > 0.3:
                print(f"      ⚠️  ВИСОКА ЧАСТОТА КОРЕКЦІЙ!")


# ===============================================
# РЕКОМЕНДОВАНІ ПАРАМЕТРИ
# ===============================================

def get_recommended_anomaly_params():
    """
    Рекомендовані параметри для покращеної роботи.
    Можна використовувати замість існуючих.
    """
    return {
        "window": 15,                    # ✅ Зменшено з 25
        "spike_z": 2.5,                  # ✅ Зменшено з 4.0
        "drop_rel": 0.15,                # ✅ Зменшено з 0.30
        "freeze_len": 4,                 # ✅ Зменшено з 5
        "eps": 1e-9,
        "enabled": True,
        # Нові покращені параметри
        "adaptive_sensitivity": True,     # ✅ Адаптивні пороги
        "correlation_check": True,        # ✅ Перехресна валідація
        "quality_monitoring": True        # ✅ Моніторинг якості
    }