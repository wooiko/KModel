# anomaly_detector.py

import numpy as np
from collections import deque
from typing import Deque, Dict, Iterable

class SignalAnomalyDetector:
    """
    Простий робастний фільтр для одно­вимірного сигналу.
    • spike  – одиничний викид       → заміняємо на медіану вікна
    • drop   – різке негативне зміщ. → гасяться миттєво (shift up)
    • drift  – повільний тренд       → забираємо лінійний тренд з вікна
    • freeze – «залипання» значення  → дублюємо останнє коректне      
    """
    def __init__(self,
                 window: int = 21,               # довжина ковзного вікна
                 spike_z: float = 4.0,           # поріг robust-z  для spike
                 drop_rel: float = 0.20,         # ≥20 % різкий спад = drop
                 # drift_slope: float = 0.02,      # 2 %/крок ⇒ drift
                 freeze_len: int = 5,            # ≥5 однакових значень
                 eps: float = 1e-9,
                 enabled: bool = True):
        if window % 2 == 0:
            window += 1
        self.window = window
        self.spike_z = spike_z
        self.drop_rel = drop_rel
        # self.drift_slope = drift_slope
        self.freeze_len = freeze_len
        self.eps = eps
        self.enabled = enabled

        self.buf: Deque[float] = deque(maxlen=window)
        self.last_good: float | None = None
        self.freeze_cnt = 0

    # ------------------------------------------------------------ helpers
    def _robust_stats(self) -> tuple[float, float]:
        arr = np.asarray(self.buf)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + self.eps
        return med, mad

    # ------------------------------------------------------------- public
    def update(self, x: float) -> float:
        # Якщо детектор вимкнено, просто повертаємо оригінальне значення.
        # Оновлюємо last_good, щоб при увімкненні детектора він мав актуальне значення.
        if not self.enabled:
            self.last_good = x
            return x

        """
        Приймає чергове значення, повертає «очищене».
        """
        self.buf.append(x)
        if len(self.buf) < 3:          # не вистачає історії
            self.last_good = x
            return x

        med, mad = self._robust_stats()

        # --- 1) freeze: n разів підряд ≈ однаково
        if len(self.buf) >= self.freeze_len and \
           np.ptp(self.buf) < self.eps:
            self.freeze_cnt += 1
        else:
            self.freeze_cnt = 0

        if self.freeze_cnt >= self.freeze_len:
            return self.last_good if self.last_good is not None else med

        # --- 2) spike (robust-z)
        z = abs(x - med) / mad
        if z > self.spike_z:
            x_clean = med
        # --- 3) drop  (раптове падіння)
        elif (med - x) / (abs(med) + self.eps) > self.drop_rel:
            x_clean = med        # «підтягуємо» до баз-лайну
        else:
            x_clean = x

        # --- 4) drift – забираємо лінійний тренд у буфері
        # if len(self.buf) == self.window:
        #     y = np.asarray(self.buf)
        #     t = np.arange(len(y))
        #     coef = np.polyfit(t, y, 1)    # slope, intercept
        #     if abs(coef[0] / (abs(coef[1]) + self.eps)) > self.drift_slope:
        #         trend = coef[0] * (len(y) - 1)  # очікувана різниця
        #         x_clean -= trend

        self.last_good = x_clean
        return x_clean


class MultiSignalDetector:
    """
    Тримає один SignalAnomalyDetector на кожну колонку.
    Виклик detect(df, cols) обробляє датасет цілком.
    """
    def __init__(self, columns: Iterable[str], **det_kwargs):
        self.detectors: Dict[str, SignalAnomalyDetector] = {
            c: SignalAnomalyDetector(**det_kwargs) for c in columns
        }

    def clean_row(self, row: dict) -> dict:
        corr = {}
        for c, det in self.detectors.items():
            corr[c] = det.update(row[c])
        return corr

    def clean_dataframe(self, df, in_place=False):
        if not in_place:
            df = df.copy()
        for c in self.detectors:
            cleaned = [self.detectors[c].update(v) for v in df[c].values]
            df[c] = cleaned
        return df