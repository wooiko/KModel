# anomaly_detector.py

import numpy as np

class AnomalyDetector:
    """
    Простий онлайн‐детектор і коректор аномалій:
      - spike/drop: якщо |value - median(window)| > z_thresh·MAD
      - freeze: ідентичні значення протягом window → відкат до median
      - drift: повільний тренд автоматично згладжується в рамках same window
    """
    def __init__(self, window:int=5, z_thresh:float=3.0):
        self.window    = window
        self.z_thresh  = z_thresh
        self.history   = []  # зберігає вже «очищені» значення

    def correct(self, value: float) -> float:
        """
        Повертає очищене значення (або оригінал), зберігає його в history.
        """
        # поки що накопичуємо початкові значення
        if len(self.history) < self.window:
            self.history.append(value)
            return value

        window_data = np.array(self.history[-self.window:])
        median = np.median(window_data)
        # median absolute deviation
        mad = np.median(np.abs(window_data - median))
        if mad < 1e-5:
            mad = np.std(window_data) + 1e-5

        # robust z‐score
        z_score = abs(0.6745 * (value - median) / mad)

        # spike/drop
        if z_score > self.z_thresh:
            corrected = median
        else:
            corrected = value

        # freeze: якщо нове ― точно таке саме як попереднє і window багато однакових
        if corrected == self.history[-1] and np.ptp(window_data) < 1e-5:
            corrected = median

        # drift тут по факту пригальмовується через те саме median‐коригування

        # додаємо в історію
        self.history.append(corrected)
        return corrected