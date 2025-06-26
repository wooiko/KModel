# maf.py

import numpy as np
from collections import deque

class MovingAverageFilter:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self.buffer.append(value)
        return float(np.mean(self.buffer))
