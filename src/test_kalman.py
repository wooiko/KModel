# test_kalman.py

import numpy as np
import matplotlib.pyplot as plt
from kalman_observer import DisturbanceObserverKalman

def main():
    # Параметри синтетичних даних
    N = 200
    true_d = np.zeros(N)
    true_d[50:] = 1.5     # step‐збурення 1.5 одиниці з часу 50
    C_d = 1.0
    sigma_noise = 0.05
    
    # Прогноз без збурень
    y_pred = np.zeros(N)
    # Вимірювання із збуренням і шумом
    y_meas = C_d * true_d + np.random.randn(N) * sigma_noise
    
    # Ініціалізація Калмана
    kf = DisturbanceObserverKalman(
        A_d=1.0, C_d=C_d,
        Q=1e-2, R=1e-2,
        P=1.0, d_est=0.0
    )
    
    d_est = np.zeros(N)
    for k in range(N):
        d_est[k] = kf.update(y_meas[k], y_pred[k])
    
    # Візуалізація
    plt.figure(figsize=(8,4))
    plt.plot(true_d,   label='True disturbance', linestyle='--')
    plt.plot(d_est,    label='Estimated disturbance')
    plt.xlabel('k')
    plt.ylabel('d')
    plt.legend()
    plt.grid(True)
    plt.title('Перевірка DisturbanceObserverKalman')
    plt.show()

if __name__ == '__main__':
    main()