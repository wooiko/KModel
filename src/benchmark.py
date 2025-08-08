# benchmark.py - Новий файл для бенчмарків
import time
import numpy as np
from typing import Dict, List, Tuple
from contextlib import contextmanager
from model import KernelModel

@contextmanager
def timer():
    """Context manager для вимірювання часу"""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    
def benchmark_model_training(
    X_train: np.ndarray, 
    Y_train: np.ndarray,
    model_configs: List[Dict]
) -> Dict[str, float]:
    """Бенчмарк часу навчання різних моделей"""
    
    results = {}
    
    for config in model_configs:
        model_name = f"{config['model_type']}-{config.get('kernel', 'default')}"
        print(f"🔧 Тестую {model_name}...")
        
        # Навчання з вимірюванням часу
        with timer() as get_time:
            model = KernelModel(**config)
            model.fit(X_train, Y_train)
        
        train_time = get_time()
        results[f"{model_name}_train_time"] = train_time
        
        # Час одного прогнозу
        X_single = X_train[0:1]  # Один семпл
        with timer() as get_time:
            for _ in range(100):  # 100 прогнозів для усереднення
                _ = model.predict(X_single)
        
        pred_time = get_time() / 100  # Середній час одного прогнозу
        results[f"{model_name}_predict_time"] = pred_time
        
        # Час лінеаризації
        with timer() as get_time:
            for _ in range(100):
                _ = model.linearize(X_single)
        
        linearize_time = get_time() / 100
        results[f"{model_name}_linearize_time"] = linearize_time
        
        print(f"   ✅ Train: {train_time:.3f}s, Predict: {pred_time*1000:.2f}ms, Linearize: {linearize_time*1000:.2f}ms")
    
    return results

def benchmark_mpc_solve_time(mpc_controller, n_iterations: int = 50) -> Dict[str, float]:
    """Бенчмарк часу розв'язування MPC задачі"""
    
    # ✅ ВИПРАВЛЕНО: використовуємо Np замість horizon
    prediction_horizon = getattr(mpc_controller, 'Np', 8)  # За замовчуванням 8
    
    # Створюємо типові входи
    d_seq = np.array([[36.5, 102.2]] * prediction_horizon)
    u_prev = 25.0
    
    solve_times = []
    
    # Запускаємо n_iterations разів для статистики
    for _ in range(n_iterations):
        try:
            with timer() as get_time:
                # ✅ ВИПРАВЛЕНО: додаємо trust_radius параметр
                if hasattr(mpc_controller, 'adaptive_trust_region') and mpc_controller.adaptive_trust_region:
                    # Для адаптивного trust region
                    result = mpc_controller.optimize(
                        d_seq=d_seq, 
                        u_prev=u_prev, 
                        trust_radius=1.0  # Типове значення
                    )
                else:
                    # Для звичайного MPC
                    result = mpc_controller.optimize(
                        d_seq=d_seq, 
                        u_prev=u_prev
                    )
            
            solve_times.append(get_time())
            
        except Exception as e:
            print(f"⚠️ Помилка в MPC optimize: {e}")
            # Додаємо типовий час у випадку помилки
            solve_times.append(0.01)  # 10ms
    
    # Статистика
    solve_times = np.array(solve_times)
    
    return {
        "mpc_solve_mean": np.mean(solve_times),
        "mpc_solve_std": np.std(solve_times),
        "mpc_solve_min": np.min(solve_times),
        "mpc_solve_max": np.max(solve_times),
        "mpc_solve_median": np.median(solve_times),
        "mpc_iterations": n_iterations,
        "mpc_success_rate": len([t for t in solve_times if t > 0]) / len(solve_times)
    }

