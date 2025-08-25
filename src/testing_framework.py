# testing_framework.py
"""
Model Testing Framework
=======================

Призначення:
- Єдина точка для побудови тестових сценаріїв порівняння моделей на базі генератора даних (data_gen.py) і реєстру моделей (model.py).
- Два шари конфігурацій: GlobalConfig (дані/спліти/фічі) і WorkingConfig (моделі/метрики/візуалізація).
- Базовий клас тесту + приклади NoiseRobustnessTest і NonlinearityRobustnessTest.
- Узгоджене масштабування X/Y, коректне порівняння в реальних одиницях, безпечні метрики при низькій дисперсії Y.

Зовнішні залежності: numpy, pandas, scikit-learn, (опційно) matplotlib для візуалізації.

Інтеграція:
- Дані: передайте функцію/об'єкт-генератор з data_gen.py (наприклад, DataGenerator) через DataFactory.
- Моделі: зареєструйте білдери з model.py у ModelFactory.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================
#        CONFIG LAYER
# =========================

@dataclass
class GlobalConfig:
    # Дані
    n_samples: int = 10_000
    seed: int = 42
    # Специфіка генератора (прозоро передається в data_gen)
    generator_params: Dict[str, Any] = field(default_factory=dict)
    # Лаги для ARX-підходів (0 — без лагів)
    lag_depth: int = 0
    # Розбиття
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    shuffle: bool = False
    # Масштабування
    scale_x: bool = True
    scale_y: bool = True

    def ensure_valid(self) -> None:
        assert 0 < self.train_size < 1 and 0 <= self.val_size < 1 and 0 < self.test_size < 1
        s = self.train_size + self.val_size + self.test_size
        if abs(s - 1.0) > 1e-6:
            # Нормалізуємо пропорції
            self.train_size /= s
            self.val_size /= s
            self.test_size /= s


@dataclass
class WorkingConfig:
    # Перелік моделей до порівняння (імена) + параметри білдерів
    model_configs: List[Dict[str, Any]] = field(default_factory=list)
    # Метрики, які обчислюємо
    metrics: Tuple[str, ...] = ("RMSE", "MAE", "R2")
    # Поведінка логування/збереження
    save_predictions: bool = True
    save_intermediate: bool = False
    # Обробка низької дисперсії
    low_var_eps: float = 1e-8
    low_var_cv_threshold: float = 0.01  # 1%

    def add_model(self, name: str, **builder_params: Any) -> None:
        self.model_configs.append({"name": name, "params": builder_params})


@dataclass
class ScenarioOverride:
    """
    Локальні зміни над GlobalConfig, специфічні для тесту/ітерації.
    Напр., інший рівень шуму або параметри нелінійностей.
    """
    name: str
    generator_overrides: Dict[str, Any] = field(default_factory=dict)
    lag_depth: Optional[int] = None


# =========================
#       DATA LAYER
# =========================

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    X_test: np.ndarray
    y_test_real: np.ndarray           # y у реальних одиницях
    y_test_scaled: Optional[np.ndarray]  # y у масштабі scaler_y
    scaler_x: Optional[StandardScaler]
    scaler_y: Optional[StandardScaler]
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None


class DataFactory:
    """
    Загортає генератор даних (з data_gen.py) і забезпечує:
    - стабільне масштабування,
    - опціональне створення лагових ознак,
    - розбиття train/val/test.
    """
    def __init__(self, generator: Callable[[Mapping[str, Any]], pd.DataFrame]):
        """
        generator: callable(global_params) -> DataFrame зі стовпцями:
            - 'features': масив/список або окремі колонки X
            - 'targets': масив/список або окремі колонки Y
        Рекомендація: повертати DataFrame з явними колонками X_* і Y_*
        """
        self.generator = generator
        self._scaler_x: Optional[StandardScaler] = None
        self._scaler_y: Optional[StandardScaler] = None

    def _create_lag_features(self, df: pd.DataFrame, lag_depth: int,
                             target_cols: Sequence[str]) -> pd.DataFrame:
        if lag_depth <= 0:
            return df
        df = df.copy()
        for col in [c for c in df.columns if c not in target_cols]:
            for l in range(1, lag_depth + 1):
                df[f"{col}_lag{l}"] = df[col].shift(l)
        # Вирізаємо прогалини
        df = df.dropna().reset_index(drop=True)
        return df

    def _split(self, X: np.ndarray, y: np.ndarray,
               train_size: float, val_size: float, test_size: float,
               shuffle: bool, seed: int) -> Tuple[np.ndarray, ...]:
        n = X.shape[0]
        idx = np.arange(n)
        if shuffle:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
        n_train = int(round(n * train_size))
        n_val = int(round(n * val_size))
        n_test = n - n_train - n_val
        if n_test <= 0:
            raise ValueError("test_size занадто малий — немає тестових зразків")
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train + n_val] if n_val > 0 else None
        y_val = y[n_train:n_train + n_val] if n_val > 0 else None
        X_test = X[-n_test:]
        y_test = y[-n_test:]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def create_dataset(self, gcfg: GlobalConfig,
                       overrides: Optional[ScenarioOverride] = None) -> Dataset:
        gcfg = GlobalConfig(**asdict(gcfg))  # ізолюємо зміни
        gcfg.ensure_valid()

        # 1) Генерація даних
        gen_params = dict(gcfg.generator_params)
        if overrides and overrides.generator_overrides:
            gen_params.update(overrides.generator_overrides)

        df: pd.DataFrame = self.generator({
            "n_samples": gcfg.n_samples,
            "seed": gcfg.seed,
            **gen_params
        })

        # Визначимо target та features
        target_cols = [c for c in df.columns if c.startswith("Y_") or c.startswith("target")]
        feature_cols = [c for c in df.columns if c not in target_cols]

        # 2) Лаги
        lag_depth = gcfg.lag_depth if overrides is None or overrides.lag_depth is None else overrides.lag_depth
        df = self._create_lag_features(df, lag_depth, target_cols)

        # 3) Масиви
        X = df[feature_cols].to_numpy(dtype=float)
        y = df[target_cols].to_numpy(dtype=float)

        # 4) Спліт
        X_train, y_train, X_val, y_val, X_test, y_test = self._split(
            X, y, gcfg.train_size, gcfg.val_size, gcfg.test_size, gcfg.shuffle, gcfg.seed
        )

        # 5) Масштабування — фіксуємо scaler-и один раз на першому датасеті
        if gcfg.scale_x:
            if self._scaler_x is None:
                self._scaler_x = StandardScaler().fit(X_train)
            X_train_s = self._scaler_x.transform(X_train)
            X_val_s = self._scaler_x.transform(X_val) if X_val is not None else None
            X_test_s = self._scaler_x.transform(X_test)
        else:
            X_train_s, X_val_s, X_test_s = X_train, X_val, X_test

        if gcfg.scale_y:
            if self._scaler_y is None:
                self._scaler_y = StandardScaler().fit(y_train)
            y_train_s = self._scaler_y.transform(y_train)
            y_val_s = self._scaler_y.transform(y_val) if y_val is not None else None
            y_test_s = self._scaler_y.transform(y_test)
            y_test_real = y_test  # збережемо «реальні» одиниці
        else:
            y_train_s, y_val_s, y_test_s = y_train, y_val, y_test
            y_test_real = y_test

        return Dataset(
            X_train=X_train_s,
            y_train=y_train_s,
            X_val=X_val_s,
            y_val=y_val_s,
            X_test=X_test_s,
            y_test_real=y_test_real,
            y_test_scaled=y_test_s,
            scaler_x=self._scaler_x if gcfg.scale_x else None,
            scaler_y=self._scaler_y if gcfg.scale_y else None,
            feature_names=feature_cols,
            target_names=target_cols
        )

class MagneticDataFactory:
    """
    Централізована фабрика генерації даних поверх data_gen.DataGenerator.
    Призначення:
      - завантаження reference_df з processed.parquet (або .csv),
      - генерація базових даних зі шумом і аномаліями,
      - опційне застосування нелінійності,
      - повернення стандартизованого DataFrame з X та Y_*.

    Використання з існуючим TestHarness (без його змін):
      mf = MagneticDataFactory("processed.parquet")
      harness = TestHarness(
          data_generator=mf.build_generator(),
          model_registry=...,
          global_config=...,
          working_config=...
      )
    """
    def __init__(self, processed_path: str = "processed.parquet"):
        from pathlib import Path
        self.processed_path = Path(processed_path)
        self.reference_df = self._load_reference_df()

    def _load_reference_df(self) -> "pd.DataFrame":
        import pandas as pd
        if not self.processed_path.exists():
            raise FileNotFoundError(f"Reference файл не знайдено: {self.processed_path}")
        suffix = self.processed_path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(self.processed_path)
        if suffix == ".csv":
            return pd.read_csv(self.processed_path)
        raise ValueError(f"Непідтримуване розширення: {suffix}")

    def _generate_core(self, params: "dict[str, object]") -> "pd.DataFrame":
        """
        Єдине місце роботи з DataGenerator:
        - базова генерація,
        - шум/аномалії,
        - опційна нелінійність,
        - стандартизація колонок (3 X + 2 Y_*).
        """
        import pandas as pd
        from data_gen import DataGenerator

        # 0) Базові параметри з дефолтами
        n_samples = int(params.get("n_samples", params.get("N_data", 7000)))
        seed = int(params.get("seed", 42))
        control_pts = int(params.get("control_pts", max(20, n_samples // 10)))
        time_step_s = float(params.get("time_step_s", 5.0))

        time_constants_s = params.get("time_constants_s", {
            "concentrate_fe_percent": 8.0,
            "tailings_fe_percent": 10.0,
            "concentrate_mass_flow": 5.0,
            "tailings_mass_flow": 7.0,
        })
        dead_times_s = params.get("dead_times_s", {
            "concentrate_fe_percent": 20.0,
            "tailings_fe_percent": 25.0,
            "concentrate_mass_flow": 20.0,
            "tailings_mass_flow": 25.0,
        })
        noise_level = params.get("noise_level", "none")  # 'none' | 'low' | 'medium' | 'high'
        plant_model_type = params.get("plant_model_type", "rf")

        # 1) Генератор
        gen = DataGenerator(
            reference_df=self.reference_df,
            seed=seed,
            time_step_s=time_step_s,
            time_constants_s=time_constants_s,
            dead_times_s=dead_times_s,
            true_model_type=plant_model_type,
        )

        # 2) Аномалії (централізовано)
        use_anomalies = bool(params.get("use_anomalies", True))
        if use_anomalies:
            train_frac = float(params.get("train_size", 0.7))
            val_frac = float(params.get("val_size", 0.15))
            test_frac = float(params.get("test_size", 0.15))
            anomaly_cfg = DataGenerator.generate_anomaly_config(
                N_data=n_samples,
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                seed=seed,
            )
        else:
            anomaly_cfg = None

        # 3) База зі шумом
        df_base = gen.generate(
            T=n_samples,
            control_pts=control_pts,
            n_neighbors=int(params.get("n_neighbors", 5)),
            noise_level=noise_level,
            anomaly_config=anomaly_cfg,
        )

        # 4) Нелінійність (за потреби)
        if bool(params.get("enable_nonlinear", True)):
            nonlin_cfg = params.get("nonlinear_config", {
                "concentrate_fe_percent": ("pow", 2.0),
                "concentrate_mass_flow": ("pow", 1.5),
            })
            df_used = gen.generate_nonlinear_variant(
                base_df=df_base,
                non_linear_factors=nonlin_cfg,
                noise_level="none",          # шум уже додано
                anomaly_config=anomaly_cfg,  # консистентний розклад аномалій
            )
        else:
            df_used = df_base

        # 5) Форматування виходу
        cols_inp = ["feed_fe_percent", "ore_mass_flow", "solid_feed_percent"]
        cols_y = ["concentrate_fe_percent", "concentrate_mass_flow"]
        missing = [c for c in cols_inp + cols_y if c not in df_used.columns]
        if missing:
            raise KeyError(f"Відсутні колонки у згенерованих даних: {missing}")

        df_out = df_used[cols_inp + cols_y].copy()
        df_out.rename(columns={
            "concentrate_fe_percent": "Y_concentrate_fe_percent",
            "concentrate_mass_flow": "Y_concentrate_mass_flow",
        }, inplace=True)
        return df_out

    def build_generator(self):
        """
        Повертає функцію-генератор, сумісну з існуючим TestHarness (через data_generator=...).
        Вона делегує всю логіку в _generate_core і підставляє дефолти для часток, якщо їх не передали.
        """
        def generate_dataframe(params: "dict[str, object]") -> "pd.DataFrame":
            p = dict(params or {})
            p.setdefault("train_size", 0.7)
            p.setdefault("val_size", 0.15)
            p.setdefault("test_size", 0.15)
            return self._generate_core(p)
        return generate_dataframe

# =========================
#       MODEL LAYER
# =========================

class ModelFactory:
    """
    Реєстр моделей. Білдер: (name, params) -> model із .fit(X, y) і .predict(X).
    """
    def __init__(self):
        self._builders: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, builder: Callable[..., Any]) -> None:
        self._builders[name] = builder

    def create(self, name: str, **params: Any) -> Any:
        if name not in self._builders:
            raise KeyError(f"Модель '{name}' не зареєстрована")
        return self._builders[name](**params)

    def list_models(self) -> List[str]:
        return list(self._builders.keys())


# =========================
#      METRICS / UTILS
# =========================

def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray, eps_var: float = 1e-8) -> float:
    # Варіант R², стійкий до низької дисперсії
    denom = float(np.var(y_true, ddof=1))
    if not np.isfinite(denom) or denom < eps_var:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def compute_metrics(y_true_real: np.ndarray, y_pred_real: np.ndarray,
                    metrics: Sequence[str],
                    low_var_eps: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "RMSE" in metrics:
        out["RMSE"] = float(np.sqrt(mean_squared_error(y_true_real, y_pred_real)))
    if "MAE" in metrics:
        out["MAE"] = float(mean_absolute_error(y_true_real, y_pred_real))
    if "R2" in metrics:
        out["R2"] = _safe_r2(y_true_real, y_pred_real, eps_var=low_var_eps)
    return out


def inverse_if_needed(y_scaled: np.ndarray, scaler_y: Optional[StandardScaler]) -> np.ndarray:
    if scaler_y is None:
        return y_scaled
    return scaler_y.inverse_transform(y_scaled)


# =========================
#       RUN ENGINE
# =========================

@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    train_time_sec: float
    predictions: np.ndarray  # у реальних одиницях


class ComparisonRunner:
    def __init__(self, model_factory: ModelFactory, wcfg: WorkingConfig):
        self.model_factory = model_factory
        self.wcfg = wcfg

    def train_and_eval(self, dataset: Dataset,
                       model_cfgs: List[Dict[str, Any]]) -> List[ModelResult]:
        results: List[ModelResult] = []
        for cfg in model_cfgs:
            name = cfg["name"]
            params = cfg.get("params", {})

            model = self.model_factory.create(name, **params)

            t0 = time.perf_counter()
            model.fit(dataset.X_train, dataset.y_train)
            train_time = time.perf_counter() - t0

            y_pred_s = model.predict(dataset.X_test)
            y_pred_real = inverse_if_needed(y_pred_s, dataset.scaler_y)

            # Вирівнюємо форми
            y_true_real = dataset.y_test_real
            if y_true_real.ndim == 1:
                y_true_real = y_true_real.reshape(-1, 1)
            if y_pred_real.ndim == 1:
                y_pred_real = y_pred_real.reshape(-1, 1)

            m = compute_metrics(y_true_real, y_pred_real, self.wcfg.metrics, self.wcfg.low_var_eps)

            results.append(ModelResult(
                name=name, metrics=m, train_time_sec=train_time, predictions=y_pred_real
            ))
        return results


# =========================
#       TESTS LAYER
# =========================

class BaseTestCase:
    """
    Базовий клас тесту:
    - має доступ до DataFactory і ComparisonRunner,
    - формує сценарні overrides,
    - повертає структуровані результати з логом метрик.
    """
    def __init__(self, name: str, data_factory: DataFactory,
                 runner: ComparisonRunner, gcfg: GlobalConfig, wcfg: WorkingConfig):
        self.name = name
        self.data_factory = data_factory
        self.runner = runner
        self.gcfg = gcfg
        self.wcfg = wcfg

    def iter_scenarios(self) -> Iterable[ScenarioOverride]:
        """
        Має бути перевизначений у нащадках: yield ScenarioOverride(...)
        """
        yield ScenarioOverride(name="base")

    def run(self) -> Dict[str, Any]:
        all_results: Dict[str, Any] = {
            "test_name": self.name,
            "global_config": asdict(self.gcfg),
            "working_config": asdict(self.wcfg),
            "scenarios": {}
        }
        for sc in self.iter_scenarios():
            ds = self.data_factory.create_dataset(self.gcfg, overrides=sc)
            results = self.runner.train_and_eval(ds, self.wcfg.model_configs)

            # Агрегуємо
            sc_res: Dict[str, Any] = {"override": asdict(sc), "models": {}}
            for mr in results:
                sc_res["models"][mr.name] = {
                    "metrics": mr.metrics,
                    "train_time_sec": mr.train_time_sec,
                }
                if self.wcfg.save_predictions:
                    sc_res["models"][mr.name]["predictions_head"] = mr.predictions[:5].tolist()
            all_results["scenarios"][sc.name] = sc_res
        return all_results


class NoiseRobustnessTest(BaseTestCase):
    """
    Варіює рівень шуму генератора (через generator_overrides) і оцінює стабільність метрик.
    Очікується, що data_gen розуміє параметр 'noise_std' або подібні.
    """
    def __init__(self, data_factory: DataFactory, runner: ComparisonRunner,
                 gcfg: GlobalConfig, wcfg: WorkingConfig,
                 noise_levels: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.2),
                 noise_key: str = "noise_std"):
        super().__init__("NoiseRobustness", data_factory, runner, gcfg, wcfg)
        self.noise_levels = list(noise_levels)
        self.noise_key = noise_key

    def iter_scenarios(self) -> Iterable[ScenarioOverride]:
        for nl in self.noise_levels:
            yield ScenarioOverride(
                name=f"noise_{nl}",
                generator_overrides={self.noise_key: float(nl)}
            )


class NonlinearityRobustnessTest(BaseTestCase):
    """
    Варіює конфіг нелінійностей генератора (data_gen повинен приймати прапори/коефіцієнти).
    """
    def __init__(self, data_factory: DataFactory, runner: ComparisonRunner,
                 gcfg: GlobalConfig, wcfg: WorkingConfig,
                 levels: Mapping[str, Mapping[str, Any]]):
        """
        levels: {"linear": {...}, "weak": {...}, "moderate": {...}, "strong": {...}}
        """
        super().__init__("NonlinearityRobustness", data_factory, runner, gcfg, wcfg)
        self.levels = levels

    def iter_scenarios(self) -> Iterable[ScenarioOverride]:
        for name, cfg in self.levels.items():
            yield ScenarioOverride(
                name=name,
                generator_overrides=dict(cfg)
            )


# =========================
#      TEST HARNESS
# =========================

class TestHarness:
    """
    Високорівневий фасад:
    - будує DataFactory і ModelFactory,
    - тримає спільні конфіги,
    - дозволяє запускати набір тестів,
    - зберігає звіт.
    """
    def __init__(self,
                 data_generator: Callable[[Mapping[str, Any]], pd.DataFrame],
                 model_registry: Mapping[str, Callable[..., Any]],
                 global_config: GlobalConfig,
                 working_config: WorkingConfig):
        self.gcfg = global_config
        self.wcfg = working_config

        # Дані
        self.data_factory = DataFactory(data_generator)
        # Моделі
        self.model_factory = ModelFactory()
        for name, builder in model_registry.items():
            self.model_factory.register(name, builder)
        # Ранер
        self.runner = ComparisonRunner(self.model_factory, self.wcfg)

    def run_tests(self, tests: Sequence[BaseTestCase]) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_config": asdict(self.gcfg),
            "working_config": asdict(self.wcfg),
            "tests": {}
        }
        for t in tests:
            res = t.run()
            report["tests"][t.name] = res
        return report

    @staticmethod
    def save_report(report: Dict[str, Any], path: Union[str, "os.PathLike"]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


# =========================
#      EXAMPLE USAGE
# =========================
"""
Приклад інтеграції (псевдокод; помістіть у окремий файл або блок if __name__ == '__main__'):

from data_gen import generate_dataframe  # ваша функція: params -> DataFrame
from model import build_arx_ols, build_arx_ridge, build_arx_lasso  # білдери моделей

# 1) Глобальна конфігурація (дані)
gcfg = GlobalConfig(
    n_samples=15000,
    seed=123,
    generator_params={"base_trend": 1.0, "noise_std": 0.02, "nonlinear": False},
    lag_depth=3,
    train_size=0.7, val_size=0.15, test_size=0.15,
    shuffle=False,
    scale_x=True, scale_y=True
)

# 2) Робоча конфігурація (моделі/метрики)
wcfg = WorkingConfig(metrics=("RMSE", "MAE", "R2"))
wcfg.add_model("ARX_OLS")
wcfg.add_model("ARX_Ridge", alpha=1.0)
wcfg.add_model("ARX_Lasso", alpha=0.01)

# 3) Реєстр моделей
model_registry = {
    "ARX_OLS": build_arx_ols,
    "ARX_Ridge": build_arx_ridge,
    "ARX_Lasso": build_arx_lasso,
}

# 4) Хаб
h = TestHarness(
    data_generator=generate_dataframe,
    model_registry=model_registry,
    global_config=gcfg,
    working_config=wcfg
)

# 5) Набір тестів
noise_test = NoiseRobustnessTest(h.data_factory, h.runner, gcfg, wcfg,
                                 noise_levels=[0.0, 0.02, 0.05, 0.1, 0.2],
                                 noise_key="noise_std")

nl_test = NonlinearityRobustnessTest(
    h.data_factory, h.runner, gcfg, wcfg,
    levels={
        "linear": {"nonlinear": False},
        "weak": {"nonlinear": True, "nl_strength": 0.3},
        "moderate": {"nonlinear": True, "nl_strength": 0.6},
        "strong": {"nonlinear": True, "nl_strength": 1.0},
    }
)

# 6) Запуск
report = h.run_tests([noise_test, nl_test])
h.save_report(report, "test_report.json")

"""
