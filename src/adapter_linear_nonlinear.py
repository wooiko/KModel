# adapter_linear_nonlinear.py
# Адаптер для нової тестової бази:
# - генератор даних поверх data_gen.py
# - реєстрація білдерів лінійних моделей із model.py
# - демонстраційний тест "Linear on Nonlinear with Anomalies"

from __future__ import annotations

from typing import Any, Mapping, Dict, Callable, Sequence, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# 1) Імпортуємо ваші генератори/моделі
from data_gen import DataGenerator
from model import KernelModel

# 2) Імпортуємо ядро нової тестової бази (з попереднього артефакту)
#    Збережіть той файл як testing_framework.py поруч з цим адаптером.
from testing_framework import (
    GlobalConfig,
    WorkingConfig,
    DataFactory,
    ModelFactory,
    ComparisonRunner,
    BaseTestCase,
    ScenarioOverride,
    TestHarness,
)


# =========================
#  ГЕНЕРАТОР ДАНИХ (АДАПТЕР)
# =========================

    use_anomalies = params.get("use_anomalies", True)
# def _load_reference_df(params: Mapping[str, Any]) -> pd.DataFrame:
#     """Завантажує референсні дані."""
#     if "reference_df" in params and params["reference_df"] is not None:
#         ref = params["reference_df"]
#         if isinstance(ref, pd.DataFrame):
#             return ref
#         elif isinstance(ref, (str, Path)):
#             p = Path(ref)
#             if p.suffix.lower() in {".parquet"} and p.exists():
#                 return pd.read_parquet(p)
#             if p.suffix.lower() in {".csv"} and p.exists():
#                 return pd.read_csv(p)
#             raise FileNotFoundError(f"reference_df={ref} не знайдено")
#         else:
#             raise TypeError("reference_df має бути DataFrame або шляхем до файлу")
#     # Fallback: шукаємо processed.parquet поруч
#     if Path("processed.parquet").exists():
#         return pd.read_parquet("processed.parquet")
#     raise FileNotFoundError(
#         "Не передано reference_df і не знайдено processed.parquet"
#     )


# def _build_anomaly_config(n_samples: int, params: Mapping[str, Any]) -> Optional[dict]:
#     """Генерує anomaly_config для DataGenerator на основі часток train/val/test."""
#     use_anomalies = params.get("use_anomalies", True)
#     if not use_anomalies:
#         return None

#     train = float(params.get("train_size", 0.7))
#     val = float(params.get("val_size", 0.15))
#     test = float(params.get("test_size", 0.15))
#     seed = int(params.get("seed", 42))

#     # Використовуємо статичний метод з вашого data_gen.py
#     return DataGenerator.generate_anomaly_config(
#         N_data=n_samples,
#         train_frac=train,
#         val_frac=val,
#         test_frac=test,
#         seed=seed,
#     )


# def generate_dataframe(params: Mapping[str, Any]) -> pd.DataFrame:
#     """
#     Адаптер-функція для DataFactory: формує дані у потрібному форматі.
#     Повертає лише потрібні колонки:
#       X: feed_fe_percent, ore_mass_flow, solid_feed_percent
#       Y: Y_concentrate_fe_percent, Y_concentrate_mass_flow
#     """
#     # 0) Набір базових параметрів
#     n_samples = int(params.get("n_samples", params.get("N_data", 7000)))
#     seed = int(params.get("seed", 42))
#     control_pts = int(params.get("control_pts", max(20, n_samples // 10)))

#     time_step_s = float(params.get("time_step_s", 5.0))
#     time_constants_s = params.get("time_constants_s", {
#         "concentrate_fe_percent": 8.0,
#         "tailings_fe_percent": 10.0,
#         "concentrate_mass_flow": 5.0,
#         "tailings_mass_flow": 7.0,
#     })
#     dead_times_s = params.get("dead_times_s", {
#         "concentrate_fe_percent": 20.0,
#         "tailings_fe_percent": 25.0,
#         "concentrate_mass_flow": 20.0,
#         "tailings_mass_flow": 25.0,
#     })
#     noise_level = params.get("noise_level", "none")  # 'none' | 'low' | 'medium' | 'high'

#     # 1) Референс і генератор
#     reference_df = _load_reference_df(params)
#     gen = DataGenerator(
#         reference_df=reference_df,
#         seed=seed,
#         time_step_s=time_step_s,
#         time_constants_s=time_constants_s,
#         dead_times_s=dead_times_s,
#         true_model_type=params.get("plant_model_type", "rf"),
#     )

#     # 2) Аномалії
#     anomaly_cfg = _build_anomaly_config(n_samples, params)

#     # 3) Базовий датасет
#     df_base = gen.generate(
#         T=n_samples,
#         control_pts=control_pts,
#         n_neighbors=int(params.get("n_neighbors", 5)),
#         noise_level=noise_level,
#         anomaly_config=anomaly_cfg,
#     )

#     # 4) Нелінійний варіант (за потреби)
#     enable_nl = bool(params.get("enable_nonlinear", True))
#     if enable_nl:
#         nonlin_cfg = params.get(
#             "nonlinear_config",
#             {
#                 "concentrate_fe_percent": ("pow", 2.0),
#                 "concentrate_mass_flow": ("pow", 1.5),
#             },
#         )
#         df_used = gen.generate_nonlinear_variant(
#             base_df=df_base,
#             non_linear_factors=nonlin_cfg,
#             noise_level="none",             # шум уже додано на базовому кроці
#             anomaly_config=anomaly_cfg,     # консистентна конфігурація аномалій
#         )
#     else:
#         df_used = df_base

#     # 5) Відібрані колонки + перейменування цілей із префіксом Y_
#     cols_inp = ["feed_fe_percent", "ore_mass_flow", "solid_feed_percent"]
#     cols_y = ["concentrate_fe_percent", "concentrate_mass_flow"]

#     missing = [c for c in cols_inp + cols_y if c not in df_used.columns]
#     if missing:
#         raise KeyError(f"Відсутні колонки у згенерованих даних: {missing}")

#     df_out = df_used[cols_inp + cols_y].copy()
#     df_out.rename(
#         columns={
#             "concentrate_fe_percent": "Y_concentrate_fe_percent",
#             "concentrate_mass_flow": "Y_concentrate_mass_flow",
#         },
#         inplace=True,
#     )

#     # тільки потрібні стовпці: 3 X + 2 Y
#     return df_out


# =========================
#    БІЛДЕРИ ЛІНІЙНИХ МОДЕЛЕЙ
# =========================

def build_arx_ols(**kwargs) -> KernelModel:
    """Звичайна лінійна регресія."""
    return KernelModel(
        model_type="linear",
        linear_type="ols",
        poly_degree=int(kwargs.get("poly_degree", 1)),
        include_bias=bool(kwargs.get("include_bias", True)),
        find_optimal_params=bool(kwargs.get("find_optimal_params", False)),
        n_iter_random_search=int(kwargs.get("n_iter_random_search", 20)),
    )

def build_arx_ridge(alpha: float = 1.0, **kwargs) -> KernelModel:
    """Ridge-регресія."""
    return KernelModel(
        model_type="linear",
        linear_type="ridge",
        alpha=float(alpha),
        poly_degree=int(kwargs.get("poly_degree", 1)),
        include_bias=bool(kwargs.get("include_bias", True)),
        find_optimal_params=bool(kwargs.get("find_optimal_params", False)),
        n_iter_random_search=int(kwargs.get("n_iter_random_search", 20)),
    )

def build_arx_lasso(alpha: float = 0.01, **kwargs) -> KernelModel:
    """Lasso-регресія."""
    return KernelModel(
        model_type="linear",
        linear_type="lasso",
        alpha=float(alpha),
        poly_degree=int(kwargs.get("poly_degree", 1)),
        include_bias=bool(kwargs.get("include_bias", True)),
        find_optimal_params=bool(kwargs.get("find_optimal_params", False)),
        n_iter_random_search=int(kwargs.get("n_iter_random_search", 20)),
    )

def get_model_registry() -> Dict[str, Callable[..., KernelModel]]:
    """Повертає реєстр білдерів для TestHarness."""
    return {
        "ARX_OLS": build_arx_ols,
        "ARX_Ridge": build_arx_ridge,
        "ARX_Lasso": build_arx_lasso,
    }

def visualize_report(report_or_path, out_dir: str = "results/figs") -> dict:
    """
    Будує візуалізації метрик з тестового звіту:
      - accuracy_comparison.png: 1x3 (RMSE, MAE, R2) лінійні графіки по сценаріях (weak→moderate→strong) з розбиттям по моделях
      - train_time.png: стовпчики часу навчання
      - metrics_long.csv: метрики у довгому форматі (для подальшого аналізу)
    Параметри:
      - report_or_path: dict (вже завантажений звіт) або шлях до JSON-файлу
      - out_dir: куди зберігати результати
    Повертає:
      - dict зі шляхами до згенерованих файлів
    """
    import json
    from pathlib import Path
    import pandas as pd

    # 0) Завантаження
    if isinstance(report_or_path, (str, Path)):
        with open(report_or_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    elif isinstance(report_or_path, dict):
        report = report_or_path
    else:
        raise TypeError("report_or_path має бути шляхом до JSON або dict-ом.")

    # 1) Дістати блок tests (він може бути словником або JSON-рядком)
    tests_raw = report.get("tests", {})
    if isinstance(tests_raw, str):
        try:
            tests = json.loads(tests_raw)
        except json.JSONDecodeError as e:
            raise ValueError("Поле 'tests' є рядком, але не парситься як JSON.") from e
    elif isinstance(tests_raw, dict):
        tests = tests_raw
    else:
        raise ValueError("Поле 'tests' має непідтримуваний формат.")

    if not tests:
        raise ValueError("У звіті відсутні тести (порожнє 'tests').")

    # Припускаємо один ключ верхнього рівня (назва тесту)
    test_name, test_payload = next(iter(tests.items()))

    scenarios = test_payload.get("scenarios", {})
    if not scenarios:
        raise ValueError(f"У тесті '{test_name}' відсутні сценарії.")

    # 2) Витягти метрики та час навчання в «довгий» датафрейм
    records_metrics = []
    records_time = []
    for scen_name, scen_payload in scenarios.items():
        models = scen_payload.get("models", {})
        for model_name, model_payload in models.items():
            # метрики
            m = model_payload.get("metrics", {})
            for metric_name, metric_value in m.items():
                records_metrics.append({
                    "scenario": scen_name,
                    "model": model_name,
                    "metric": metric_name,
                    "value": float(metric_value),
                })
            # час навчання
            tt = model_payload.get("train_time_sec", None)
            if tt is not None:
                records_time.append({
                    "scenario": scen_name,
                    "model": model_name,
                    "train_time_sec": float(tt),
                })

    if not records_metrics:
        raise ValueError("Не знайдено метрик у звіті.")

    df_metrics = pd.DataFrame.from_records(records_metrics)
    df_time = pd.DataFrame.from_records(records_time) if records_time else pd.DataFrame(columns=["scenario", "model", "train_time_sec"])

    # 3) Впорядкування сценаріїв (якщо доступні weak/moderate/strong — використати цей порядок)
    order_candidates = ["weak", "moderate", "strong"]
    scenarios_present = list(df_metrics["scenario"].unique())
    if all(x in scenarios_present for x in order_candidates):
        scenario_order = order_candidates
    else:
        scenario_order = sorted(scenarios_present)

    # 4) Малювання
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4.1) accuracy_comparison.png — 3 підграфіки: RMSE, MAE, R2
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        metrics_to_plot = ["RMSE", "MAE", "R2"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

        for ax, metric in zip(axes, metrics_to_plot):
            dd = df_metrics[df_metrics["metric"] == metric].copy()
            if dd.empty:
                ax.set_visible(False)
                continue
            dd["scenario"] = pd.Categorical(dd["scenario"], categories=scenario_order, ordered=True)
            dd.sort_values(["scenario", "model"], inplace=True)
            sns.lineplot(
                data=dd,
                x="scenario", y="value", hue="model", marker="o", ax=ax
            )
            ax.set_title(metric)
            ax.set_xlabel("Scenario")
            ax.set_ylabel(metric)
            ax.legend(title="Model", fontsize=8, title_fontsize=9)

        acc_path = out_dir / "accuracy_comparison.png"
        fig.suptitle(f"Accuracy metrics by scenario — {test_name}", fontsize=12)
        fig.savefig(acc_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        raise RuntimeError(f"Не вдалося побудувати accuracy_comparison.png: {e}")

    # 4.2) train_time.png — стовпчики часу навчання (якщо є)
    time_path = None
    if not df_time.empty:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="whitegrid")
            df_time["scenario"] = pd.Categorical(df_time["scenario"], categories=scenario_order, ordered=True)
            df_time.sort_values(["scenario", "model"], inplace=True)

            plt.figure(figsize=(8, 4))
            sns.barplot(data=df_time, x="scenario", y="train_time_sec", hue="model")
            plt.title(f"Train time by scenario — {test_name}")
            plt.xlabel("Scenario")
            plt.ylabel("Train time, sec")
            plt.legend(title="Model", fontsize=8, title_fontsize=9)
            time_path = out_dir / "train_time.png"
            plt.tight_layout()
            plt.savefig(time_path, dpi=150)
            plt.close()
        except Exception as e:
            raise RuntimeError(f"Не вдалося побудувати train_time.png: {e}")

    # 5) Зберегти таблицю метрик у довгому форматі
    csv_path = out_dir / "metrics_long.csv"
    df_metrics.sort_values(["metric", "scenario", "model"]).to_csv(csv_path, index=False, encoding="utf-8")

    # 6) Повернути шляхи
    out = {
        "accuracy_comparison": str(acc_path),
        "train_time": str(time_path) if time_path else None,
        "metrics_long_csv": str(csv_path),
    }
    print("Візуалізації збережено:", out)
    return out

# =========================
#        КАСТОМНИЙ ТЕСТ
# =========================

class LinearOnNonlinearAnomaliesTest(BaseTestCase):
    """
    Тест для порівняння ЛІНІЙНИХ моделей на НЕЛІНІЙНИХ даних із аномаліями.
    Варіюємо рівень нелінійності, а аномалії — завжди увімкнено.
    """
    def __init__(
        self,
        data_factory: DataFactory,
        runner: ComparisonRunner,
        gcfg: GlobalConfig,
        wcfg: WorkingConfig,
        nl_levels: Mapping[str, Mapping[str, Any]] | None = None,
        noise_level: str = "medium",         # стабільний шум
        anomaly_severity: str = "medium",    # стабільні аномалії
    ):
        super().__init__("LinearOnNonlinearWithAnomalies", data_factory, runner, gcfg, wcfg)
        self.noise_level = noise_level
        self.anomaly_severity = anomaly_severity
        # значення факторів: опуклість Fe, слабша нелінійність для мас
        self.nl_levels = nl_levels or {
            "weak": {"concentrate_fe_percent": ("pow", 1.2), "concentrate_mass_flow": ("pow", 1.1)},
            "moderate": {"concentrate_fe_percent": ("pow", 1.6), "concentrate_mass_flow": ("pow", 1.3)},
            "strong": {"concentrate_fe_percent": ("pow", 2.0), "concentrate_mass_flow": ("pow", 1.5)},
        }

    def iter_scenarios(self):
        for name, cfg in self.nl_levels.items():
            yield ScenarioOverride(
                name=name,
                generator_overrides={
                    # постійно: аномалії та шум
                    "use_anomalies": True,
                    "anomaly_severity": self.anomaly_severity,
                    "noise_level": self.noise_level,
                    # перемикаємо рівень нелінійності
                    "enable_nonlinear": True,
                    "nonlinear_config": dict(cfg),
                },
            )


# =========================
#      ШВИДКИЙ ЗАПУСК/ДЕМО
# =========================

def make_default_configs() -> tuple[GlobalConfig, WorkingConfig]:
    gcfg = GlobalConfig(
        n_samples=7000,
        seed=42,
        lag_depth=3,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        shuffle=False,
        scale_x=True,
        scale_y=True,
    )

    # Якщо у вашій реалізації GlobalConfig немає поля base_generator_params,
    # можна покласти ці значення у gcfg.generator_params — фреймворк їх підхопить.
    gcfg.base_generator_params = {
        "time_step_s": 5.0,
        "time_constants_s": {
            "concentrate_fe_percent": 8.0,
            "tailings_fe_percent": 10.0,
            "concentrate_mass_flow": 5.0,
            "tailings_mass_flow": 7.0,
        },
        "dead_times_s": {
            "concentrate_fe_percent": 20.0,
            "tailings_fe_percent": 25.0,
            "concentrate_mass_flow": 20.0,
            "tailings_mass_flow": 25.0,
        },
        "control_pts": 700,
        "enable_nonlinear": True,
        "nonlinear_config": {
            "concentrate_fe_percent": ("pow", 2.0),
            "concentrate_mass_flow": ("pow", 1.5),
        },
        "noise_level": "medium",
        "use_anomalies": True,
        "anomaly_severity": "medium",
    }

    wcfg = WorkingConfig(metrics=("RMSE", "MAE", "R2"))
    wcfg.add_model("ARX_OLS")
    wcfg.add_model("ARX_Ridge", alpha=1.0)
    wcfg.add_model("ARX_Lasso", alpha=0.01)
    
    return gcfg, wcfg

def run_demo():
    from typing import Any, Dict
    from pathlib import Path
    from testing_framework import TestHarness, MagneticDataFactory

    # 1) Конфіги
    gcfg, wcfg = make_default_configs()

    # 2) Реєстр моделей
    model_registry = get_model_registry()

    # 3) Централізована фабрика + генератор (сумісно з існуючим TestHarness)
    mag_factory = MagneticDataFactory(processed_path="processed.parquet")
    generator_fn = mag_factory.build_generator()

    # 4) TestHarness у поточному інтерфейсі (без змін у TestHarness)
    h = TestHarness(
        data_generator=generator_fn,
        model_registry=model_registry,
        global_config=gcfg,
        working_config=wcfg,
    )

    # 5) Тест: Лінійні моделі на нелінійних даних із аномаліями
    test = LinearOnNonlinearAnomaliesTest(
        data_factory=h.data_factory,  # якщо у твоєму TestHarness атрибуту data_factory немає — напиши, адаптую виклик
        runner=h.runner,
        gcfg=gcfg,
        wcfg=wcfg,
        noise_level="medium",
        anomaly_severity="medium",
    )

    # 6) Запуск і збереження звіту
    report: Dict[str, Any] = h.run_tests([test])
    Path("results").mkdir(exist_ok=True)
    report_path = "results/linear_on_nonlinear_anomalies.json"
    h.save_report(report, report_path)
    print(f"✅ Звіт збережено у {report_path}")

    # 7) Візуалізація
    figs_dir = "results/figs_linear_on_nonlinear_anomalies"
    artifacts = visualize_report(report_path, out_dir=figs_dir)
    print("✅ Візуалізація готова:", artifacts)

    return report

if __name__ == "__main__":
    run_demo()