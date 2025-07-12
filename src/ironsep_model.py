# ironsep_model.py

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from data_gen import DataGenerator
from sklearn.model_selection import train_test_split

class IronSeparationModel:
    """
    Гібридна модель процесу магнітної сепарації з автоматичним перейменуванням стовпців
    """
    
    # Словник для перейменування стовпців
    COLUMN_MAPPING = {
        'feed_fe_percent': 'feed_fe',
        'ore_mass_flow': 'ore_flow',
        'solid_feed_percent': 'solid_feed',
        'concentrate_fe_percent': 'conc_fe',
        'concentrate_mass_flow': 'conc_mass'
    }
    
    def __init__(self, tech_constants: dict):
        """
        Ініціалізація моделі
        :param tech_constants: словник технологічних констант 
            {'tau_fe': float, 'tau_mass': float, 'theta': float}
        """
        self.tech_constants = tech_constants
        self.fe_model = None
        self.mass_model = None
        self.feature_names = None
        self.original_columns = None
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Перейменування стовпців та перевірка наявності необхідних даних"""
        # Створюємо копію даних
        processed_data = data.copy()
        
        # Перейменування стовпців
        rename_dict = {k: v for k, v in self.COLUMN_MAPPING.items() if k in processed_data.columns}
        processed_data.rename(columns=rename_dict, inplace=True)
        
        # Зберігаємо оригінальні назви стовпців
        self.original_columns = list(processed_data.columns)
        
        # Перевірка наявності необхідних стовпців
        required_columns = ['feed_fe', 'ore_flow', 'solid_feed', 'conc_fe', 'conc_mass']
        missing = [col for col in required_columns if col not in processed_data.columns]
        if missing:
            raise ValueError(f"Відсутні обов'язкові стовпці: {missing}")
            
        return processed_data
    
    def fit(self, data: pd.DataFrame):
        """
        Навчання статичних моделей з автоматичним перейменуванням стовпців
        :param data: DataFrame з даними
        """
        # Попередня обробка даних
        processed_data = self.preprocess_data(data)
        
        # Вибір ознак та цільових змінних
        X = processed_data[['feed_fe', 'ore_flow', 'solid_feed']].values
        y_fe = processed_data['conc_fe'].values
        y_mass = processed_data['conc_mass'].values
        
        # Назви ознак для інтерпретації
        self.feature_names = ['feed_fe', 'ore_flow', 'solid_feed']

        # Зберігаємо діапазони ознак для подальшого використання
        self.feature_ranges = {
            'feed_fe': (processed_data['feed_fe'].min(), processed_data['feed_fe'].max()),
            'ore_flow': (processed_data['ore_flow'].min(), processed_data['ore_flow'].max()),
            'solid_feed': (processed_data['solid_feed'].min(), processed_data['solid_feed'].max())
        }
        
        # Поліноміальна модель для Fe (2-й ступінь)
        self.fe_model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=0.1)
        )
        self.fe_model.fit(X, y_fe)
        
        # Поліноміальна модель для маси (2-й ступінь)
        self.mass_model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=0.1)
        )
        self.mass_model.fit(X, y_mass)
        
        # Розрахунок нелінійних коефіцієнтів
        self._calculate_nonlinear_coefs(X, y_fe, y_mass)
        
    def _calculate_nonlinear_coefs(self, X, y_fe, y_mass):
        """Коректний розрахунок степеневих коефіцієнтів"""
        try:
            # Обчислення базових прогнозів
            fe_base = self.fe_model.predict(X)
            mass_base = self.mass_model.predict(X)
            
            # Вибірка тільки реалістичних значень
            valid_mask = (fe_base > 1) & (mass_base > 1) & (y_fe > 1) & (y_mass > 1)
            X_valid = X[valid_mask]
            y_fe_valid = y_fe[valid_mask]
            y_mass_valid = y_mass[valid_mask]
            fe_base_valid = fe_base[valid_mask]
            mass_base_valid = mass_base[valid_mask]
            
            # Розрахунок коефіцієнтів як відношення логарифмів
            with np.errstate(divide='ignore', invalid='ignore'):
                self.alpha_fe = np.median(np.log(y_fe_valid / fe_base_valid) / np.log(X_valid[:, 0]))
                self.alpha_mass = np.median(np.log(y_mass_valid / mass_base_valid) / np.log(X_valid[:, 0]))
            
            # Обмеження коефіцієнтів
            self.alpha_fe = np.clip(self.alpha_fe, -0.1, 0.1)
            self.alpha_mass = np.clip(self.alpha_mass, -0.1, 0.1)
            
        except Exception as e:
            print(f"Помилка розрахунку коефіцієнтів: {e}")
            self.alpha_fe = 0.02
            self.alpha_mass = 0.01
        
    def predict_static(self, X: np.ndarray) -> tuple:
        """Прогноз статичних значень з корекцією"""
        # Базовий прогноз
        conc_fe_base = self.fe_model.predict(X)
        conc_mass_base = self.mass_model.predict(X)
        
        # Степеневі поправочні коефіцієнти
        feed_fe = X[:, 0]
        feed_fe_ref = 35.0  # Опорне значення
        
        # Застосування корекції відносно опорного значення
        fe_correction = np.power(feed_fe / feed_fe_ref, self.alpha_fe, where=feed_fe>0)
        mass_correction = np.power(feed_fe / feed_fe_ref, self.alpha_mass, where=feed_fe>0)
        
        # Кінцевий прогноз
        conc_fe = conc_fe_base * fe_correction
        conc_mass = conc_mass_base * mass_correction
        
        # Фізичні обмеження
        conc_fe = np.clip(conc_fe, 0, 65)  # Максимум 65% для заліза
        conc_mass = np.maximum(conc_mass, 0)
        
        return conc_fe, conc_mass
    
    def predict_dynamic(self, X: np.ndarray, t: float) -> tuple:
        """Прогноз з урахуванням динаміки"""
        if t < 0:
            raise ValueError("Час t не може бути від'ємним")
        
        # Статичний прогноз
        fe_static, mass_static = self.predict_static(X)
        
        # Застосування динаміки
        t_corrected = max(t - self.tech_constants['theta'], 0)
        
        # Коефіцієнти динаміки
        fe_dynamic_factor = 1 - np.exp(-t_corrected / self.tech_constants['tau_fe'])
        mass_dynamic_factor = 1 - np.exp(-t_corrected / self.tech_constants['tau_mass'])
        
        # Динамічний прогноз
        fe_dynamic = fe_static * fe_dynamic_factor
        mass_dynamic = mass_static * mass_dynamic_factor
        
        # Фізичні обмеження
        fe_dynamic = np.clip(fe_dynamic, 0, 65)
        mass_dynamic = np.maximum(mass_dynamic, 0)
        
        return fe_dynamic, mass_dynamic
    
    def get_model_equation(self) -> dict:
        """Отримання рівнянь моделі у текстовому вигляді"""
        # Для простоти демонструємо тільки для Fe
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        poly_features.fit(np.zeros((1, len(self.feature_names))))  # Фіктивне навчання
        feature_names_ext = poly_features.get_feature_names_out(self.feature_names)
        
        # Коефіцієнти моделі
        ridge = self.fe_model.named_steps['ridge']
        coefs = ridge.coef_
        intercept = ridge.intercept_
        
        # Формування рівняння
        equation = f"conc_fe = {intercept:.4f}"
        for name, coef in zip(feature_names_ext, coefs):
            if abs(coef) > 1e-3:
                equation += f" + {coef:.4f}*{name}"
        
        # Умовне додавання степеневого множника (корекція 2)
        if abs(self.alpha_fe) > 1e-3:
            equation += f" * (feed_fe)^{self.alpha_fe:.3f}"
        
        return {
            'fe_equation': equation,
            'tau_fe': self.tech_constants['tau_fe'],
            'tau_mass': self.tech_constants['tau_mass']
        }
    
    def save_report(self, filename: str):
        """Збереження звіту у текстовому файлі"""
        with open(filename, 'w') as f:
            model_eq = self.get_model_equation()
            f.write("МАТЕМАТИЧНА МОДЕЛЬ ПРОЦЕСУ МАГНІТНОЇ СЕПАРАЦІЇ\n\n")
            f.write("Статична модель концентрації Fe:\n")
            f.write(model_eq['fe_equation'] + "\n\n")
            f.write("Динамічні параметри:\n")
            f.write(f"  τ_fe = {model_eq['tau_fe']} сек\n")
            f.write(f"  τ_mass = {model_eq['tau_mass']} сек\n")
            f.write(f"  θ = {self.tech_constants['theta']} сек\n")
            
            # Додаткова інформація про стовпці
            f.write("\nВикористані стовпці даних:\n")
            f.write(f"  Оригінальні назви: {self.original_columns}\n")
            f.write(f"  Внутрішні назви: {list(self.COLUMN_MAPPING.values())}\n")
            
if __name__ == '__main__':

    # 1. Підготовка даних
    try:
        data = pd.read_parquet('processed.parquet')
    except FileNotFoundError:
        print("Помилка: файл 'processed.parquet' не знайдено.")
        exit()
           
    true_gen = DataGenerator(
        reference_df=data,
        ore_flow_var_pct=3.0, 
        seed=42
        )
    data = true_gen.generate(T=2000, control_pts=200, n_neighbors=5)
   
    X = data[['feed_fe_percent', 'ore_mass_flow', 'solid_feed_percent']].values
    y_fe = data['concentrate_fe_percent'].values
    y_mass = data['concentrate_mass_flow'].values
    
    # 2. Технологічні константи
    tech_const = {
        'tau_fe': 8.2,
        'tau_mass': 5.7,
        'theta': 18.5
    }
    
    # 3. Створення та навчання моделі
    model = IronSeparationModel(tech_constants=tech_const)
    model.fit(data)
    
    # 4. Генерація тестових даних у діапазонах
    def generate_test_data(model: IronSeparationModel, n_samples: int = 1) -> pd.DataFrame:
        """
        Генерує тестові дані в межах мінімальних і максимальних значень ознак
        :param model: навчена модель
        :param n_samples: кількість тестових зразків
        :return: DataFrame з тестовими даними
        """
        # Перевірка, чи модель навчена
        if model.feature_ranges is None:
            raise RuntimeError("Модель не навчена. Спочатку викличіть метод fit().")
        
        # Генерація випадкових значень у діапазонах
        test_data = {}
        for feature, (min_val, max_val) in model.feature_ranges.items():
            test_data[feature] = np.random.uniform(min_val, max_val, n_samples)
        
        return pd.DataFrame(test_data)
    
    test_data_internal = generate_test_data(model, n_samples=5)
    print("Згенеровані тестові дані (внутрішній формат):")
    print(test_data_internal)
    
    # 5. Перетворення у зовнішній формат (якщо потрібно)
    def convert_to_external_format(df: pd.DataFrame) -> pd.DataFrame:
        """Перетворює DataFrame з внутрішніми назвами у зовнішній формат"""
        reverse_mapping = {v: k for k, v in model.COLUMN_MAPPING.items()}
        return df.rename(columns=reverse_mapping)
    
    test_data_external = convert_to_external_format(test_data_internal)
    print("\nТестові дані у зовнішньому форматі:")
    print(test_data_external)
    
    # 6. Прогнозування для кожної тестової точки
    for i in range(len(test_data_internal)):
        X_test = test_data_internal.iloc[i:i+1].values
        
        # Прогнозування
        fe_static, mass_static = model.predict_static(X_test)
        fe_dynamic, mass_dynamic = model.predict_dynamic(X_test, t=50.0)
        
        print(f"\nТест #{i+1}:")
        print(f"Вхідні дані: feed_fe={X_test[0,0]:.2f}, ore_flow={X_test[0,1]:.2f}, solid_feed={X_test[0,2]:.2f}")
        print(f"Статичний прогноз: conc_fe={fe_static[0]:.2f}%, mass={mass_static[0]:.2f} т/год")
        print(f"Динамічний прогноз (t=50с): conc_fe={fe_dynamic[0]:.2f}%, mass={mass_dynamic[0]:.2f} т/год")