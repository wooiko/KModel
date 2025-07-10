import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from collections import deque

class DataGenerator:

    ERROR_PERCENTS_NULL = {
        'feed_fe_percent': 0.0,
        'solid_feed_percent': 0.0,
        'concentrate_fe_percent': 0.0,
        'concentrate_mass_flow': 0.0,
        'tailings_fe_percent': 0.0,
        'tailings_mass_flow': 0.0,
        'ore_mass_flow': 0.0
    }

    ERROR_PERCENTS_LOW = {
        'feed_fe_percent': 0.5,
        'solid_feed_percent': 1.0,
        'concentrate_fe_percent': 0.3,
        'concentrate_mass_flow': 2.0,
        'tailings_fe_percent': 0.4,
        'tailings_mass_flow': 2.5,
        'ore_mass_flow': 1.0
    }

    ERROR_PERCENTS_MEDIUM = {
        'feed_fe_percent': 0.75,
        'solid_feed_percent': 1.5,
        'concentrate_fe_percent': 0.5,
        'concentrate_mass_flow': 3.5,
        'tailings_fe_percent': 0.65,
        'tailings_mass_flow': 4.0,
        'ore_mass_flow': 2.5
    }

    ERROR_PERCENTS_HIGH = {
        'feed_fe_percent': 1.0,
        'solid_feed_percent': 2.0,
        'concentrate_fe_percent': 0.7,
        'concentrate_mass_flow': 5.0,
        'tailings_fe_percent': 0.9,
        'tailings_mass_flow': 5.5,
        'ore_mass_flow': 4.0
    }

    # Співвідношення компонентів стандартного відхилення шуму (абс, відн, низькочастотний)
    # Ми використовуємо (абс, відн) для розрахунку σ(t) = E_base_factor * (ratio_abs * P_mean + ratio_rel * P(t))
    ERROR_RATIOS = {
        'feed_fe_percent': (0.7, 0.3, 0.0),
        'solid_feed_percent': (0.4, 0.6, 0.3),
        'ore_mass_flow': (0.3, 0.7, 0.2),
        'concentrate_fe_percent': (0.7, 0.3, 0.0),
        'tailings_fe_percent': (0.6, 0.4, 0.0),
        'concentrate_mass_flow': (0.3, 0.7, 0.1),
        'tailings_mass_flow': (0.3, 0.7, 0.1)
    }

    # Мапінг назв рівнів шуму до словників похибок
    _noise_levels_map = {
        'none': ERROR_PERCENTS_NULL,
        'low': ERROR_PERCENTS_LOW,
        'medium': ERROR_PERCENTS_MEDIUM,
        'high': ERROR_PERCENTS_HIGH
    }
    """
    Генератор синтетичних даних для системи прогнозуючого керування:
    - формує згладжені часові ряди вхідних сигналів
    - навчає kNN-модель на первинному наборі даних
    - прогнозує вихідні параметри
    - коригує їх за законом балансу маси та заліза
    - дозволяє формувати зразки з лагами
    """
    
    def __init__(self,
                     reference_df: pd.DataFrame,
                     ore_flow_var_pct: float = 3.0,
                     seed: int = 0,
                     time_step_s: float = 5.0,
                     # === МОДИФІКОВАНІ ПАРАМЕТРИ ===
                     # Замість скалярів тепер використовуються словники.
                     # Ключ - назва колонки, значення - параметр в секундах.
                     time_constants_s: dict = None,
                     dead_times_s: dict = None,
                     # ===============================
                     true_model_type: str = 'rf'
                    ):
            self.rng = np.random.default_rng(seed)
            self.seed = seed
    
            # 1) Зберігаємо базові параметри часу
            self.time_step_s = time_step_s
    
            # 2) Вхідні й вихідні параметри (визначені тут для використання нижче)
            self.input_cols = ['feed_fe_percent',
                               'ore_mass_flow',
                               'solid_feed_percent']
            self.output_cols = [
                'concentrate_fe_percent',
                'tailings_fe_percent',
                'concentrate_mass_flow',
                'tailings_mass_flow'
            ]
    
            # 3) === ЛОГІКА ОБРОБКИ ІНДИВІДУАЛЬНИХ ЗАтримок ===
            # Встановлюємо значення за замовчуванням, якщо словники не передано
            default_tc = {'default': 8.0}
            default_dt = {'default': 20.0}
    
            self.time_constants_s = time_constants_s if time_constants_s is not None else default_tc
            self.dead_times_s = dead_times_s if dead_times_s is not None else default_dt
    
            # Обчислюємо параметри динаміки (кроки затримки та альфа-коефіцієнти)
            # для кожного вихідного сигналу окремо і зберігаємо їх у словниках.
            # Метод .get() безпечно повертає значення 'default', якщо для конкретного
            # виходу немає ключа в переданому словнику.
            self.dead_time_steps = {
                col: round(self.dead_times_s.get(col, self.dead_times_s.get('default', 20.0)) / self.time_step_s)
                for col in self.output_cols
            }
            self.lag_filter_alphas = {
                col: 1.0 / ((self.time_constants_s.get(col, self.time_constants_s.get('default', 8.0)) / self.time_step_s) + 1.0)
                for col in self.output_cols
            }
            # =======================================================
    
            # 4) Зберігаємо оригінальні дані та вставляємо ore_mass_flow, якщо немає
            self.original_dataset = reference_df.copy().reset_index(drop=True)
            if 'ore_mass_flow' not in self.original_dataset.columns:
                if 'feed_fe_percent' in self.original_dataset.columns:
                    pos = list(self.original_dataset.columns).index('feed_fe_percent') + 1
                else:
                    pos = len(self.original_dataset.columns)
                self.original_dataset.insert(loc=pos,
                                             column='ore_mass_flow',
                                             value=100.0)
    
            # 5) Використовуємо оновлені дані для навчання моделі
            self.ref = self.original_dataset.copy()
    
            # 6) Навчаємо модель "реального процесу"
            self.true_model_type = true_model_type
            self._fit_model()
    
            # 7) Діапазони генерації вхідних сигналів
            base_flow = self.ref['ore_mass_flow'].mean()
            dv = base_flow * ore_flow_var_pct / 100.0
            self.ranges = {
                'ore_mass_flow':      (base_flow - dv, base_flow + dv),
                'feed_fe_percent':    (self.ref['feed_fe_percent'].min(),
                                       self.ref['feed_fe_percent'].max()),
                'solid_feed_percent': (self.ref['solid_feed_percent'].min(),
                                       self.ref['solid_feed_percent'].max())
            }
            self._input_ranges = dict(self.ranges)
    
            # 8) Мін/макс для clamp після шуму/аномалій
            self._parameter_ranges_for_clamping = {}
            for col in self.input_cols + self.output_cols:
                col_series = self.original_dataset[col]
                self._parameter_ranges_for_clamping[col] = (
                    col_series.min(),
                    col_series.max()
                )

    def _fit_model(self, n_neighbors: int = 5):
        """
        Навчає модель "реального процесу" (plant model).
        Може бути або KNeighborsRegressor, або RandomForestRegressor.
        """
        X = self.ref[self.input_cols]
        y = self.ref[self.output_cols]
        
        if self.true_model_type == 'knn':
            print("INFO: 'true_gen' (plant) використовує модель KNeighborsRegressor.")
            self._model = KNeighborsRegressor(n_neighbors=n_neighbors)
            
        elif self.true_model_type == 'rf':
            print("INFO: 'true_gen' (plant) використовує модель RandomForestRegressor.")
            self._model = RandomForestRegressor(
                n_estimators=100,      # Стандартна кількість дерев
                random_state=self.seed # Для відтворюваності
            )
        else:
            raise ValueError(f"Невідомий тип моделі для true_gen: '{self.true_model_type}'")
            
        self._model.fit(X, y)

    def _generate_inputs(self, T: int, control_pts: int) -> pd.DataFrame:
        t = np.arange(T)
        df = pd.DataFrame(index=t)
        for c in self.input_cols:
            lo, hi = self.ranges[c]
            # обрати контрольні точки
            pts = sorted(np.concatenate(([0],
                         self.rng.choice(np.arange(1,T-1), max(0,control_pts-2), replace=False),
                         [T-1])))
            vals = self.rng.uniform(lo, hi, len(pts))
            kind = 'cubic' if len(pts)>=4 else 'quadratic' if len(pts)==3 else 'linear'
            f = interp1d(pts, vals, kind=kind, fill_value='extrapolate')
            v = f(t)
            # масштабуємо до [lo,hi]
            v = lo + (v-v.min())*(hi-lo)/(v.max()-v.min() + 1e-9)
            df[c] = v
        return df

    def _predict_outputs(self, inp: pd.DataFrame) -> pd.DataFrame:
        y = self._model.predict(inp[self.input_cols])
        return pd.DataFrame(y, index=inp.index, columns=self.output_cols)

    def _apply_mass_balance(self, inp: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        df = out.copy()
        # 1. масовий баланс: ore = conc + tail
        total_out = df.concentrate_mass_flow + df.tailings_mass_flow
        factor = inp.ore_mass_flow / (total_out.replace(0,1e-9))
        df.concentrate_mass_flow *= factor
        df.tailings_mass_flow   *= factor
        # 2. баланс Fe
        m_in_fe = inp.ore_mass_flow * inp.feed_fe_percent/100
        m_conc_fe = df.concentrate_mass_flow * df.concentrate_fe_percent/100
        m_tail_fe = m_in_fe - m_conc_fe
        # tailings_fe_percent = m_tail_fe / tail_mass *100
        df.tailings_fe_percent = m_tail_fe / (df.tailings_mass_flow.replace(0,1e-9)) * 100
        # клампінг на [0,100]
        df.tailings_fe_percent = df.tailings_fe_percent.clip(0,100)
        return df

    def _apply_dynamics(self, df_ideal: pd.DataFrame) -> pd.DataFrame:
        df_dynamic = df_ideal.copy()
    
        for col in self.output_cols:
            ideal_output = df_ideal[col].values
            dynamic_output = np.zeros_like(ideal_output)
    
            # <<< ВИКОРИСТОВУЄМО ІНДИВІДУАЛЬНІ ПАРАМЕТРИ >>>
            current_dead_time_steps = self.dead_time_steps[col]
            current_alpha = self.lag_filter_alphas[col]
    
            delayed_output = pd.Series(ideal_output).shift(current_dead_time_steps).bfill().values
    
            dynamic_output[0] = delayed_output[0] 
            for t in range(1, len(ideal_output)):
                dynamic_output[t] = (current_alpha * delayed_output[t] +
                                     (1 - current_alpha) * dynamic_output[t-1])
    
            df_dynamic[col] = dynamic_output
    
        return df_dynamic

    def _derive(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['mass_pull_percent'] = df.concentrate_mass_flow / df.ore_mass_flow * 100
        m_in_fe  = df.ore_mass_flow * df.feed_fe_percent/100
        m_conc_fe= df.concentrate_mass_flow * df.concentrate_fe_percent/100
        df['fe_recovery_percent'] = (m_conc_fe / m_in_fe) * 100
        return df
    
    def add_noise(self, data: pd.DataFrame, noise_level: str='none') -> pd.DataFrame:
        df = data.copy()
        if noise_level not in self._noise_levels_map:
            return df
        errs = self._noise_levels_map[noise_level]
        for col, ratio_tuple in self.ERROR_RATIOS.items():
            if col not in df or errs.get(col, 0) == 0:
                continue
            r_abs, r_rel = ratio_tuple[0], ratio_tuple[1]  # беремо лише 2 перші значення
            base_pct = errs[col] / 100
            param_mean = self.original_dataset[col].mean()
            sigma = base_pct * (r_abs * param_mean + r_rel * df[col])
            sigma = sigma.clip(lower=1e-9)
            noise = self.rng.normal(loc=0, scale=sigma)
            df[col] = df[col] + noise
            # clamp по діапазону
            if col == 'ore_mass_flow':
                lo, hi = self._input_ranges[col]
            else:
                lo, hi = self._parameter_ranges_for_clamping[col]
            df[col] = df[col].clip(lo, hi)
        return df

    def generate_anomalies(self, df: pd.DataFrame, anomaly_config: dict) -> pd.DataFrame:
        df = df.copy()
        for param, anomalies in anomaly_config.items():
            for anom in anomalies:
                start = anom['start']
                dur   = anom['duration']
                typ   = anom['type']
                # обрізати індекси, щоб не вийти за межі
                idx = list(range(start, min(start + dur, len(df))))
                dur_act = len(idx)
                if dur_act == 0:
                    continue
    
                # для spike і freeze нічого не змінюємо
                if typ == 'spike':
                    mag = anom['magnitude'] * (df[param].max() - df[param].min())
                    df.loc[idx, param] += mag
                elif typ == 'freeze':
                    # нічого не змінюємо — значення "зависають"
                    df.loc[idx, param] = df.loc[idx[0], param]
    
                else:
                    # обчислюємо абсолютний діапазон для drop/drift
                    mag_range = anom['magnitude'] * (df[param].max() - df[param].min())
    
                    if typ == 'drift':
                        # лінійний дрейф від 0 до mag_range, але довжиною dur_act
                        drift = np.linspace(0, mag_range, dur_act)
                        df.loc[idx, param] += drift
    
                    elif typ == 'drop':
                        # різке падіння — постійний “спад” на весь інтервал
                        drop = np.full(dur_act, mag_range)
                        df.loc[idx, param] -= drop
    
        return df

    def generate(self, T:int, control_pts:int, n_neighbors:int=5,
                 noise_level:str='none', anomaly_config:dict=None) -> pd.DataFrame:
        # self._fit_knn(n_neighbors)
        inp = self._generate_inputs(T, control_pts)
        out = self._predict_outputs(inp)
        
        # Розраховуємо ідеальні, збалансовані виходи
        out_balanced = self._apply_mass_balance(inp, out)
        
        # >>> ЗАСТОСОВУЄМО ДИНАМІКУ ПРОЦЕСУ <<<
        # Тепер out_dynamic - це те, що ми б "виміряли" на реальному заводі
        out_dynamic = self._apply_dynamics(out_balanced)
        
        # Об'єднуємо входи з новими, динамічними виходами
        full = pd.concat([inp, out_dynamic], axis=1)
        full = self._derive(full) # Перераховуємо mass pull і recovery
        
        # Шум і аномалії додаємо до вже динамічного процесу
        if noise_level!='none':
            full = self.add_noise(full, noise_level)
        if anomaly_config:
            full = self.generate_anomalies(full, anomaly_config)
            
        return full

    def generate_nonlinear_variant(self, 
                                     base_df: pd.DataFrame,
                                     non_linear_factors: dict,
                                     noise_level: str = 'none', 
                                     anomaly_config: dict = None
                                    ) -> pd.DataFrame:
            """
            Створює новий датасет на основі існуючого, посилюючи нелінійну залежність
            між входами та виходами, зберігаючи при цьому закон балансу мас.
    
            Args:
                base_df (pd.DataFrame): Датафрейм, згенерований методом generate().
                                        Використовуються тільки вхідні сигнали з нього.
                non_linear_factors (dict): Словник для керування нелінійністю.
                                           Приклад: 
                                           {
                                             'concentrate_fe_percent': ('pow', 1.2), # звести в ступінь 1.2
                                             'concentrate_mass_flow': ('pow', 0.8) # звести в ступінь 0.8
                                           }
                                           Підтримувані типи: 'pow'.
                noise_level (str): Рівень шуму для додавання до фінального датасету.
                anomaly_config (dict): Конфігурація аномалій для додавання.
    
            Returns:
                pd.DataFrame: Новий датафрейм з посиленою нелінійністю.
            """
            print("INFO: Generating a dataset variant with enhanced non-linearity...")
            
            # 1. Використовуємо вхідні сигнали з базового датасету
            inp = base_df[self.input_cols].copy()
    
            # 2. Отримуємо ідеальні (незбалансовані) виходи від внутрішньої моделі
            out_ideal = self._predict_outputs(inp)
            
            # 3. Ключовий крок: застосовуємо нелінійні перетворення
            out_nl = out_ideal.copy()
            for col, (transform_type, factor) in non_linear_factors.items():
                if col not in out_nl.columns:
                    print(f"WARNING: Column '{col}' for non-linear transform not found in outputs. Skipping.")
                    continue
                
                if transform_type == 'pow':
                    # Зведення в ступінь - простий спосіб додати нелінійність
                    # ( опуклість для > 1, увігнутість для < 1)
                    series = out_nl[col]
                    # Нормалізуємо до [0, 1] щоб уникнути великих значень, застосовуємо степінь, повертаємо до масштабу
                    min_val, max_val = series.min(), series.max()
                    series_norm = (series - min_val) / (max_val - min_val + 1e-9)
                    series_transformed = np.power(series_norm, factor)
                    out_nl[col] = series_transformed * (max_val - min_val) + min_val
                else:
                    print(f"WARNING: Unknown transform type '{transform_type}'. Skipping column '{col}'.")
    
            # 4. Застосовуємо баланс мас, використовуючи оригінальні входи та модифіковані виходи
            out_balanced_nl = self._apply_mass_balance(inp, out_nl)
    
            # 5. Застосовуємо динаміку процесу до нових збалансованих виходів
            out_dynamic_nl = self._apply_dynamics(out_balanced_nl)
    
            # 6. Збираємо фінальний датафрейм
            full_nl = pd.concat([inp, out_dynamic_nl], axis=1)
            full_nl = self._derive(full_nl)  # Перераховуємо mass pull і recovery
    
            # 7. Додаємо шум та аномалії, якщо потрібно
            if noise_level != 'none':
                full_nl = self.add_noise(full_nl, noise_level)
            if anomaly_config:
                full_nl = self.generate_anomalies(full_nl, anomaly_config)
                
            print("INFO: Non-linear variant generated successfully.")
            return full_nl
    @staticmethod
    def create_lagged_dataset(df: pd.DataFrame,
                              lags: int = 2
                             ) -> (np.ndarray, np.ndarray):
        """
        Формує X, Y для навчання прогнозної моделі з лагами.
        X має форму (N−lags)×[3×(lags+1)], Y — (N−lags)×4.
        """
        vals = df.reset_index(drop=True)
        N = len(vals)
        X, Y = [], []
        for i in range(lags, N):
            past = vals.loc[i-lags:i, ['feed_fe_percent','ore_mass_flow','solid_feed_percent']].values.flatten()
            X.append(past)
            Y.append(vals.loc[i, ['concentrate_fe_percent','tailings_fe_percent',
                                  'concentrate_mass_flow','tailings_mass_flow']].values)
        return np.array(X), np.array(Y)
    
    @staticmethod
    def generate_anomaly_config(
        N_data: int,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42
    ):
        """
        Генерує конфігураційний словник аномалій для різних сегментів даних.
        Використовує ізольований генератор випадкових чисел для відтворюваності.
        """
        # Створюємо локальний, ізольований генератор для уникнення побічних ефектів
        rng = np.random.default_rng(seed)

        # Нормалізуємо частки, щоб сума була 1
        total_frac = train_frac + val_frac + test_frac
        train_frac /= total_frac
        val_frac /= total_frac
        test_frac /= total_frac

        train_end = int(train_frac * N_data)
        val_end = train_end + int(val_frac * N_data)

        segments = {
            'train': (0, train_end),
            'val': (train_end, val_end),
            'test': (val_end, N_data)
        }

        params = [
            'ore_mass_flow',
            'feed_fe_percent',
            'solid_feed_percent',
            'concentrate_fe_percent',
            'tailings_fe_percent',
            'concentrate_mass_flow',
            'tailings_mass_flow'
        ]

        anomaly_types = ['spike', 'drift', 'drop', 'freeze']
        durations_map = {
            'spike': 1,
            'drift': 30,
            'drop': 30,
            'freeze': 20
        }

        config = {}

        for param in params:
            config[param] = []
            for seg_name, (seg_start, seg_end) in segments.items():
                seg_len = seg_end - seg_start
                if seg_len <= 0:
                    continue

                # Використовуємо локальний генератор rng
                a_type = rng.choice(anomaly_types)
                duration = durations_map[a_type]
                if duration > seg_len:
                    duration = seg_len  # обрізаємо до довжини сегменту
                    if duration == 0:
                        continue

                start_low = seg_start
                start_high = seg_end - duration
                if start_high < start_low:
                    start_high = start_low

                # Використовуємо локальний генератор rng
                start = rng.integers(low=start_low, high=start_high, endpoint=True)

                if a_type == 'freeze':
                    anomaly = {
                        'start': start,
                        'duration': duration,
                        'type': a_type
                    }
                else:
                    # Використовуємо локальний генератор rng
                    sign = rng.choice([1, -1])
                    magnitude = sign * rng.uniform(0.1, 0.3)
                    anomaly = {
                        'start': start,
                        'duration': duration,
                        'magnitude': magnitude,
                        'type': a_type
                    }
                config[param].append(anomaly)

        return config
    
class StatefulPlantMixin:
    """
    Додає до DataGenerator інкрементальний метод step().
    Використовує СЛОВНИК FIFO-буферів для індивідуальних
    dead-time та first-order-lag для кожного виходу.
    """
    def reset_state(self, init_history: np.ndarray):
        """
        init_history shape: (L+1, 3) — останні L+1 векторів [feed_fe, ore_flow, u]
        Ініціалізує словники з попередніми виходами та буферами затримок.
        """
        feed_fe, ore_flow, u_last = init_history[-1]
        
        # _predict_raw повертає (inp_df, y_bal_df)
        inp_df_init, y_bal_df_init = self._predict_raw(feed_fe, ore_flow, u_last)
        
        # Створюємо словник для зберігання попередніх значень кожного виходу
        self._prev_output = {
            col: y_bal_df_init[[col]].copy() for col in self.output_cols
        }
        
        # Створюємо СЛОВНИК з FIFO-буферами, по одному для кожного виходу
        self._delay_fifo = {}
        for col in self.output_cols:
            # maxlen тепер індивідуальний для кожного буфера
            max_len = self.dead_time_steps[col]
            # Початкове значення для заповнення буфера
            initial_value = y_bal_df_init[[col]].copy()
            
            # Якщо max_len > 0, створюємо та заповнюємо буфер
            if max_len > 0:
                self._delay_fifo[col] = deque([initial_value] * max_len, maxlen=max_len)
            # Якщо затримки немає, створюємо порожній буфер
            else:
                 self._delay_fifo[col] = deque(maxlen=0)


    def _predict_raw(self, feed_fe, ore_flow, u):
        """
        Обчислює ідеальні (без затримки та інерції) виходи.
        """
        inp = pd.DataFrame([[feed_fe, ore_flow, u]],
                           columns=self.input_cols)
        y_ideal = self._predict_outputs(inp)
        y_bal   = self._apply_mass_balance(inp, y_ideal)
        return inp, y_bal

    def _ideal_to_dynamic(self, inp_df, y_bal_df):
        """
        Застосовує індивідуальні dead-time та first-order-lag до ідеальних виходів.
        """
        dyn_output_df = pd.DataFrame(index=y_bal_df.index)

        # Ітеруємо по кожному вихідному сигналу
        for col in self.output_cols:
            # 1. Dead-time (робота з індивідуальним буфером)
            fifo = self._delay_fifo[col]
            ideal_output_col = y_bal_df[[col]] # DataFrame з однією колонкою

            if fifo.maxlen > 0:
                delayed_output_for_this_step = fifo.popleft()
                fifo.append(ideal_output_col)
            else:
                # Якщо затримки немає, використовуємо поточне ідеальне значення
                delayed_output_for_this_step = ideal_output_col

            # 2. First-order lag (з індивідуальним alpha)
            alpha = self.lag_filter_alphas[col]
            prev_out_col = self._prev_output[col]
            
            # Розрахунок нового динамічного значення
            dyn_val = (alpha * delayed_output_for_this_step.iloc[0, 0] + 
                       (1 - alpha) * prev_out_col.iloc[0, 0])
            
            dyn_output_df[col] = [dyn_val]
            
            # Оновлюємо попередній вихід для НАСТУПНОГО кроку
            self._prev_output[col].iloc[0, 0] = dyn_val

        # Комбінуємо входи та динамічні виходи, потім обчислюємо похідні величини
        full = pd.concat([inp_df, dyn_output_df], axis=1)
        full = self._derive(full)
        return full

    def step(self, feed_fe, ore_flow, u):
        """
        Повертає DataFrame з одним рядком виміряних величин на поточному кроці.
        Приймає поточні вхідні змінні та керуючу дію.
        """
        inp_df, y_bal_df = self._predict_raw(feed_fe, ore_flow, u)
        return self._ideal_to_dynamic(inp_df, y_bal_df)

# Наслідуємо від обох класів, щоб отримати функціональність Plant (динаміку)
# та DataGenerator (базові моделі та масовий баланс)
class StatefulDataGenerator(StatefulPlantMixin, DataGenerator):
    pass