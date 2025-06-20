import numpy as np
import pandas as pd
import random

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
                 time_constant_s: float = 8.0,
                 dead_time_s: float = 20.0,
                 # >>> НОВИЙ ПАРАМЕТР <<<
                 true_model_type: str = 'rf' # 'rf' або 'knn'
                ):
        self.rng = np.random.default_rng(seed)
        self.seed = seed # Зберігаємо для відтворюваності RandomForest

        self.time_step_s = time_step_s
        self.dead_time_steps = round(dead_time_s / time_step_s)
        time_constant_steps = time_constant_s / time_step_s
        self.lag_filter_alpha = 1.0 / (time_constant_steps + 1.0)
        
        # 2) Зберігаємо оригінальні дані та вставляємо ore_mass_flow, якщо немає
        self.original_dataset = reference_df.copy().reset_index(drop=True)
        if 'ore_mass_flow' not in self.original_dataset.columns:
            if 'feed_fe_percent' in self.original_dataset.columns:
                pos = list(self.original_dataset.columns).index('feed_fe_percent') + 1
            else:
                pos = len(self.original_dataset.columns)
            self.original_dataset.insert(loc=pos,
                                         column='ore_mass_flow',
                                         value=100.0)

        # 3) Використовуємо оновлені дані для навчання kNN
        self.ref = self.original_dataset.copy()

        # 4) Вхідні й вихідні параметри
        self.input_cols = ['feed_fe_percent',
                           'ore_mass_flow',
                           'solid_feed_percent']
        self.output_cols = [
            'concentrate_fe_percent',
            'tailings_fe_percent',
            'concentrate_mass_flow',
            'tailings_mass_flow'
        ]

        # 5) Навчаємо kNN на чистих даних
        self.true_model_type = true_model_type
        self._fit_model() # <<< ЗАМІНЮЄМО ВИКЛИК
        
        # 6) Діапазони генерації вхідних сигналів
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

        # 7) Мін/макс для clamp після шуму/аномалій
        self._parameter_ranges_for_clamping = {}
        for col in self.input_cols + self.output_cols:
            col_series = self.original_dataset[col]
            self._parameter_ranges_for_clamping[col] = (
                col_series.min(),
                col_series.max()
            )
        
    # def _fit_knn(self, n_neighbors: int = 5):
    #     X = self.ref[self.input_cols]
    #     y = self.ref[self.output_cols]
    #     self._model = KNeighborsRegressor(n_neighbors=n_neighbors)
    #     self._model.fit(X, y)

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
        """
        Застосовує динаміку першого порядку із запізненням до виходів.
        df_ideal: датафрейм з "ідеальними" миттєвими виходами.
        """
        df_dynamic = df_ideal.copy()
        
        # Застосовуємо динаміку тільки до вихідних колонок
        for col in self.output_cols:
            ideal_output = df_ideal[col].values
            dynamic_output = np.zeros_like(ideal_output)
            
            # 1. Застосовуємо запізнення (dead time)
            # Вихід в момент t залежить від ідеального виходу в момент (t - dead_time)
            delayed_output = pd.Series(ideal_output).shift(self.dead_time_steps).bfill().values
            
            # 2. Застосовуємо інерційність (first-order lag)
            # y(t) = a * y_ideal(t) + (1-a) * y(t-1)
            # Початкове значення = перше значення після затримки
            dynamic_output[0] = delayed_output[0] 
            for t in range(1, len(ideal_output)):
                dynamic_output[t] = (self.lag_filter_alpha * delayed_output[t] +
                                     (1 - self.lag_filter_alpha) * dynamic_output[t-1])
            
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
        random.seed(seed)
    
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
    
                a_type = random.choice(anomaly_types)
                duration = durations_map[a_type]
                if duration > seg_len:
                    duration = seg_len  # обрізаємо до довжини сегменту
                    if duration == 0:
                        continue
    
                start_low = seg_start
                start_high = seg_end - duration
                if start_high < start_low:
                    start_high = start_low
    
                start = random.randint(start_low, start_high)
    
                if a_type == 'freeze':
                    anomaly = {
                        'start': start,
                        'duration': duration,
                        'type': a_type
                    }
                else:
                    sign = random.choice([1, -1])
                    magnitude = sign * random.uniform(0.1, 0.3)
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
    Використовує FIFO-буфер для dead-time та first-order-lag.
    """
    def reset_state(self, init_history: np.ndarray):
        """
        init_history shape: (L+1, 3) — останні L+1 векторів [feed_fe, ore_flow, u]
        Буфер запізнення заповнюємо останнім елементом.
        """
        # Переконайтесь, що self.dead_time_steps та self.lag_filter_alpha 
        # існують (ініціалізуються в DataGenerator або тут)
        
        feed_fe, ore_flow, u_last = init_history[-1]
        
        # _predict_raw повертає (inp_df, y_bal_df)
        inp_df_init, y_bal_df_init = self._predict_raw(feed_fe, ore_flow, u_last)
        
        # Це початкове значення для динамічного фільтра (lag)
        # Вибираємо тільки вихідні стовпці для _prev_output
        self._prev_output = y_bal_df_init[self.output_cols].copy() 
        
        # Dead-time FIFO (зберігаємо тільки concentrate/tailing блок)
        # Заповнюємо буфер значеннями, які б пройшли через "мертвий час"
        self._delay_fifo = deque(
            [y_bal_df_init[self.output_cols].copy()] * self.dead_time_steps, # Зберігаємо лише вихідні стовпці
            maxlen=self.dead_time_steps
        )

    # --- приватні допоміжні ---
    def _predict_raw(self, feed_fe, ore_flow, u):
        """
        Обчислює ідеальні (без затримки та інерції) виходи.
        """
        inp = pd.DataFrame([[feed_fe, ore_flow, u]],
                           columns=self.input_cols)
        y_ideal = self._predict_outputs(inp) # Це метод з DataGenerator
        y_bal   = self._apply_mass_balance(inp, y_ideal) # Це метод з DataGenerator
        return inp, y_bal

    def _ideal_to_dynamic(self, inp_df, y_bal_df):
        """
        Застосовує dead-time та first-order-lag до ідеальних виходів.
        """
        # 1. Dead-time
        # Дістаємо найстаріший елемент з FIFO-буфера. Якщо dead_time_steps=0, то це просто поточний y_bal_df.
        delayed_output_for_this_step = self._delay_fifo.popleft() if self.dead_time_steps > 0 else y_bal_df[self.output_cols]
        # Додаємо поточні ідеальні виходи в кінець FIFO-буфера
        self._delay_fifo.append(y_bal_df[self.output_cols].copy()) # Зберігаємо тільки вихідні стовпці

        # 2. First-order lag
        dyn_output = delayed_output_for_this_step.copy()
        alpha = self.lag_filter_alpha # Коефіцієнт інерції
        
        for col in self.output_cols:
            dyn_output[col].iloc[0] = (alpha * delayed_output_for_this_step[col].iloc[0] + 
                                       (1 - alpha) * self._prev_output[col].iloc[0])
        
        self._prev_output = dyn_output.copy() # Оновлюємо попередній вихід для наступного кроку

        # Комбінуємо входи та динамічні виходи, потім обчислюємо похідні величини
        full = pd.concat([inp_df, dyn_output], axis=1) # dyn_output тепер лише вихідні стовпці
        full = self._derive(full) # Метод з DataGenerator, що додає fe_recovery, mass_pull
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