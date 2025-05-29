import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.neighbors import KNeighborsRegressor

class DataGenerator:
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
                 ore_flow_var_pct: float = 3.0):
        # очікуємо в reference_df колонки:
        # 'feed_fe_percent','solid_feed_percent',
        # 'concentrate_fe_percent','tailings_fe_percent',
        # 'concentrate_mass_flow','tailings_mass_flow'
        self.ref = reference_df.copy()
        # якщо немає ore_mass_flow, вважаємо константою 100
        if 'ore_mass_flow' not in self.ref:
            self.ref['ore_mass_flow'] = 100.0
        self.input_cols  = ['feed_fe_percent','ore_mass_flow','solid_feed_percent']
        self.output_cols = ['concentrate_fe_percent','tailings_fe_percent',
                            'concentrate_mass_flow','tailings_mass_flow']
        self._fit_knn()
        # діапазони для генерації
        base = self.ref['ore_mass_flow'].mean()
        dv = base * ore_flow_var_pct/100
        self.ranges = {
            'ore_mass_flow': (base-dv, base+dv),
            'feed_fe_percent': (self.ref.feed_fe_percent.min(), self.ref.feed_fe_percent.max()),
            'solid_feed_percent': (self.ref.solid_feed_percent.min(), self.ref.solid_feed_percent.max()),
        }

    def _fit_knn(self, n_neighbors: int = 5):
        X = self.ref[self.input_cols]
        y = self.ref[self.output_cols]
        self._model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self._model.fit(X, y)

    def _generate_inputs(self, T: int, control_pts: int) -> pd.DataFrame:
        t = np.arange(T)
        df = pd.DataFrame(index=t)
        for c in self.input_cols:
            lo, hi = self.ranges[c]
            # обрати контрольні точки
            pts = sorted(np.concatenate(([0],
                         np.random.choice(np.arange(1,T-1), max(0,control_pts-2), replace=False),
                         [T-1])))
            vals = np.random.uniform(lo, hi, len(pts))
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

    def _derive(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['mass_pull_percent'] = df.concentrate_mass_flow / df.ore_mass_flow * 100
        m_in_fe  = df.ore_mass_flow * df.feed_fe_percent/100
        m_conc_fe= df.concentrate_mass_flow * df.concentrate_fe_percent/100
        df['fe_recovery_percent'] = (m_conc_fe / m_in_fe) * 100
        return df

    def generate(self,
                 T: int,
                 control_pts: int,
                 n_neighbors: int = 5) -> pd.DataFrame:
        """
        Повертає DataFrame з колонками:
        [feed_fe_percent, ore_mass_flow, solid_feed_percent,
         concentrate_fe_percent, tailings_fe_percent,
         concentrate_mass_flow, tailings_mass_flow,
         mass_pull_percent, fe_recovery_percent]
        """
        self._fit_knn(n_neighbors)
        inp = self._generate_inputs(T, control_pts)
        out = self._predict_outputs(inp)
        out = self._apply_mass_balance(inp, out)
        full = pd.concat([inp, out], axis=1)
        full = self._derive(full)
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