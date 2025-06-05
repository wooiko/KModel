# noise_constants.py

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

# Середні значення y для кожного каналу (має бути задано під ваш процес)
Y_MEANS = {
    'feed_fe_percent': 25.0,
    'solid_feed_percent': 30.0,
    'concentrate_fe_percent': 50.0,
    'concentrate_mass_flow': 100.0,
    'tailings_fe_percent': 10.0,
    'tailings_mass_flow': 80.0,
    'ore_mass_flow': 120.0
}