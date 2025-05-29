# config.py
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataGenConfig:
    ore_flow_var_pct: float
    N_data: int
    control_pts: int
    n_neighbors: int

@dataclass
class ModelConfig:
    type: str
    alpha: float
    gamma: Optional[float]

@dataclass
class MPCConfig:
    horizon: int
    lag: int
    u_min: float
    u_max: float
    delta_u_max: float
    lambda_obj: float

@dataclass
class SplitConfig:
    train_size: float
    val_size: float
    test_size: float

@dataclass
class SimConfig:
    reference_data_path: str
    output_results_path: str

@dataclass
class Config:
    random_seed: int
    data_gen: DataGenConfig
    model: ModelConfig
    mpc: MPCConfig
    split: SplitConfig
    sim: SimConfig

def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return Config(
        random_seed=cfg["random_seed"],
        data_gen=DataGenConfig(**cfg["data_gen"]),
        model=ModelConfig(**cfg["model"]),
        mpc=MPCConfig(
            horizon=cfg["mpc"]["horizon"],
            lag=cfg["mpc"]["lag"],
            u_min=cfg["mpc"]["u_min"],
            u_max=cfg["mpc"]["u_max"],
            delta_u_max=cfg["mpc"]["delta_u_max"],
            lambda_obj=cfg["mpc"]["lambda"],
        ),
        split=SplitConfig(**cfg["split"]),
        sim=SimConfig(**cfg["sim"])
    )