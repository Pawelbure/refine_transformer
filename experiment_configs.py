# experiment_configs.py
#
# Central place for all hyperparameters:
#   - data simulation
#   - dataset splitting / windowing
#   - KoopmanAE
#   - Transformer
#   - evaluation
#
# Add your own experiment entries to EXPERIMENTS below, e.g.
# "experiment2_2025-12-01", etc.

from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class SimulationConfig:
    G: float
    T_SPAN: Tuple[float, float]
    NUM_STEPS: int
    NUM_TRAJECTORIES: int
    NUM_TRAJ_OOD: int
    PERTURBATION: float


@dataclass
class DatasetConfig:
    SEQ_LEN: int        # a.k.a. LOOKBACK / window length
    HORIZON: int        # prediction horizon for training
    TRAIN_FRAC: float
    VAL_FRAC: float


@dataclass
class KoopmanConfig:
    LATENT_DIM: int
    HIDDEN_DIM: int
    LR: float
    BATCH_SIZE: int
    EPOCHS: int
    KOOPMAN_LAMBDA: float
    K_MAX: int          # multi-step consistency horizon


@dataclass
class TransformerConfig:
    LATENT_DIM: int     # should match KoopmanConfig.LATENT_DIM
    NHEAD: int
    NUM_LAYERS: int
    DIM_FEEDFORWARD: int
    DROPOUT: float
    LR: float
    BATCH_SIZE: int
    EPOCHS: int
    ROLLOUT_STEPS: int  # long rollout horizon used in eval
    MAX_LEN_EXTRA: int  # extra margin for positional encoding length
    LOSS_X_WEIGHT: float
    TEACHER_FORCING_INIT: float
    TEACHER_FORCING_FINAL: float
    LATENT_NOISE_STD: float
    FINE_TUNE_ENCODER: bool
    GRAD_CLIP: float


@dataclass
class EvalConfig:
    # here you can put additional eval-specific params if needed
    # e.g. OOD rollout horizon, etc.
    OOD_ROLLOUT_STEPS: int


@dataclass
class ExperimentConfig:
    name: str
    DATA_ROOT: str
    simulation: SimulationConfig
    dataset: DatasetConfig
    koopman: KoopmanConfig
    transformer: TransformerConfig
    eval: EvalConfig


# ----------------------------------------------------------------
# Define experiments here
# ----------------------------------------------------------------

EXPERIMENTS: Dict[str, ExperimentConfig] = {}

EXPERIMENTS["experiment1_2025-11-28"] = ExperimentConfig(
    name="experiment1_2025-11-28",
    DATA_ROOT="data",
    simulation=SimulationConfig(
        G=1.0,
        T_SPAN=(0.0, 10.0),
        NUM_STEPS=2000,
        NUM_TRAJECTORIES=600,
        NUM_TRAJ_OOD=3,
        PERTURBATION=0.05,
    ),
    dataset=DatasetConfig(
        SEQ_LEN=500,
        HORIZON=8,
        TRAIN_FRAC=0.7,
        VAL_FRAC=0.15,
    ),
    koopman=KoopmanConfig(
        LATENT_DIM=8,
        HIDDEN_DIM=64,
        LR=1e-3,
        BATCH_SIZE=32,
        EPOCHS=20,
        KOOPMAN_LAMBDA=1.0,
        K_MAX=8,
    ),
    transformer=TransformerConfig(
        LATENT_DIM=8,       # must match koopman.LATENT_DIM
        NHEAD=4,
        NUM_LAYERS=4,
        DIM_FEEDFORWARD=256,
        DROPOUT=0.1,
        LR=1e-3,
        BATCH_SIZE=32,
        EPOCHS=40,
        ROLLOUT_STEPS=20,
        MAX_LEN_EXTRA=200,   # PE length >= T_total + margin for offset-aware PE
        LOSS_X_WEIGHT=1.5,
        TEACHER_FORCING_INIT=1.0,
        TEACHER_FORCING_FINAL=0.2,
        LATENT_NOISE_STD=0.01,
        FINE_TUNE_ENCODER=True,
        GRAD_CLIP=1.0,
    ),
    eval=EvalConfig(
        OOD_ROLLOUT_STEPS=400,
    ),
)

EXPERIMENTS["experiment2_2025-11-28_high-variance"] = ExperimentConfig(
    name="experiment2_2025-11-28_high-variance",
    DATA_ROOT="data",
    simulation=SimulationConfig(
        G=1.0,
        T_SPAN=(0.0, 5.0),
        NUM_STEPS=2000,
        NUM_TRAJECTORIES=3000,
        NUM_TRAJ_OOD=3,
        PERTURBATION=0.4,
    ),
    dataset=DatasetConfig(
        SEQ_LEN=400,
        HORIZON=8,
        TRAIN_FRAC=0.7,
        VAL_FRAC=0.15,
    ),
    koopman=KoopmanConfig(
        LATENT_DIM=8,
        HIDDEN_DIM=64,
        LR=1e-3,
        BATCH_SIZE=64,
        EPOCHS=4,
        KOOPMAN_LAMBDA=5.0,
        K_MAX=8,
    ),
    transformer=TransformerConfig(
        LATENT_DIM=8,       # must match koopman.LATENT_DIM
        NHEAD=4,
        NUM_LAYERS=4,
        DIM_FEEDFORWARD=256,
        DROPOUT=0.1,
        LR=1e-3,
        BATCH_SIZE=64,
        EPOCHS=12,
        ROLLOUT_STEPS=100,
        MAX_LEN_EXTRA=200,   # PE length >= T_total + margin for offset-aware PE
        LOSS_X_WEIGHT=1.5,
        TEACHER_FORCING_INIT=1.0,
        TEACHER_FORCING_FINAL=0.3,
        LATENT_NOISE_STD=0.02,
        FINE_TUNE_ENCODER=True,
        GRAD_CLIP=1.0,
    ),
    eval=EvalConfig(
        OOD_ROLLOUT_STEPS=400,
    ),
)

EXPERIMENTS["test_experiment"] = ExperimentConfig(
    name="test_experiment",
    DATA_ROOT="data",
    simulation=SimulationConfig(
        G=1.0,
        T_SPAN=(0.0, 6.0),
        NUM_STEPS=200,
        NUM_TRAJECTORIES=30,
        NUM_TRAJ_OOD=3,
        PERTURBATION=0.0,
    ),
    dataset=DatasetConfig(
        SEQ_LEN=50,
        HORIZON=4,
        TRAIN_FRAC=0.7,
        VAL_FRAC=0.15,
    ),
    koopman=KoopmanConfig(
        LATENT_DIM=8,
        HIDDEN_DIM=64,
        LR=1e-3,
        BATCH_SIZE=64,
        EPOCHS=30,
        KOOPMAN_LAMBDA=5.0,
        K_MAX=8,
    ),
    transformer=TransformerConfig(
        LATENT_DIM=8,       # must match koopman.LATENT_DIM
        NHEAD=4,
        NUM_LAYERS=4,
        DIM_FEEDFORWARD=128,
        DROPOUT=0.05,
        LR=1e-3,
        BATCH_SIZE=64,
        EPOCHS=30,
        ROLLOUT_STEPS=100,
        MAX_LEN_EXTRA=50,   # PE length = SEQ_LEN + ROLLOUT_STEPS + MAX_LEN_EXTRA
        LOSS_X_WEIGHT=1.5,
        TEACHER_FORCING_INIT=1.0,
        TEACHER_FORCING_FINAL=0.4,
        LATENT_NOISE_STD=0.0,
        FINE_TUNE_ENCODER=True,
        GRAD_CLIP=1.0,
    ),
    eval=EvalConfig(
        OOD_ROLLOUT_STEPS=400,
    ),
)

# ----------------------------------------------------------------
# Helper API
# ----------------------------------------------------------------

DEFAULT_EXPERIMENT = "experiment1_2025-11-28"


def get_experiment_config(name: str = None) -> ExperimentConfig:
    if name is None:
        name = DEFAULT_EXPERIMENT
    if name not in EXPERIMENTS:
        raise KeyError(
            f"Unknown experiment '{name}'. Available: {list(EXPERIMENTS.keys())}"
        )
    return EXPERIMENTS[name]
