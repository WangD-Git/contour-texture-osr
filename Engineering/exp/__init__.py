"""Paper experiments: baseline / msp / ablation / cd."""

from .baseline import main as run_baseline
from .msp import main as run_msp
from .ablation import main as run_ablation
from .cd import main as run_cd

__all__ = ["run_baseline", "run_msp", "run_ablation", "run_cd"]
