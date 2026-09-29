"""RareCal: audit and repair how often a generative model produces rare events."""
from .severity import WhitenedRadius, MaxPatchRadius
from .calibration import CalibrationLaw
from .audit import audit, verdict, AuditReport, TailRatio
from .wrap import Wrapper, rank_targets
from .transport import gradient_flow, minimal_displacement
from .intervals import wald, wilson, clopper_pearson, quadrature
from .theory import (frechet_distance, frechet_distance_features,
                     sample_size_lower_bound, frechet_cost_bound)

__version__ = "0.1.0"
__all__ = ["WhitenedRadius", "MaxPatchRadius", "CalibrationLaw", "audit", "verdict",
           "AuditReport", "TailRatio", "Wrapper", "rank_targets", "gradient_flow",
           "minimal_displacement", "wald", "wilson", "clopper_pearson", "quadrature",
           "frechet_distance", "frechet_distance_features", "sample_size_lower_bound",
           "frechet_cost_bound"]
