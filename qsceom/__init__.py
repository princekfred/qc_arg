"""QSC-EOM and related quantum-chemistry solvers."""

from .UCCSD import gs_exact
from .adaptvqe import adapt_vqe
from .qc_arg import canvas
from .qsceom import qsceom

try:
    from ._version import __version__
except ImportError:
    __version__ = "0+unknown"

__all__ = ["__version__", "adapt_vqe", "canvas", "gs_exact", "qsceom"]
