# Safe import for Colab compatibility
try:
    from .ez_tree import *
except ImportError:
    # C++ extension not available, will use Python fallback
    pass
