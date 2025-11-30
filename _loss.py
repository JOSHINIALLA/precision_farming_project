"""Compatibility shim for mis-pickled scikit-learn internal _loss module.
The original models were pickled with a top-level module name `_loss`.
This file re-exports the actual sklearn._loss submodules so that unpickling succeeds.
"""
from importlib import import_module

# Import underlying sklearn internal loss subpackages
_loss_pkg = import_module('sklearn._loss')  # noqa: F401 (ensures base package initialized)
loss = import_module('sklearn._loss.loss')
link = import_module('sklearn._loss.link')

# Optionally expose common symbols at top-level to satisfy attribute lookups
try:
    from sklearn._loss.loss import *  # type: ignore  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass
try:
    from sklearn._loss.link import *  # type: ignore  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass
