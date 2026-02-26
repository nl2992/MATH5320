"""Public import namespace for the MATH5320 Portfolio Risk System."""

from __future__ import annotations

import sys

import src as _src
from src import config, credit, data, demo_presets, portfolio, pricing, risk, schemas, services, ui

__version__ = _src.__version__

# Re-export the main internal package modules under a cleaner public namespace.
sys.modules[__name__ + ".config"] = config
sys.modules[__name__ + ".credit"] = credit
sys.modules[__name__ + ".data"] = data
sys.modules[__name__ + ".demo_presets"] = demo_presets
sys.modules[__name__ + ".portfolio"] = portfolio
sys.modules[__name__ + ".pricing"] = pricing
sys.modules[__name__ + ".risk"] = risk
sys.modules[__name__ + ".schemas"] = schemas
sys.modules[__name__ + ".services"] = services
sys.modules[__name__ + ".ui"] = ui

__all__ = [
    "__version__",
    "config",
    "credit",
    "data",
    "demo_presets",
    "portfolio",
    "pricing",
    "risk",
    "schemas",
    "services",
    "ui",
]
