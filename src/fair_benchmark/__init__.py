"""Independent fair-v3 protocol and training controls; no import-time GPU access."""

from .spec import load_protocol, make_plan

__all__ = ["load_protocol", "make_plan"]
