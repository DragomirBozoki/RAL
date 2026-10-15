"""RAL: Robot Audio Layer.

A vendor-neutral audio layer for robots. See ``docs/DESIGN.md`` for the architecture.
Importing this package must only need the core dependencies; optional integrations
are imported lazily inside the modules that use them.
"""

__version__ = "0.0.1.dev0"

__all__ = ["__version__"]
