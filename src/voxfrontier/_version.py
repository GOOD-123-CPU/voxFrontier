"""Single source of truth for the VoxFrontier version.

Kept separate from ``__init__.py`` so that ``setuptools`` can read it at
build time without importing the full package (which requires numpy etc.).
"""

__version__ = "1.0.0"
