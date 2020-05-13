"""Independent checks for a Forcing Atlas evidence file.

This package does not import the training loop or the energy-balance solver.
It recomputes a few decisive relationships from the stored numbers and rejects
the file when those relationships fail.
"""

__all__ = ["verify_path"]
