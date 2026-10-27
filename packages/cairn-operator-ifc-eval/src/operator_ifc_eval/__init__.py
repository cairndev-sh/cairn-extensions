"""cairn-operator-ifc-eval — IFC snippet validation (cairn-extensions#12).

Importing this package must NOT import ``ifcopenshell``: it is the optional ``[ifc]`` extra, imported
only by :mod:`operator_ifc_eval.validate` when a check runs.
"""
from __future__ import annotations

__all__ = ["validate"]
