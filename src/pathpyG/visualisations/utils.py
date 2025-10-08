"""Helper functions for plotting."""

# =============================================================================
# File      : utils.py -- Helpers for the plotting functions
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Tue 2023-10-24 18:18 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================


def rgb_to_hex(rgb: tuple) -> str:
    """Convert rgb color tuple to hex string.
    
    Args:
        rgb (tuple): RGB color tuple either in range 0-1 or 0-255.
    """
    if all(0.0 <= val <= 1.0 for val in rgb):
        rgb = tuple(int(val * 255) for val in rgb)
    elif not all(0 <= val <= 255 for val in rgb):
        raise ValueError("RGB values must be in range 0-1 or 0-255.")
    return "#%02x%02x%02x" % rgb


def hex_to_rgb(value: str) -> tuple:
    """Convert hex string to rgb color tuple."""
    value = value.lstrip("#")
    _l = len(value)
    return tuple(int(value[i : i + _l // 3], 16) for i in range(0, _l, _l // 3))


def cm_to_inch(value: float) -> float:
    """Convert cm to inch."""
    return value / 2.54

def inch_to_cm(value: float) -> float:
    """Convert inch to cm."""
    return value * 2.54

def unit_str_to_float(value: str, unit: str) -> float:
    """Convert string with unit to float in `unit`."""
    if value.endswith("cm"):
        return float(value[:-2]) if unit == "cm" else cm_to_inch(float(value[:-2]))
    elif value.endswith("in"):
        return inch_to_cm(float(value[:-2])) if unit == "cm" else float(value[:-2])
    else:
        raise ValueError("Value must end with 'cm' or 'in'.")

# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
