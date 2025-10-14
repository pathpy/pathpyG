"""Helper functions for plotting."""

# =============================================================================
# File      : utils.py -- Helpers for the plotting functions
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Tue 2023-10-24 18:18 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================

import base64
import os
import tempfile
from pathlib import Path
from typing import Callable


def prepare_tempfile() -> tuple[str, str]:
    """Prepare temporary directory and filename for compilation."""
    # get current directory
    current_dir = os.getcwd()

    # get temporal directory
    temp_dir = tempfile.mkdtemp()

    # change to output dir
    os.chdir(temp_dir)

    return temp_dir, current_dir


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


def inch_to_px(value: float, dpi: int = 96) -> float:
    """Convert inch to px."""
    return value * dpi


def px_to_inch(value: float, dpi: int = 96) -> float:
    """Convert px to inch."""
    return value / dpi


def unit_str_to_float(value: str, unit: str) -> float:
    """Convert string with unit to float in `unit`."""
    conversion_functions: dict[str, Callable[[float], float]] = {
        "cm_to_in": cm_to_inch,
        "in_to_cm": inch_to_cm,
        "in_to_px": inch_to_px,
        "px_to_in": px_to_inch,
        "cm_to_px": lambda x: inch_to_px(cm_to_inch(x)),
        "px_to_cm": lambda x: cm_to_inch(px_to_inch(x)),
    }
    conversion_key = f"{value[-2:]}_to_{unit}"
    if conversion_key in conversion_functions:
        return conversion_functions[conversion_key](float(value[:-2]))
    elif value[-2:] == unit:
        return float(value[:-2])
    else:
        raise ValueError(f"The provided conversion '{conversion_key}' is not supported.")


def image_to_base64(image_path):
    """Convert local image to base64 data URL."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Detect image type
    suffix = path.suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg', 
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml'
    }
    mime_type = mime_types.get(suffix, 'image/png')
    
    # Read and encode
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    
    return f"data:{mime_type};base64,{encoded}"

# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
