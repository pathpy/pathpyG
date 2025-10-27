"""Visualization Utilities for PathpyG.

Essential helper functions for network visualization backends. This module provides
utilities for file management, color conversion, unit conversion, and image processing
to support the various visualization backends in PathpyG.

!!! abstract "Key Utilities"
    - :material-folder-cog: **File Management** - Temporary directory handling for compilation
    - :material-palette: **Color Conversion** - RGB/Hex color format transformations
    - :material-ruler: **Unit Conversion** - Between cm, inches, and pixels
    - :material-image: **Image Processing** - Base64 encoding for web compatibility

These utilities are primarily used internally by visualization backends but can also
be useful for custom visualization development and data preprocessing.

## Usage Examples

!!! example "Color Format Conversion"
    ```python
    from pathpyG.visualisations.utils import rgb_to_hex, hex_to_rgb

    # Convert RGB to hex
    hex_color = rgb_to_hex((255, 0, 0))  # "#ff0000"
    hex_color = rgb_to_hex((1.0, 0.0, 0.0))  # Also "#ff0000"

    # Convert hex to RGB
    rgb_color = hex_to_rgb("#ff0000")  # (255, 0, 0)
    ```

!!! example "Unit Conversions for Layout"
    ```python
    from pathpyG.visualisations.utils import unit_str_to_float

    # Convert between different units
    width_px = unit_str_to_float("12cm", "px")  # Converts 12cm to pixels
    height_in = unit_str_to_float("800px", "in")  # Converts 800px to inches
    ```
"""

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

from IPython.core.getipython import get_ipython


def in_jupyter_notebook() -> bool:
    """Detects whether the current Python session is running inside a Jupyter Notebook.

    Returns:
        bool: True if running inside a Jupyter notebook, False otherwise
    """
    try:
        return "IPKernelApp" in get_ipython().config
    except NameError:
        return False
    except AttributeError:
        return False


def prepare_tempfile() -> tuple[str, str]:
    """Prepare temporary directory for backend compilation processes.

    Creates a secure temporary directory and changes the working directory
    to it. This is essential for LaTeX compilation and other backends that
    generate intermediate files during the rendering process.

    Returns:
        tuple[str, str]: (temp_directory_path, original_directory_path)

    !!! warning "Directory Management"
        The caller is responsible for:

        - Restoring the original working directory
        - Cleaning up the temporary directory when done
    """
    # get current directory
    current_dir = os.getcwd()

    # get temporal directory
    temp_dir = tempfile.mkdtemp()

    # change to output dir
    os.chdir(temp_dir)

    return temp_dir, current_dir


def rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB color tuple to hexadecimal color string.

    Accepts RGB values in either 0-1 float range (matplotlib style) or
    0-255 integer range (web/PIL style) and converts to standard hex format.

    Args:
        rgb: RGB color tuple - either (r, g, b) with values 0-1 or 0-255

    Returns:
        str: Hexadecimal color string (e.g., "#ff0000" for red)

    Raises:
        ValueError: If RGB values are outside valid ranges

    Examples:
        ```python
        # Float values (matplotlib/numpy style)
        hex_color = rgb_to_hex((1.0, 0.0, 0.0))  # "#ff0000" (red)

        # Integer values (web/PIL style)
        hex_color = rgb_to_hex((255, 128, 0))  # "#ff8000" (orange)
        ```

    !!! tip "Format Detection"
        The function automatically detects whether input values are in 0-1
        or 0-255 range and converts appropriately.
    """
    if all(0.0 <= val <= 1.0 for val in rgb):
        rgb = tuple(int(val * 255) for val in rgb)
    elif not all(0 <= val <= 255 for val in rgb) or any(not isinstance(val, int) for val in rgb):
        raise ValueError("RGB values must be in range 0-1 or 0-255.")
    return "#%02x%02x%02x" % rgb


def hex_to_rgb(value: str) -> tuple:
    """Convert hexadecimal color string to RGB color tuple.

    Parses standard hex color strings (with or without '#' prefix) and
    returns RGB values in 0-255 integer range suitable for most graphics libraries.

    Args:
        value: Hexadecimal color string (e.g., "#ff0000" or "ff0000")

    Returns:
        tuple: RGB color tuple with values in range 0-255

    Examples:
        ```python
        # Standard hex with hash
        rgb = hex_to_rgb("#ff0000")  # (255, 0, 0) - red

        # Hex without hash
        rgb = hex_to_rgb("00ff00")  # (0, 255, 0) - green

        # Short hex notation
        rgb = hex_to_rgb("#f0f")  # (255, 0, 255) - magenta
        ```
    """
    value = value.lstrip("#")
    _l = len(value)
    return tuple((int(value[i : i + _l // 3], 16) + 1)**(6 // _l) - 1 for i in range(0, _l, _l // 3))


def cm_to_inch(value: float) -> float:
    """Convert centimeters to inches.

    Converts metric length measurements to imperial inches for compatibility
    with systems that use imperial units.

    Args:
        value: Length in centimeters

    Returns:
        float: Equivalent length in inches (1 cm = 0.393701 in)

    Examples:
        ```python
        # Convert A4 width to inches
        width_in = cm_to_inch(21.0)  # 8.268 inches

        # Convert small measurement
        thickness_in = cm_to_inch(0.1)  # 0.039 inches
        ```
    """
    return value / 2.54


def inch_to_cm(value: float) -> float:
    """Convert inches to centimeters.

    Converts imperial length measurements to metric centimeters for
    standardization and international compatibility.

    Args:
        value: Length in inches

    Returns:
        float: Equivalent length in centimeters (1 in = 2.54 cm)

    Examples:
        ```python
        # Convert US letter width to cm
        width_cm = inch_to_cm(8.5)  # 21.59 cm

        # Convert screen size
        screen_cm = inch_to_cm(15.6)  # 39.624 cm
        ```
    """
    return value * 2.54


def inch_to_px(value: float, dpi: int = 96) -> float:
    """Convert inches to pixels based on DPI resolution.

    Converts physical measurements to screen pixels using dots-per-inch
    resolution for accurate display sizing across different screens.

    Args:
        value: Length in inches
        dpi: Resolution in dots per inch (default: 96 - standard web DPI)

    Returns:
        float: Equivalent length in pixels

    Examples:
        ```python
        # Standard web resolution
        width_px = inch_to_px(8.5)  # 816.0 pixels (96 DPI)

        # High-resolution display
        width_px = inch_to_px(8.5, 300)  # 2550.0 pixels (300 DPI)
        ```
    """
    return value * dpi


def px_to_inch(value: float, dpi: int = 96) -> float:
    """Convert pixels to inches based on DPI resolution.

    Converts screen pixels to physical measurements using dots-per-inch
    resolution for print layout and physical sizing calculations.

    Args:
        value: Length in pixels
        dpi: Resolution in dots per inch (default: 96 - standard web DPI)

    Returns:
        float: Equivalent length in inches

    Examples:
        ```python
        # Standard web resolution
        width_in = px_to_inch(800)  # 8.333 inches (96 DPI)

        # Print resolution conversion
        width_in = px_to_inch(2400, 300)  # 8.0 inches (300 DPI)
        ```
    """
    return value / dpi


def unit_str_to_float(value: str, unit: str) -> float:
    """Convert string with unit suffix to float in target unit.

    Parses strings containing numeric values with unit suffixes (e.g., "10px", "5cm")
    and converts to the specified target unit using appropriate conversion functions.

    Args:
        value: String with numeric value and 2-character unit suffix
        unit: Target unit for conversion ("px", "cm", "in")

    Returns:
        float: Converted numeric value in target unit

    Raises:
        ValueError: If conversion between units is not supported

    Examples:
        ```python
        # Convert pixel string to centimeters
        cm_value = unit_str_to_float("800px", "cm")  # 21.17 cm (96 DPI)

        # Convert cm string to inches
        in_value = unit_str_to_float("21cm", "in")  # 8.268 inches

        # Same unit (no conversion needed)
        px_value = unit_str_to_float("100px", "px")  # 100.0
        ```

    !!! warning "Supported Conversions"
        Only supports conversions between "px", "cm", and "in" units.
        Pixel conversions assume 96 DPI by default.

    Supported conversion patterns:

    | From | To | Function |
    |------|----| ---------|
    | cm   | in | `cm_to_inch()` |
    | in   | cm | `inch_to_cm()` |
    | in   | px | `inch_to_px()` |
    | px   | in | `px_to_inch()` |
    | cm   | px | `cm_to_inch() + inch_to_px()` |
    | px   | cm | `px_to_inch() + inch_to_cm()` |
    """
    conversion_functions: dict[str, Callable[[float], float]] = {
        "cm_to_in": cm_to_inch,
        "in_to_cm": inch_to_cm,
        "in_to_px": inch_to_px,
        "px_to_in": px_to_inch,
        "cm_to_px": lambda x: inch_to_px(cm_to_inch(x)),
        "px_to_cm": lambda x: inch_to_cm(px_to_inch(x)),
    }
    conversion_key = f"{value[-2:]}_to_{unit}"
    if conversion_key in conversion_functions:
        return conversion_functions[conversion_key](float(value[:-2]))
    elif value[-2:] == unit:
        return float(value[:-2])
    else:
        raise ValueError(f"The provided conversion '{conversion_key}' is not supported.")


def image_to_base64(image_path):
    """Convert local image file to base64 data URL for embedding.

    Reads an image file from disk and converts it to a base64-encoded data URL
    that can be embedded directly in HTML, SVG, or other formats without
    requiring external file references.

    Args:
        image_path: Path to the image file (str or Path object)

    Returns:
        str: Base64 data URL (e.g., "data:image/png;base64,iVBORw0KGgoAAAA...")

    Raises:
        FileNotFoundError: If the specified image file does not exist

    Examples:
        ```python
        # Convert PNG logo to data URL
        logo_data = image_to_base64("logo.png")
        # Returns: "data:image/png;base64,iVBORw0KGgoAAAA..."

        # Use in HTML template
        html = f'<img src="{logo_data}" alt="Logo">'

        # Use in SVG embedding
        svg_image = f'<image href="{logo_data}" x="10" y="10"/>'
        ```

    !!! info "Supported Formats"
        Automatically detects MIME types for PNG, JPEG, GIF, and SVG files
        based on file extension. Defaults to PNG for unknown extensions.

    !!! tip "Use Cases"
        - Embedding images in standalone HTML/SVG files
        - Creating self-contained visualizations
        - Avoiding external file dependencies in templates
        - Allows visualizations in VSCode Jupyter notebook- and browser-environments where local file access is restricted
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Detect image type
    suffix = path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
    }
    mime_type = mime_types.get(suffix, "image/png")

    # Read and encode
    with open(image_path, "rb") as f:
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
