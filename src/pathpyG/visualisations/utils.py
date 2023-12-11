"""Helper functions for plotting."""
# =============================================================================
# File      : utils.py -- Helpers for the plotting functions
# Author    : Jürgen Hackl <hackl@princeton.edu>
# Time-stamp: <Tue 2023-10-24 18:18 juergen>
#
# Copyright (c) 2016-2023 Pathpy Developers
# =============================================================================
from typing import Optional


def rgb_to_hex(rgb: tuple) -> str:
    """Convert rgb color tuple to hex string."""
    return "#%02x%02x%02x" % rgb


def hex_to_rgb(value: str) -> tuple:
    """Convert hex string to rgb color tuple."""
    value = value.lstrip("#")
    _l = len(value)
    return tuple(int(value[i : i + _l // 3], 16) for i in range(0, _l, _l // 3))


class Colormap:
    """Very simple colormap class."""

    def __call__(
        self,
        values: list,
        alpha: Optional[float] = None,
        bytes: bool = False,
    ) -> list:
        """Return color value."""
        vmin, vmax = min(values), max(values)
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        return [
            self.color_tuple(v)
            for v in ((x - vmin) / (vmax - vmin) * 100 for x in values)
        ]

    @staticmethod
    def color_tuple(n: float) -> tuple:
        """Return color ramp from green to red."""
        return (int((255 * n) * 0.01), int((255 * (100 - n)) * 0.01), 0, 255)


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 79
# End:
