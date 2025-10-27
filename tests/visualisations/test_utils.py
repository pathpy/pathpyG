"""Unit tests for visualization utilities."""

import base64
import os
from unittest.mock import MagicMock

import pytest

from pathpyG.visualisations.utils import (
    cm_to_inch,
    hex_to_rgb,
    image_to_base64,
    in_jupyter_notebook,
    inch_to_cm,
    inch_to_px,
    prepare_tempfile,
    px_to_inch,
    rgb_to_hex,
    unit_str_to_float,
)


class TestJupyterDetection:
    """Test Jupyter notebook detection utilities."""

    def test_in_jupyter_notebook_true(self, monkeypatch):
        """Test that in_jupyter_notebook returns True when running in Jupyter."""
        mock_ipython = MagicMock()
        mock_ipython.config = {"IPKernelApp": {}}
        
        def mock_get_ipython():
            return mock_ipython
        
        monkeypatch.setattr("pathpyG.visualisations.utils.get_ipython", mock_get_ipython)
        assert in_jupyter_notebook() is True

    def test_in_jupyter_notebook_false_no_ipkernelapp(self, monkeypatch):
        """Test that in_jupyter_notebook returns False when IPKernelApp not in config."""
        mock_ipython = MagicMock()
        mock_ipython.config = {"SomeOtherApp": {}}
        
        def mock_get_ipython():
            return mock_ipython
        
        monkeypatch.setattr("pathpyG.visualisations.utils.get_ipython", mock_get_ipython)
        assert in_jupyter_notebook() is False

    def test_in_jupyter_notebook_name_error(self, monkeypatch):
        """Test that in_jupyter_notebook returns False when get_ipython raises NameError."""
        def mock_get_ipython_raises_name_error():
            raise NameError("name 'get_ipython' is not defined")
        
        monkeypatch.setattr("pathpyG.visualisations.utils.get_ipython", mock_get_ipython_raises_name_error)
        
        assert in_jupyter_notebook() is False

    def test_in_jupyter_notebook_attribute_error(self, monkeypatch):
        """Test that in_jupyter_notebook returns False when get_ipython().config raises AttributeError."""
        mock_ipython = MagicMock()
        del mock_ipython.config  # Remove config attribute to trigger AttributeError
        
        def mock_get_ipython():
            return mock_ipython
        
        monkeypatch.setattr("pathpyG.visualisations.utils.get_ipython", mock_get_ipython)
        
        assert in_jupyter_notebook() is False


class TestColorConversion:
    """Test RGB/Hex color conversion utilities."""

    def test_rgb_to_hex_float_values(self):
        """Test RGB to hex with float values (0-1 range)."""
        assert rgb_to_hex((1.0, 0.0, 0.0)) == "#ff0000"
        assert rgb_to_hex((0.0, 1.0, 0.0)) == "#00ff00"
        assert rgb_to_hex((0.0, 0.0, 1.0)) == "#0000ff"
        assert rgb_to_hex((1.0, 1.0, 1.0)) == "#ffffff"
        assert rgb_to_hex((0.0, 0.0, 0.0)) == "#000000"

    def test_rgb_to_hex_int_values(self):
        """Test RGB to hex with integer values (0-255 range)."""
        assert rgb_to_hex((255, 0, 0)) == "#ff0000"
        assert rgb_to_hex((0, 255, 0)) == "#00ff00"
        assert rgb_to_hex((0, 0, 255)) == "#0000ff"
        assert rgb_to_hex((255, 128, 0)) == "#ff8000"
        assert rgb_to_hex((128, 128, 128)) == "#808080"

    def test_rgb_to_hex_mixed_precision(self):
        """Test RGB to hex with edge cases and precision."""
        assert rgb_to_hex((0.5, 0.5, 0.5)) == "#7f7f7f"
        assert rgb_to_hex((0.25, 0.75, 0.5)) == "#3fbf7f"

    def test_rgb_to_hex_invalid_values(self):
        """Test RGB to hex with invalid values."""
        with pytest.raises(ValueError, match="RGB values must be in range"):
            rgb_to_hex((256, 0, 0))
        with pytest.raises(ValueError, match="RGB values must be in range"):
            rgb_to_hex((1.5, 0.5, 0.5))
        with pytest.raises(ValueError, match="RGB values must be in range"):
            rgb_to_hex((-1, 0, 0))

    def test_hex_to_rgb_with_hash(self):
        """Test hex to RGB with hash prefix."""
        assert hex_to_rgb("#ff0000") == (255, 0, 0)
        assert hex_to_rgb("#00ff00") == (0, 255, 0)
        assert hex_to_rgb("#0000ff") == (0, 0, 255)
        assert hex_to_rgb("#ffffff") == (255, 255, 255)
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_hex_to_rgb_without_hash(self):
        """Test hex to RGB without hash prefix."""
        assert hex_to_rgb("ff0000") == (255, 0, 0)
        assert hex_to_rgb("00ff00") == (0, 255, 0)
        assert hex_to_rgb("0000ff") == (0, 0, 255)

    def test_hex_to_rgb_short_notation(self):
        """Test hex to RGB with short notation."""
        assert hex_to_rgb("#f0f") == (255, 0, 255)
        assert hex_to_rgb("fff") == (255, 255, 255)
        assert hex_to_rgb("000") == (0, 0, 0)

    def test_rgb_hex_roundtrip(self):
        """Test that RGB->Hex->RGB conversion is stable."""
        original = (200, 100, 50)
        hex_color = rgb_to_hex(original)
        result = hex_to_rgb(hex_color)
        assert result == original


class TestUnitConversion:
    """Test length unit conversion utilities."""

    def test_cm_to_inch(self):
        """Test centimeters to inches conversion."""
        assert cm_to_inch(2.54) == pytest.approx(1.0)
        assert cm_to_inch(10.0) == pytest.approx(3.937, rel=1e-3)
        assert cm_to_inch(0) == 0
        assert cm_to_inch(21.0) == pytest.approx(8.268, rel=1e-3)

    def test_inch_to_cm(self):
        """Test inches to centimeters conversion."""
        assert inch_to_cm(1.0) == pytest.approx(2.54)
        assert inch_to_cm(0) == 0
        assert inch_to_cm(8.5) == pytest.approx(21.59)
        assert inch_to_cm(11.0) == pytest.approx(27.94)

    def test_cm_inch_roundtrip(self):
        """Test that cm->inch->cm conversion is stable."""
        original = 15.5
        result = inch_to_cm(cm_to_inch(original))
        assert result == pytest.approx(original)

    def test_inch_to_px_default_dpi(self):
        """Test inches to pixels with default 96 DPI."""
        assert inch_to_px(1.0) == 96.0
        assert inch_to_px(0) == 0
        assert inch_to_px(8.5) == 816.0
        assert inch_to_px(0.5) == 48.0

    def test_inch_to_px_custom_dpi(self):
        """Test inches to pixels with custom DPI."""
        assert inch_to_px(1.0, dpi=300) == 300.0
        assert inch_to_px(8.5, dpi=300) == 2550.0
        assert inch_to_px(1.0, dpi=72) == 72.0

    def test_px_to_inch_default_dpi(self):
        """Test pixels to inches with default 96 DPI."""
        assert px_to_inch(96) == pytest.approx(1.0)
        assert px_to_inch(0) == 0
        assert px_to_inch(800) == pytest.approx(8.333, rel=1e-3)
        assert px_to_inch(48) == pytest.approx(0.5)

    def test_px_to_inch_custom_dpi(self):
        """Test pixels to inches with custom DPI."""
        assert px_to_inch(300, dpi=300) == pytest.approx(1.0)
        assert px_to_inch(2400, dpi=300) == pytest.approx(8.0)
        assert px_to_inch(72, dpi=72) == pytest.approx(1.0)

    def test_px_inch_roundtrip(self):
        """Test that px->inch->px conversion is stable."""
        original = 1000.0
        result = inch_to_px(px_to_inch(original))
        assert result == pytest.approx(original)

    def test_unit_str_to_float_same_unit(self):
        """Test unit string conversion when units match."""
        assert unit_str_to_float("100px", "px") == 100.0
        assert unit_str_to_float("50cm", "cm") == 50.0
        assert unit_str_to_float("10in", "in") == 10.0

    def test_unit_str_to_float_cm_to_in(self):
        """Test cm to inches conversion."""
        result = unit_str_to_float("2.54cm", "in")
        assert result == pytest.approx(1.0)

    def test_unit_str_to_float_in_to_cm(self):
        """Test inches to cm conversion."""
        result = unit_str_to_float("1in", "cm")
        assert result == pytest.approx(2.54)

    def test_unit_str_to_float_in_to_px(self):
        """Test inches to pixels conversion."""
        assert unit_str_to_float("1in", "px") == 96.0
        assert unit_str_to_float("2in", "px") == 192.0

    def test_unit_str_to_float_px_to_in(self):
        """Test pixels to inches conversion."""
        result = unit_str_to_float("96px", "in")
        assert result == pytest.approx(1.0)

    def test_unit_str_to_float_cm_to_px(self):
        """Test cm to pixels conversion."""
        result = unit_str_to_float("2.54cm", "px")
        assert result == pytest.approx(96.0)

    def test_unit_str_to_float_px_to_cm(self):
        """Test pixels to cm conversion."""
        result = unit_str_to_float("96px", "cm")
        assert result == pytest.approx(2.54)

    def test_unit_str_to_float_unsupported_conversion(self):
        """Test that unsupported conversions raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            unit_str_to_float("100mm", "px")  # mm not supported


class TestImageConversion:
    """Test image to base64 conversion utilities."""

    def test_image_to_base64_png(self, tmp_path):
        """Test PNG image to base64 conversion."""
        # Create a minimal PNG file (1x1 red pixel)
        png_data = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
            0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D,
            0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND chunk
            0x44, 0xAE, 0x42, 0x60, 0x82
        ])
        
        img_path = tmp_path / "test.png"
        img_path.write_bytes(png_data)
        
        result = image_to_base64(str(img_path))
        
        # Check format
        assert result.startswith("data:image/png;base64,")
        
        # Check that it's valid base64
        encoded_part = result.split(",")[1]
        decoded = base64.b64decode(encoded_part)
        assert decoded == png_data

    def test_image_to_base64_jpeg(self, tmp_path):
        """Test JPEG image to base64 conversion."""
        # Create a minimal JPEG file
        jpeg_data = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
            0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9
        ])
        
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(jpeg_data)
        
        result = image_to_base64(str(img_path))
        
        # Check format
        assert result.startswith("data:image/jpeg;base64,")
        
        # Check content
        encoded_part = result.split(",")[1]
        decoded = base64.b64decode(encoded_part)
        assert decoded == jpeg_data

    def test_image_to_base64_jpeg_extensions(self, tmp_path):
        """Test that .jpeg and .jpg both use image/jpeg mime type."""
        jpeg_data = bytes([0xFF, 0xD8, 0xFF, 0xD9])
        
        for ext in [".jpg", ".jpeg"]:
            img_path = tmp_path / f"test{ext}"
            img_path.write_bytes(jpeg_data)
            result = image_to_base64(str(img_path))
            assert result.startswith("data:image/jpeg;base64,")

    def test_image_to_base64_gif(self, tmp_path):
        """Test GIF image to base64 conversion."""
        # Minimal GIF header
        gif_data = bytes([
            0x47, 0x49, 0x46, 0x38, 0x39, 0x61,  # GIF89a
            0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x3B
        ])
        
        img_path = tmp_path / "test.gif"
        img_path.write_bytes(gif_data)
        
        result = image_to_base64(str(img_path))
        assert result.startswith("data:image/gif;base64,")

    def test_image_to_base64_svg(self, tmp_path):
        """Test SVG image to base64 conversion."""
        svg_content = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        
        img_path = tmp_path / "test.svg"
        img_path.write_bytes(svg_content)
        
        result = image_to_base64(str(img_path))
        assert result.startswith("data:image/svg+xml;base64,")

    def test_image_to_base64_unknown_extension(self, tmp_path):
        """Test that unknown extensions default to image/png."""
        data = b"some image data"
        img_path = tmp_path / "test.xyz"
        img_path.write_bytes(data)
        
        result = image_to_base64(str(img_path))
        assert result.startswith("data:image/png;base64,")

    def test_image_to_base64_nonexistent_file(self):
        """Test that nonexistent files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            image_to_base64("/nonexistent/path/to/image.png")

    def test_image_to_base64_with_path_object(self, tmp_path):
        """Test that Path objects work correctly."""
        png_data = b"\x89PNG\x0D\x0A\x1A\x0A"
        img_path = tmp_path / "test.png"
        img_path.write_bytes(png_data)
        
        # Pass as Path object
        result = image_to_base64(img_path)
        assert result.startswith("data:image/png;base64,")


class TestTempfileManagement:
    """Test temporary directory management utilities."""

    def test_prepare_tempfile_creates_temp_dir(self):
        """Test that prepare_tempfile creates a temporary directory."""
        original_cwd = os.getcwd()
        
        try:
            temp_dir, stored_cwd = prepare_tempfile()
            
            # Check that temp_dir exists and is a directory
            assert os.path.isdir(temp_dir)
            
            # Check that stored_cwd is the original directory
            assert stored_cwd == original_cwd
            
            # Check that we're now in the temp directory
            assert os.getcwd() == temp_dir
            
        finally:
            # Clean up: restore original directory
            os.chdir(original_cwd)
            # Try to remove temp dir if it exists
            if 'temp_dir' in locals():
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass  # Directory might not be empty or already removed

    def test_prepare_tempfile_changes_cwd(self):
        """Test that prepare_tempfile changes the current working directory."""
        original_cwd = os.getcwd()
        
        try:
            temp_dir, _ = prepare_tempfile()
            current_cwd = os.getcwd()
            
            # Verify we changed directory
            assert current_cwd != original_cwd
            assert current_cwd == temp_dir
            
        finally:
            os.chdir(original_cwd)
            if 'temp_dir' in locals():
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass

    def test_prepare_tempfile_returns_different_dirs(self):
        """Test that multiple calls create different temp directories."""
        original_cwd = os.getcwd()
        temp_dirs = []
        
        try:
            for _ in range(3):
                temp_dir, _ = prepare_tempfile()
                temp_dirs.append(temp_dir)
                os.chdir(original_cwd)  # Reset for next iteration
            
            # All temp directories should be unique
            assert len(set(temp_dirs)) == 3
            
        finally:
            os.chdir(original_cwd)
            for temp_dir in temp_dirs:
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass
