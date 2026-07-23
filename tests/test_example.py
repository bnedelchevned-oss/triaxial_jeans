"""Example tests for the triaxial_jeans package."""

import pytest


def test_import():
    """Test that the package can be imported."""
    import triaxial_jeans
    assert triaxial_jeans.__version__ is not None


def test_version():
    """Test the package version."""
    import triaxial_jeans
    assert triaxial_jeans.__version__ == "0.1.0"
