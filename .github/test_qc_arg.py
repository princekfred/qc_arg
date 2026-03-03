"""
Unit and regression test for the qc_arg package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import qc_arg


def test_qc_arg_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "qc_arg" in sys.modules
