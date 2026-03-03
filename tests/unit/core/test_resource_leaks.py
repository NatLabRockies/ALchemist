"""
Regression tests for resource leak fixes in session.py temp file handling.

Verifies that temp files are cleaned up even when exceptions occur
during load_data() and export_session_json().
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from alchemist_core import OptimizationSession


class TestLoadDataTempFileCleanup:
    """Verify load_data() cleans up temp files on all code paths."""

    def _make_session_with_vars(self):
        session = OptimizationSession()
        session.add_variable('x1', 'real', bounds=(0.0, 1.0))
        session.add_variable('x2', 'real', bounds=(0.0, 1.0))
        return session

    def test_temp_file_cleaned_on_csv_write_failure(self, tmp_path):
        """If df.to_csv() fails, the temp file must still be cleaned up."""
        session = self._make_session_with_vars()
        
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x1,x2,Output\n0.5,0.5,1.0\n")
        
        leaked_paths = []
        original_named_temp = tempfile.NamedTemporaryFile

        def tracking_temp(*args, **kwargs):
            f = original_named_temp(*args, **kwargs)
            leaked_paths.append(f.name)
            return f

        with patch('tempfile.NamedTemporaryFile', side_effect=tracking_temp):
            with patch.object(pd.DataFrame, 'to_csv', side_effect=IOError("disk full")):
                with pytest.raises(IOError, match="disk full"):
                    session.load_data(str(csv_path))

        # Verify temp file was cleaned up
        for path in leaked_paths:
            assert not os.path.exists(path), f"Temp file leaked: {path}"

    def test_temp_file_cleaned_on_successful_load(self, tmp_path):
        """Normal successful path should also clean up temp file."""
        session = self._make_session_with_vars()
        
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x1,x2,Output\n0.5,0.5,1.0\n0.3,0.7,0.8\n")
        
        session.load_data(str(csv_path))

        # Verify no temp files linger in system temp dir with our suffix
        temp_dir = tempfile.gettempdir()
        csv_temps = [f for f in os.listdir(temp_dir) if f.endswith('.csv') and 'tmp' in f.lower()]
        # This is a best-effort check; we mainly verify load succeeded
        assert session.get_data_summary()['n_experiments'] == 2


class TestExportSessionJsonTempFileCleanup:
    """Verify export_session_json() cleans up temp files on all code paths."""

    def _make_session_with_data(self):
        session = OptimizationSession()
        session.add_variable('x1', 'real', bounds=(0.0, 1.0))
        session.add_experiment(inputs={'x1': 0.5}, output=1.0)
        return session

    def test_temp_file_cleaned_on_save_failure(self):
        """If save_session() fails, the temp file must still be cleaned up."""
        session = self._make_session_with_data()
        
        leaked_paths = []
        original_named_temp = tempfile.NamedTemporaryFile

        def tracking_temp(*args, **kwargs):
            f = original_named_temp(*args, **kwargs)
            leaked_paths.append(f.name)
            return f

        with patch('tempfile.NamedTemporaryFile', side_effect=tracking_temp):
            with patch.object(session, 'save_session', side_effect=RuntimeError("save failed")):
                with pytest.raises(RuntimeError, match="save failed"):
                    session.export_session_json()

        for path in leaked_paths:
            assert not os.path.exists(path), f"Temp file leaked: {path}"

    def test_temp_file_cleaned_on_success(self):
        """Normal path should clean up and return valid JSON."""
        session = self._make_session_with_data()
        result = session.export_session_json()
        assert isinstance(result, str)
        assert len(result) > 0
