"""
Unit tests for the desktop UI's variable-row formatter.

These cover the regression where loading a session that contained a
`discrete` or `context` variable would crash `_sync_session_to_ui`
with `KeyError: 'values'` because the helper assumed every non-real/
non-integer variable was categorical.

The formatter is module-level (not method-level) so it can be tested
without instantiating CustomTkinter.
"""

import pytest

# CustomTkinter must initialise a Tk root on import, which fails in headless
# CI. Skip the whole module if it isn't importable in this environment.
pytest.importorskip("customtkinter")

from ui.ui import _variable_to_sheet_row


class TestVariableToSheetRow:

    def test_real_variable(self):
        row = _variable_to_sheet_row({"name": "temp", "type": "real", "min": 100, "max": 200})
        assert row == ["temp", "real", 100, 200, ""]

    def test_integer_variable(self):
        row = _variable_to_sheet_row({"name": "n", "type": "integer", "min": 1, "max": 5})
        assert row == ["n", "integer", 1, 5, ""]

    def test_categorical_variable_with_values_key(self):
        """Canonical SearchSpace schema: categorical uses 'values'."""
        row = _variable_to_sheet_row(
            {"name": "cat", "type": "categorical", "values": ["A", "B", "C"]}
        )
        assert row == ["cat", "categorical", "", "", "A, B, C"]

    def test_categorical_variable_with_categories_key(self):
        """Web export schema: categorical uses 'categories'. Must not raise."""
        row = _variable_to_sheet_row(
            {"name": "cat", "type": "categorical", "categories": ["A", "B"]}
        )
        assert row == ["cat", "categorical", "", "", "A, B"]

    def test_discrete_variable_does_not_raise(self):
        """Regression: previously crashed with KeyError('values')."""
        row = _variable_to_sheet_row(
            {"name": "SAR", "type": "discrete", "allowed_values": [80, 280]}
        )
        assert row == ["SAR", "discrete", "", "", "80, 280"]

    def test_context_variable_does_not_raise(self):
        """Regression: previously crashed with KeyError('values')."""
        row = _variable_to_sheet_row({"name": "batch_quality", "type": "context"})
        assert row == ["batch_quality", "context", "", "", ""]

    def test_unknown_type_falls_back_without_raising(self):
        """Defensive: an unknown variable type must not crash the UI sync."""
        row = _variable_to_sheet_row({"name": "weird", "type": "something_new"})
        # Just verify we didn't crash and produced a sane row.
        assert row[0] == "weird"
        assert row[1] == "something_new"

    def test_missing_optional_fields_renders_blanks(self):
        """Defensive: missing min/max/values render as blanks, not crashes."""
        row = _variable_to_sheet_row({"name": "x", "type": "real"})
        assert row == ["x", "real", "", "", ""]
