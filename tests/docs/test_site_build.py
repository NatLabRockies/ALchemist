"""The zensical-dependent docs tests skip without zensical, and fail instead when CI requires it."""
from pathlib import Path

import pytest

from site_build import REQUIRE_ENV, require_zensical


def test_missing_zensical_skips_by_default():
    with pytest.raises(pytest.skip.Exception):
        require_zensical(finder=lambda: None, environ={})


def test_missing_zensical_fails_when_required():
    with pytest.raises(pytest.fail.Exception):
        require_zensical(finder=lambda: None, environ={REQUIRE_ENV: "1"})


def test_found_zensical_is_returned():
    path = Path("/opt/bin/zensical")
    assert require_zensical(finder=lambda: path, environ={REQUIRE_ENV: "1"}) == path
