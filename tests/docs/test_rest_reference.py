"""The OpenAPI export matches the live app and the page loads it."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_export_matches_app(tmp_path):
    sys.path.insert(0, str(ROOT / "docs" / "scripts"))
    import export_openapi
    from api.main import app
    out = export_openapi.main(tmp_path / "openapi.json")
    assert json.loads(out.read_text()) == app.openapi()


def test_rest_page_built_with_pinned_scalar(built_site):
    html = (built_site / "reference" / "rest-api" / "index.html").read_text()
    assert "@scalar/api-reference@1.74.0" in html
    assert 'integrity="sha384-' in html
    assert 'data-openapi="../openapi.json"' in html
    assert (built_site / "reference" / "openapi.json").exists()
