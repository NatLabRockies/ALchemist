"""Every ```http block names a real method and path, and only known body keys."""
import json
import re

import pytest

from doc_blocks import DOCS, doc_pages, extract_blocks

REQUEST_LINE = re.compile(r"^(GET|POST|PUT|PATCH|DELETE)\s+(\S+)")


@pytest.fixture(scope="module")
def schema():
    from api.main import app
    return app.openapi()


def _template_regex(path: str) -> re.Pattern:
    return re.compile("^" + re.sub(r"\{[^}]+\}", r"[^/]+", path) + "$")


def _match(schema, method, path):
    path = path.split("?", 1)[0]
    for template, ops in schema["paths"].items():
        if _template_regex(template).match(path) and method.lower() in ops:
            return ops[method.lower()]
    return None


def _body_keys(schema, op):
    ref = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {}).get("$ref")
    if not ref:
        return None
    props = schema["components"]["schemas"][ref.rsplit("/", 1)[1]].get("properties", {})
    return set(props)


CASES = [(p, b) for p in doc_pages() for b in extract_blocks(p.read_text()) if b.lang == "http"]


@pytest.mark.parametrize("page,block", CASES, ids=[f"{p.relative_to(DOCS)}:{b.line}" for p, b in CASES])
def test_http_example_matches_schema(schema, page, block):
    first, _, rest = block.code.partition("\n")
    m = REQUEST_LINE.match(first)
    assert m, f"first line must be 'METHOD /path': {first!r}"
    method, path = m.groups()
    op = _match(schema, method, path)
    assert op is not None, f"{method} {path} is not in the OpenAPI schema"
    body = rest.split("\n\n", 1)[1].strip() if "\n\n" in rest else ""
    if body.startswith("{"):
        keys = _body_keys(schema, op)
        assert keys is not None, f"{method} {path} takes no JSON body"
        unknown = set(json.loads(body)) - keys
        assert not unknown, f"unknown body keys {sorted(unknown)}"


def test_detects_bad_path(schema):
    assert _match(schema, "POST", "/api/v1/sessions/abc/models/train") is None
    assert _match(schema, "POST", "/api/v1/sessions/abc/model/train") is not None
