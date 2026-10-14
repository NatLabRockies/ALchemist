"""Built landing page has the hero, both calls to action and three cards."""
import re

import pytest


@pytest.fixture(scope="module")
def index_html(built_site):
    return (built_site / "index.html").read_text()


def test_landing_hero_and_ctas(index_html):
    assert 'class="alc-hero"' in index_html
    assert 'href="get-started/first-loop/"' in index_html
    assert "pip install alchemist-nrel" in index_html
    assert index_html.count('class="alc-card"') == 3


def test_landing_hides_sidebars(index_html):
    assert 'data-md-type="navigation" hidden' in index_html
    assert 'data-md-type="toc" hidden' in index_html


def test_header_logo_is_the_wordmark_in_both_schemes(index_html):
    header = re.search(r'<header class="md-header[ "].*?</header>', index_html, re.S).group(0)
    assert re.search(r'<img class="only-light" src="(\./)?assets/NEW_LOGO_LIGHT\.png"', header)
    assert re.search(r'<img class="only-dark" src="(\./)?assets/NEW_LOGO_DARK\.png"', header)
    assert "alc-wordmark" not in index_html


def test_landing_title_is_not_doubled(index_html):
    assert "<title>ALchemist</title>" in index_html
