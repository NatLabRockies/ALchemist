from doc_blocks import extract_blocks

PAGE = '''# Title

```python
a = 1
```

=== "Python"

    ```python
    b = a + 1
    ```

=== "REST"

    ```http
    POST /api/v1/sessions
    ```

```{ .python .no-run }
raise SystemExit
```
'''


def test_extracts_top_level_python_block():
    blocks = extract_blocks(PAGE)
    assert blocks[0].lang == "python" and blocks[0].code == "a = 1\n" and blocks[0].line == 3


def test_extracts_indented_blocks_inside_tabs():
    blocks = extract_blocks(PAGE)
    assert blocks[1].lang == "python" and blocks[1].code == "b = a + 1\n"
    assert blocks[2].lang == "http" and blocks[2].code.startswith("POST /api/v1/sessions")


def test_attribute_fence_sets_lang_and_classes():
    block = extract_blocks(PAGE)[3]
    assert block.lang == "python" and "no-run" in block.classes


def test_unclosed_fence_raises():
    import pytest
    with pytest.raises(ValueError, match="unclosed"):
        extract_blocks("```python\nx = 1\n")
