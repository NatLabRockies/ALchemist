"""
Pytest configuration for API tests.
"""

import sys

import pytest
from pathlib import Path

# Add project root to path - must be done before any api imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Added to sys.path: {project_root}")


@pytest.fixture(autouse=True, scope="session")
def _isolate_session_store(tmp_path_factory):
    """Point the API's global session store at a temp directory.

    Its default is ``cache/`` under the working directory, so every API test
    used to leave a session file in the checkout -- hundreds of them, which a
    server started there then had to read.
    """
    from api.services.session_store import session_store
    saved = session_store.persist_dir, session_store.recovery_dir
    root = tmp_path_factory.mktemp("session_store")
    session_store.persist_dir = root / "sessions"
    session_store.recovery_dir = root / "recovery"
    session_store.persist_dir.mkdir()
    session_store.recovery_dir.mkdir()
    yield
    session_store.persist_dir, session_store.recovery_dir = saved
