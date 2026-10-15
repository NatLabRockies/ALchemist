"""
Session Store - Session management with disk persistence.

Stores OptimizationSession instances with recovery backup system.

A session lives in memory and in ``<persist_dir>/<session_id>.json``. The
store starts empty and reads a session's file the first time its ID is used,
so a known session survives a restart without the server loading every file
it has ever written. The API saves a session after every request that changes
it (``api/main.py``, ``autosave_sessions``); writes are atomic.
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
import uuid
from alchemist_core.session import OptimizationSession
from ..middleware.error_handlers import SessionFileUnreadableError
import contextlib
import logging
import json
import os
import re
import tempfile
import time
from pathlib import Path
import threading

# TODO: Consider migrating per-session `threading.Lock()` to an async-compatible
# `anyio.Lock()` (or `asyncio.Lock`) for cleaner async endpoint integration.
#
# Rationale / next steps:
# - Many API endpoints are `async def` and blocking the event loop with
#   `threading.Lock().acquire()` is undesirable.
# - A migration plan is in `memory/SESSION_LOCKING_ASYNC_PLAN.md` describing
#   how to transition to `anyio.Lock()` and update handlers to use `async with`.

logger = logging.getLogger(__name__)

# Session IDs the store will look up on disk: what create() mints (UUIDs) and
# the IDs callers pass to import_session. Nothing that can name a path. Used
# with ``fullmatch``: ``$`` would also accept a trailing newline.
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]+")


class SessionStore:
    """Session store with disk persistence."""
    
    def __init__(self, default_ttl_hours: int = 24, persist_dir: Optional[str] = None):
        """
        Initialize session store.
        
        Args:
            default_ttl_hours: Legacy parameter (kept for compatibility, not used for TTL)
            persist_dir: Directory to persist sessions (None = memory only)
        """
        self._sessions: Dict[str, Dict] = {}
        self.persist_dir = Path(persist_dir) if persist_dir else Path("cache/sessions")
        self.recovery_dir = Path("cache/recovery")
        
        # Create directories
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        
        # One lock per session ID serialises that session's first load against
        # a second request for it and against its deletion, so two requests
        # never read it into separate objects and a delete cannot be undone by
        # a load in progress. A load can retrain a model, so these must stay
        # per ID: a store-wide lock would stall every other session behind it.
        # ``_id_locks_guard`` only protects the dict and is never held across
        # a load.
        self._id_locks: Dict[str, threading.Lock] = {}
        self._id_locks_guard = threading.Lock()

        logger.info(f"SessionStore initialized with persist_dir={self.persist_dir}, recovery_dir={self.recovery_dir}")
    
    def _id_lock(self, session_id: str) -> threading.Lock:
        """The lock for one session ID (callers pass an ID that passed validation)."""
        with self._id_locks_guard:
            return self._id_locks.setdefault(session_id, threading.Lock())

    def _is_valid_id(self, session_id) -> bool:
        return bool(self.persist_dir) and isinstance(session_id, str) \
            and bool(_SESSION_ID_RE.fullmatch(session_id))

    def _get_session_file(self, session_id: str) -> Path:
        """Get path to session file."""
        return self.persist_dir / f"{session_id}.json"
    
    def _save_to_disk(self, session_id: str):
        """Write the session to its file, atomically. Raises on failure.

        The JSON is written to a hidden file beside the target and renamed over
        it, so a crash or a failed write leaves the previous file whole.
        """
        if not self.persist_dir:
            return

        session_file = self._get_session_file(session_id)
        session_data = self._sessions[session_id]
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{session_id}.", suffix=".tmp", dir=self.persist_dir
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            session_data["session"].save_session(str(tmp_path))
            with open(tmp_path, 'r') as f:
                session_json = json.load(f)
            session_json["_session_store_metadata"] = {
                "created_at": session_data["created_at"].isoformat(),
                "last_accessed": session_data["last_accessed"].isoformat()
            }
            with open(tmp_path, 'w') as f:
                json.dump(session_json, f, indent=2)
                # On disk before the rename makes it the session file.
                f.flush()
                os.fsync(f.fileno())
            # mkstemp makes 0600; the file was 0644 when written in place.
            os.chmod(tmp_path, 0o644)
            self._replace_with_retry(tmp_path, session_file)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _retry_on_permission_error(operation, attempts: int = 5, delay: float = 0.1):
        """Call ``operation()``, retrying on PermissionError.

        On Windows a rename or unlink fails while another process (an AV
        scanner, a copy in progress) holds the file open without share-delete;
        the hold is usually brief. The last PermissionError is raised.
        """
        for attempt in range(1, attempts + 1):
            try:
                return operation()
            except PermissionError:
                if attempt == attempts:
                    raise
                time.sleep(delay)

    @classmethod
    def _replace_with_retry(cls, src: Path, dst: Path):
        cls._retry_on_permission_error(lambda: os.replace(src, dst))

    def _try_save(self, session_id: str) -> bool:
        """Save, logging rather than raising; for callers whose change stands
        whether or not it reached disk (the next save retries it)."""
        try:
            self._save_to_disk(session_id)
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False

    def _file_exists_exactly(self, session_id: str) -> bool:
        """True if ``<session_id>.json`` is in the directory under exactly that name.

        macOS and Windows file systems ignore case, so opening ``<ID>.json``
        would read the file of ``<id>`` and load a second copy of the session
        under another key.
        """
        try:
            return f"{session_id}.json" in os.listdir(self.persist_dir) \
                and self._get_session_file(session_id).is_file()
        except OSError:
            return False

    def _ensure_loaded(self, session_id: str) -> bool:
        """True if the session is in memory, reading its file first if needed.

        False when there is no such session. A file that exists but cannot be
        read raises ``SessionFileUnreadableError`` (the API answers 503), so a
        caller never mistakes a session it cannot serve for one that is gone.

        The model is rebuilt from the session's last training call (the file's
        ``model_config.training_config``), which can take seconds; the lock
        held meanwhile is this session ID's own.
        """
        if session_id in self._sessions:
            return True
        if not self._is_valid_id(session_id):
            return False
        with self._id_lock(session_id):
            if session_id in self._sessions:
                return True
            session_file = self._get_session_file(session_id)
            if not self._file_exists_exactly(session_id):
                return False
            try:
                with open(session_file, 'r') as f:
                    session_json = json.load(f)
                metadata = session_json.pop("_session_store_metadata", {})
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    json.dump(session_json, tmp)
                    temp_path = tmp.name
                try:
                    session = OptimizationSession.load_session(temp_path, retrain_on_load=True)
                finally:
                    Path(temp_path).unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Failed to load session from {session_file}: {e}")
                raise SessionFileUnreadableError(
                    f"Session {session_id} exists but its file could not be read: {e}"
                ) from e
            now = datetime.now().isoformat()
            self._sessions[session_id] = {
                "session": session,
                "created_at": datetime.fromisoformat(metadata.get("created_at", now)),
                "last_accessed": datetime.fromisoformat(metadata.get("last_accessed", now)),
                "lock": threading.Lock()
            }
            logger.info(f"Loaded session {session_id} from disk")
            return True

    def _delete_from_disk(self, session_id: str):
        """Delete session file from disk."""
        if not self.persist_dir:
            return
        
        session_file = self._get_session_file(session_id)
        try:
            # A file that cannot be removed raises (the API answers 500), so a
            # delete that returned 204 never comes back after a restart; only
            # a missing file counts as removed.
            self._retry_on_permission_error(session_file.unlink)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.error(f"Failed to delete session file {session_id}: {e}")
            raise
    
    def create(self, name: Optional[str] = None, description: Optional[str] = None, tags: Optional[list] = None) -> str:
        """
        Create a new session.
        
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        session = OptimizationSession()
        # Ensure session metadata matches store id
        try:
            session.metadata.session_id = session_id
        except Exception:
            pass
        # Populate optional metadata
        if name:
            session.metadata.name = name
        if description:
            session.metadata.description = description
        if tags:
            try:
                session.metadata.tags = tags
            except Exception:
                pass

        self._sessions[session_id] = {
            "session": session,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "lock": threading.Lock()
        }
        
        self._try_save(session_id)
        logger.info(f"Created session {session_id}")
        return session_id
    
    def get(self, session_id: str) -> Optional[OptimizationSession]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            OptimizationSession or None if not found
        """
        if not self._ensure_loaded(session_id):
            logger.warning(f"Session {session_id} not found")
            return None
        
        # Read once: a delete between the load and here removes the entry.
        session_data = self._sessions.get(session_id)
        if session_data is None:
            return None
        lock = session_data.get("lock")
        if lock:
            with lock:
                # Update last accessed time (no save to disk)
                session_data["last_accessed"] = datetime.now()
                return session_data["session"]
        else:
            # Fallback (no lock present)
            session_data["last_accessed"] = datetime.now()
            return session_data["session"]
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        A session that is not in memory is deleted by removing its file; it is
        not loaded first. Deletion takes the session ID's load lock, so a
        first load in progress finishes before the delete and cannot bring the
        session back afterwards.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        valid = self._is_valid_id(session_id)
        if not valid and session_id not in self._sessions:
            return False
        with self._id_lock(session_id) if valid else contextlib.nullcontext():
            if session_id in self._sessions:
                lock = self._sessions[session_id].get("lock")
                # The file goes first: if it cannot be removed the session
                # stays in memory too, matching the 500 the caller gets.
                if lock:
                    with lock:
                        self._delete_from_disk(session_id)
                        del self._sessions[session_id]
                else:
                    self._delete_from_disk(session_id)
                    del self._sessions[session_id]
            else:
                if not self._file_exists_exactly(session_id):
                    return False
                session_file = self._get_session_file(session_id)
                try:
                    # A file that cannot be removed raises (the API answers
                    # 500); only a missing file is "not found".
                    self._retry_on_permission_error(session_file.unlink)
                except FileNotFoundError:
                    return False
                except OSError as e:
                    logger.error(f"Failed to delete session file {session_id}: {e}")
                    raise
        logger.info(f"Deleted session {session_id}")
        return True

    def get_info(self, session_id: str) -> Optional[Dict]:
        """
        Get session metadata.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session info or None
        """
        if not self._ensure_loaded(session_id):
            return None
        
        session_data = self._sessions.get(session_id)
        if session_data is None:
            return None
        session = session_data["session"]
        
        return {
            "session_id": session_id,
            "created_at": session_data["created_at"].isoformat(),
            "last_accessed": session_data["last_accessed"].isoformat(),
            "search_space": session.get_search_space_summary(),
            "data": session.get_data_summary(),
            "model": session.get_model_summary()
        }
    
    def extend_ttl(self, session_id: str, hours: int = None) -> bool:
        """
        Legacy method - no longer used (sessions don't expire).
        Kept for API compatibility.
        
        Args:
            session_id: Session identifier
            hours: Ignored
            
        Returns:
            True if session exists, False otherwise
        """
        if not self._ensure_loaded(session_id):
            return False
        logger.info(f"extend_ttl called for session {session_id} (no-op - TTL removed)")
        return True
    
    def _cleanup_expired(self):
        """Legacy method - no longer used (sessions don't expire)."""
        # No-op: sessions no longer have TTL expiration
        pass
    
    def count(self) -> int:
        """Get count of active sessions."""
        return len(self._sessions)
    
    def list_all(self) -> list:
        """Get list of all active session IDs."""
        return list(self._sessions.keys())
    
    def export_session(self, session_id: str) -> Optional[str]:
        """
        Export a session as JSON string for download.
        
        Args:
            session_id: Session identifier
            
        Returns:
            JSON string of session data or None if not found
        """
        if not self._ensure_loaded(session_id):
            return None
        entry = self._sessions.get(session_id)
        if entry is None:
            return None

        def export() -> str:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    temp_path = tmp.name
                entry["session"].save_session(temp_path)
                with open(temp_path, 'r') as f:
                    return f.read()
            finally:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)

        try:
            lock = entry.get("lock")
            if lock:
                with lock:
                    return export()
            return export()
        except Exception as e:
            logger.error(f"Failed to export session {session_id}: {e}")
            return None

    def persist_session_to_disk(self, session_id: str) -> bool:
        """
        Persist the in-memory session to disk (overwrite existing persisted file).

        Returns True on success, False otherwise.
        """
        if not self._ensure_loaded(session_id):
            return False
        entry = self._sessions.get(session_id)
        if entry is None:
            return False
        try:
            lock = entry.get('lock')
            if lock:
                with lock:
                    self._save_to_disk(session_id)
            else:
                self._save_to_disk(session_id)
            return True
        except Exception as e:
            logger.error(f"Failed to persist session {session_id}: {e}")
            return False
    
    def import_session(self, session_data: str, session_id: Optional[str] = None) -> Optional[str]:
        """
        Import a session from JSON string.
        
        Args:
            session_data: JSON string of session data
            session_id: Optional custom session ID (generates new one if None)
            
        Returns:
            Session ID or None if import failed
        """
        temp_path = None
        try:
            # Write JSON to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                temp_path = tmp.name
                tmp.write(session_data)

            # Rebuild the model from the file's last training call; a failed
            # retrain is logged and the session loads without a model.
            session = OptimizationSession.load_session(temp_path, retrain_on_load=True)

            # Generate new session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())

            # Ensure session metadata session_id matches store id
            try:
                session.metadata.session_id = session_id
            except Exception:
                pass

            # Store session with metadata and lock; under the ID's load lock
            # so it cannot interleave with a first load or a delete of it.
            with self._id_lock(session_id) if self._is_valid_id(session_id) \
                    else contextlib.nullcontext():
                self._sessions[session_id] = {
                    "session": session,
                    "created_at": datetime.now(),
                    "last_accessed": datetime.now(),
                    "lock": threading.Lock()
                }

            self._try_save(session_id)

            logger.info(f"Imported session {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Failed to import session: {e}")
            return None
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    # ============================================================
    # Session Locking for Programmatic Control
    # ============================================================
    
    def lock_session(self, session_id: str, locked_by: str, client_id: Optional[str] = None) -> Dict:
        """Lock a session for external programmatic control."""
        if not self._ensure_loaded(session_id):
            raise KeyError(f"Session {session_id} not found")
        
        lock_token = str(uuid.uuid4())
        lock_time = datetime.now()
        
        self._sessions[session_id]["lock_info"] = {
            "locked": True,
            "locked_by": locked_by,
            "client_id": client_id,
            "locked_at": lock_time.isoformat(),
            "lock_token": lock_token
        }
        
        self._try_save(session_id)
        logger.info(f"Session {session_id} locked by {locked_by}")
        
        return {
            "locked": True,
            "locked_by": locked_by,
            "locked_at": lock_time.isoformat(),
            "lock_token": lock_token
        }
    
    def unlock_session(self, session_id: str, lock_token: Optional[str] = None) -> Dict:
        """Unlock a session."""
        if not self._ensure_loaded(session_id):
            raise KeyError(f"Session {session_id} not found")
        
        lock_info = self._sessions[session_id].get("lock_info", {})
        
        # If token provided, verify it
        if lock_token and lock_info.get("lock_token") != lock_token:
            raise ValueError("Invalid lock token")
        
        # Clear lock info
        self._sessions[session_id]["lock_info"] = {
            "locked": False,
            "locked_by": None,
            "client_id": None,
            "locked_at": None,
            "lock_token": None
        }
        
        self._try_save(session_id)
        logger.info(f"Session {session_id} unlocked")
        
        return {
            "locked": False,
            "locked_by": None,
            "locked_at": None,
            "lock_token": None
        }
    
    def get_lock_status(self, session_id: str) -> Dict:
        """Get current lock status without exposing the token."""
        if not self._ensure_loaded(session_id):
            raise KeyError(f"Session {session_id} not found")
        
        lock_info = self._sessions[session_id].get("lock_info", {})
        
        # Don't log status checks - they happen frequently via polling
        return {
            "locked": lock_info.get("locked", False),
            "locked_by": lock_info.get("locked_by"),
            "locked_at": lock_info.get("locked_at"),
            "lock_token": None  # Never expose token in status check
        }

    # ============================================================
    # Recovery / Backup System
    # ============================================================
    
    def _get_recovery_file(self, session_id: str) -> Path:
        """Get path to recovery backup file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.recovery_dir / f"{session_id}_recovery_{timestamp}.json"
    
    def save_recovery_backup(self, session_id: str) -> bool:
        """
        Save a recovery backup for crash protection.
        
        This is called periodically (every 30s) from frontend to create
        silent backups. User never sees these unless needed for recovery.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self._ensure_loaded(session_id):
            return False
        
        try:
            lock = self._sessions[session_id].get("lock")
            
            # Clean up old recovery files for this session first
            self._cleanup_old_recovery_files(session_id)
            
            recovery_file = self._get_recovery_file(session_id)
            
            if lock:
                with lock:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                        self._sessions[session_id]["session"].save_session(tmp.name)
                        temp_path = tmp.name
                    
                    # Read and add recovery metadata
                    with open(temp_path, 'r') as f:
                        session_json = json.load(f)
                    
                    session_json["_recovery_metadata"] = {
                        "session_id": session_id,
                        "backup_time": datetime.now().isoformat(),
                        "session_name": self._sessions[session_id]["session"].metadata.name
                    }
                    
                    # Write to recovery file
                    with open(recovery_file, 'w') as f:
                        json.dump(session_json, f, indent=2)
                    
                    # Clean up temp file
                    Path(temp_path).unlink()
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    self._sessions[session_id]["session"].save_session(tmp.name)
                    temp_path = tmp.name
                
                with open(temp_path, 'r') as f:
                    session_json = json.load(f)
                
                session_json["_recovery_metadata"] = {
                    "session_id": session_id,
                    "backup_time": datetime.now().isoformat(),
                    "session_name": self._sessions[session_id]["session"].metadata.name
                }
                
                with open(recovery_file, 'w') as f:
                    json.dump(session_json, f, indent=2)
                
                Path(temp_path).unlink()
            
            logger.debug(f"Recovery backup saved for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save recovery backup for {session_id}: {e}")
            return False
    
    def _cleanup_old_recovery_files(self, session_id: str, keep_newest: int = 1):
        """
        Clean up old recovery files for a session, keeping only the newest.
        
        Args:
            session_id: Session identifier
            keep_newest: Number of newest files to keep (default 1)
        """
        if not self.recovery_dir.exists():
            return
        
        # Find all recovery files for this session
        pattern = f"{session_id}_recovery_*.json"
        recovery_files = sorted(self.recovery_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        
        # Delete all but the newest
        for old_file in recovery_files[:-keep_newest]:
            try:
                old_file.unlink()
                logger.debug(f"Deleted old recovery file: {old_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete old recovery file {old_file}: {e}")
    
    def clear_recovery_backup(self, session_id: str) -> bool:
        """
        Delete all recovery backups for a session.
        
        Called after user successfully saves their session to their computer.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if any files were deleted
        """
        if not self.recovery_dir.exists():
            return False
        
        deleted = False
        pattern = f"{session_id}_recovery_*.json"
        
        for recovery_file in self.recovery_dir.glob(pattern):
            try:
                recovery_file.unlink()
                logger.info(f"Deleted recovery backup: {recovery_file.name}")
                deleted = True
            except Exception as e:
                logger.error(f"Failed to delete recovery file {recovery_file}: {e}")
        
        return deleted
    
    def list_recovery_sessions(self) -> List[Dict]:
        """
        List all available recovery sessions.
        
        Returns list of recovery metadata for frontend to display.
        """
        if not self.recovery_dir.exists():
            return []
        
        recoveries = []
        
        # Group by session_id (only show newest for each)
        session_files = {}
        for recovery_file in self.recovery_dir.glob("*_recovery_*.json"):
            try:
                # Extract session_id from filename
                parts = recovery_file.stem.split("_recovery_")
                if len(parts) != 2:
                    continue
                
                session_id = parts[0]
                
                # Keep only newest file per session
                if session_id not in session_files:
                    session_files[session_id] = recovery_file
                else:
                    # Compare modification times
                    if recovery_file.stat().st_mtime > session_files[session_id].stat().st_mtime:
                        session_files[session_id] = recovery_file
            except Exception as e:
                logger.warning(f"Error processing recovery file {recovery_file}: {e}")
        
        # Load metadata from newest files
        for session_id, recovery_file in session_files.items():
            try:
                with open(recovery_file, 'r') as f:
                    session_json = json.load(f)
                
                metadata = session_json.get("_recovery_metadata", {})
                session_meta = session_json.get("metadata", {})
                
                # Get session statistics
                experiments = session_json.get("experiments", {})
                n_experiments = experiments.get("n_total", 0)
                
                search_space = session_json.get("search_space", {})
                n_variables = len(search_space.get("variables", []))
                
                model_config = session_json.get("model_config", {})
                # ``fitted: false`` marks a block that only keeps the training
                # call for a model that is not currently fitted.
                model_trained = bool(model_config) and model_config.get("fitted", True) is not False
                
                recoveries.append({
                    "session_id": session_id,
                    "session_name": metadata.get("session_name", session_meta.get("name", "Untitled Session")),
                    "backup_time": metadata.get("backup_time", datetime.fromtimestamp(recovery_file.stat().st_mtime).isoformat()),
                    "n_variables": n_variables,
                    "n_experiments": n_experiments,
                    "model_trained": model_trained,
                    "file_path": str(recovery_file)
                })
            except Exception as e:
                logger.error(f"Error reading recovery file {recovery_file}: {e}")
        
        # Sort by backup time (newest first)
        recoveries.sort(key=lambda x: x["backup_time"], reverse=True)
        
        return recoveries
    
    def restore_from_recovery(self, session_id: str) -> Optional[str]:
        """
        Restore a session from recovery backup.
        
        Creates a new active session from the recovery file.
        
        Args:
            session_id: Original session ID to restore
            
        Returns:
            New session ID if successful, None otherwise
        """
        if not self.recovery_dir.exists():
            return None
        
        # Find newest recovery file for this session
        pattern = f"{session_id}_recovery_*.json"
        recovery_files = sorted(self.recovery_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not recovery_files:
            logger.warning(f"No recovery files found for session {session_id}")
            return None
        
        recovery_file = recovery_files[0]
        
        try:
            # Read recovery file
            with open(recovery_file, 'r') as f:
                session_data = f.read()
            
            # Import as new session (generates new ID)
            new_session_id = self.import_session(session_data)
            
            if new_session_id:
                logger.info(f"Restored session {session_id} as new session {new_session_id}")
                # Keep recovery file until user explicitly saves
                return new_session_id
            
        except Exception as e:
            logger.error(f"Failed to restore from recovery {recovery_file}: {e}")
        
        return None
    
    def cleanup_old_recoveries(self, max_age_hours: int = 24):
        """
        Clean up recovery files older than specified hours.
        
        Called periodically to prevent accumulation.
        
        Args:
            max_age_hours: Maximum age in hours (default 24)
        """
        if not self.recovery_dir.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        deleted_count = 0
        
        for recovery_file in self.recovery_dir.glob("*_recovery_*.json"):
            try:
                if recovery_file.stat().st_mtime < cutoff_time:
                    recovery_file.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old recovery file: {recovery_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete recovery file {recovery_file}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old recovery files")


    # ============================================================
    # Edison Result Cache
    # ============================================================

    def get_edison_cache(self, session_id: str) -> dict:
        """Return (and lazily create) the Edison result cache for this session.

        The cache is a plain dict stored directly on the session store entry,
        keyed by a short hash of (job_type, query).  It is intentionally kept
        separate from the OptimizationSession object so it never enters the
        JSON serialisation / disk-persistence path.

        Stores two kinds of entries:
          - Pending:  {"status": "running",   "task_id": str, "trajectory_url": str}
          - Complete: {"status": "complete",  "task_id": str, "trajectory_url": str,
                       "answer": str, "formatted_answer": str,
                       "has_successful_answer": bool}

        Returns an empty (unattached) dict if the session is not found.
        """
        if not self._ensure_loaded(session_id):
            return {}
        entry = self._sessions[session_id]
        return entry.setdefault("edison_cache", {})


# Global session store instance
session_store = SessionStore(default_ttl_hours=24)
