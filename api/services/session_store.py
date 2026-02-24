"""
Session Store - Session management with disk persistence.

Stores OptimizationSession instances with recovery backup system.
Sessions persist in RAM until explicitly saved by user.
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
import uuid
from alchemist_core.session import OptimizationSession
import logging
import json
import tempfile
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
        
        # Note: No longer auto-loading sessions on startup
        # Sessions are created on-demand or loaded explicitly by user
        logger.info(f"SessionStore initialized with persist_dir={self.persist_dir}, recovery_dir={self.recovery_dir}")
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get path to session file."""
        return self.persist_dir / f"{session_id}.json"
    
    def _save_to_disk(self, session_id: str):
        """Save session to disk as JSON."""
        if not self.persist_dir:
            return
        
        try:
            session_file = self._get_session_file(session_id)
            session_data = self._sessions[session_id]
            
            # Create a temporary file for the session
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                session_data["session"].save_session(tmp.name)
                temp_path = tmp.name
            
            # Store metadata alongside session
            metadata = {
                "created_at": session_data["created_at"].isoformat(),
                "last_accessed": session_data["last_accessed"].isoformat()
            }
            
            # Load session JSON and add metadata
            with open(temp_path, 'r') as f:
                session_json = json.load(f)
            
            session_json["_session_store_metadata"] = metadata
            
            # Write combined data
            with open(session_file, 'w') as f:
                json.dump(session_json, f, indent=2)
            
            # Clean up temp file
            Path(temp_path).unlink()
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
    
    def _load_from_disk(self):
        """Load all sessions from disk."""
        if not self.persist_dir or not self.persist_dir.exists():
            return
        
        loaded_count = 0
        for session_file in self.persist_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_json = json.load(f)
                
                # Extract metadata
                metadata = session_json.pop("_session_store_metadata", {})
                
                # No longer check for expiration - TTL system removed
                
                # Write session data to temp file and load
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    json.dump(session_json, tmp, indent=2)
                    temp_path = tmp.name
                
                # Load without retraining by default during startup
                session = OptimizationSession.load_session(temp_path, retrain_on_load=False)
                Path(temp_path).unlink()
                session_id = session_file.stem
                self._sessions[session_id] = {
                    "session": session,
                    "created_at": datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                    "last_accessed": datetime.fromisoformat(metadata.get("last_accessed", datetime.now().isoformat())),
                    "lock": threading.Lock()
                }
                loaded_count += 1
                
            except Exception as e:
                logger.error(f"Failed to load session from {session_file}: {e}")
        
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} sessions from disk")
    
    def _delete_from_disk(self, session_id: str):
        """Delete session file from disk."""
        if not self.persist_dir:
            return
        
        try:
            session_file = self._get_session_file(session_id)
            if session_file.exists():
                session_file.unlink()
        except Exception as e:
            logger.error(f"Failed to delete session file {session_id}: {e}")
    
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
        
        # Note: No automatic disk save on creation
        # User will explicitly save when ready
        
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
        if session_id not in self._sessions:
            logger.warning(f"Session {session_id} not found")
            return None
        
        session_data = self._sessions[session_id]
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
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            lock = self._sessions[session_id].get("lock")
            if lock:
                with lock:
                    del self._sessions[session_id]
                    self._delete_from_disk(session_id)
            else:
                del self._sessions[session_id]
                self._delete_from_disk(session_id)
            logger.info(f"Deleted session {session_id}")
            return True
        return False
    
    def get_info(self, session_id: str) -> Optional[Dict]:
        """
        Get session metadata.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session info or None
        """
        if session_id not in self._sessions:
            return None
        
        session_data = self._sessions[session_id]
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
        if session_id not in self._sessions:
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
        if session_id not in self._sessions:
            return None

        try:
            lock = self._sessions[session_id].get("lock")
            if lock:
                with lock:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                        self._sessions[session_id]["session"].save_session(tmp.name)
                        temp_path = tmp.name

                    # Read the JSON content
                    with open(temp_path, 'r') as f:
                        json_content = f.read()

                    # Clean up temp file
                    Path(temp_path).unlink()
                    return json_content
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    self._sessions[session_id]["session"].save_session(tmp.name)
                    temp_path = tmp.name

                with open(temp_path, 'r') as f:
                    json_content = f.read()

                Path(temp_path).unlink()
                return json_content
        except Exception as e:
            logger.error(f"Failed to export session {session_id}: {e}")
            return None

    def persist_session_to_disk(self, session_id: str) -> bool:
        """
        Persist the in-memory session to disk (overwrite existing persisted file).

        Returns True on success, False otherwise.
        """
        if session_id not in self._sessions:
            return False
        try:
            lock = self._sessions[session_id].get('lock')
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
        try:
            # Write JSON to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp.write(session_data)
                temp_path = tmp.name
            
            # Load session without automatic retrain
            session = OptimizationSession.load_session(temp_path, retrain_on_load=False)
            Path(temp_path).unlink()
            
            # Generate new session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())

            # Ensure session metadata session_id matches store id
            try:
                session.metadata.session_id = session_id
            except Exception:
                pass

            # Store session with metadata and lock
            self._sessions[session_id] = {
                "session": session,
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "lock": threading.Lock()
            }
            
            self._save_to_disk(session_id)
            
            logger.info(f"Imported session {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to import session: {e}")
            return None

    # ============================================================
    # Session Locking for Programmatic Control
    # ============================================================
    
    def lock_session(self, session_id: str, locked_by: str, client_id: Optional[str] = None) -> Dict:
        """Lock a session for external programmatic control."""
        if session_id not in self._sessions:
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
        
        self._save_to_disk(session_id)
        logger.info(f"Session {session_id} locked by {locked_by}")
        
        return {
            "locked": True,
            "locked_by": locked_by,
            "locked_at": lock_time.isoformat(),
            "lock_token": lock_token
        }
    
    def unlock_session(self, session_id: str, lock_token: Optional[str] = None) -> Dict:
        """Unlock a session."""
        if session_id not in self._sessions:
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
        
        self._save_to_disk(session_id)
        logger.info(f"Session {session_id} unlocked")
        
        return {
            "locked": False,
            "locked_by": None,
            "locked_at": None,
            "lock_token": None
        }
    
    def get_lock_status(self, session_id: str) -> Dict:
        """Get current lock status without exposing the token."""
        if session_id not in self._sessions:
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
        if session_id not in self._sessions:
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
                model_trained = model_config is not None and len(model_config) > 0
                
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
        entry = self._sessions.get(session_id)
        if entry is None:
            return {}
        return entry.setdefault("edison_cache", {})


# Global session store instance
session_store = SessionStore(default_ttl_hours=24)
