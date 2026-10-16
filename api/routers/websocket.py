"""
WebSocket router for real-time session updates.

Provides real-time push notifications for session events like lock status changes,
eliminating the need for client-side polling.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Track active WebSocket connections per session
# Structure: {session_id: {websocket1, websocket2, ...}}
active_connections: Dict[str, Set[WebSocket]] = {}


@router.websocket("/ws/sessions/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time session updates.
    
    Clients connect to this endpoint to receive push notifications about
    session events (lock status changes, experiment additions, etc.).
    
    Args:
        websocket: WebSocket connection
        session_id: Session ID to subscribe to
    """
    await websocket.accept()
    logger.debug(f"WebSocket connected: session_id={session_id}")
    
    # Register this connection for this session
    if session_id not in active_connections:
        active_connections[session_id] = set()
    active_connections[session_id].add(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "event": "connected",
            "session_id": session_id,
            "message": "WebSocket connection established"
        })
        
        # Keep connection alive and listen for client messages
        while True:
            # Receive messages from client (for future bi-directional features)
            data = await websocket.receive_text()
            
            # Echo back for debugging (can be removed in production)
            try:
                message = json.loads(data)
                logger.debug(f"Received from client: {message}")
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client: {data}")
            
    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected: session_id={session_id}")
    finally:
        # Clean up on disconnect
        if session_id in active_connections:
            active_connections[session_id].discard(websocket)
            
            # Remove session entry if no more connections
            if not active_connections[session_id]:
                del active_connections[session_id]
                logger.debug(f"No more connections for session {session_id}")


async def broadcast_to_session(session_id: str, event: dict):
    """
    Broadcast an event to all WebSocket clients connected to a session.
    
    Args:
        session_id: Session ID to broadcast to
        event: Event data to send (will be JSON serialized)
    """
    if session_id not in active_connections:
        logger.debug(f"No active connections for session {session_id}")
        return
    
    # Track dead connections
    dead_connections = set()
    
    # Send to all connected clients. Over a copy: a connect or disconnect
    # during an awaited send_json changes the set, and iterating the live
    # set would raise inside the HTTP handler that broadcast.
    for connection in list(active_connections.get(session_id, ())):
        try:
            await connection.send_json(event)
            logger.debug(f"Broadcast to session {session_id}: {event.get('event')}")
        except Exception as e:
            logger.warning(f"Failed to send to connection: {e}")
            dead_connections.add(connection)
    
    # Clean up dead connections
    if dead_connections:
        active_connections[session_id] -= dead_connections
        logger.info(f"Cleaned up {len(dead_connections)} dead connections")
        
        # Remove session if no connections left
        if not active_connections[session_id]:
            del active_connections[session_id]


def get_connection_count(session_id: str) -> int:
    """
    Get the number of active WebSocket connections for a session.
    
    Args:
        session_id: Session ID to check
        
    Returns:
        Number of active connections
    """
    return len(active_connections.get(session_id, set()))


def get_all_connection_counts() -> Dict[str, int]:
    """
    Get connection counts for all sessions.
    
    Returns:
        Dictionary mapping session_id to connection count
    """
    return {
        session_id: len(connections)
        for session_id, connections in active_connections.items()
    }
