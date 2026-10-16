"""
Event system for ALchemist Core.

Provides a simple event emitter pattern for progress reporting and status updates
without requiring UI dependencies.
"""

from typing import Dict, Callable, Any, List
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EventEmitter:
    """
    Simple event emitter for progress and status updates.
    
    Allows components to emit events that can be subscribed to by UI or other systems
    without creating tight coupling.
    
    Example:
        >>> emitter = EventEmitter()
        >>> 
        >>> # Subscribe to an event
        >>> def on_progress(data):
        ...     print(f"Progress: {data['current']}/{data['total']}")
        >>> 
        >>> emitter.on("progress", on_progress)
        >>> 
        >>> # Emit an event
        >>> emitter.emit("progress", {"current": 5, "total": 10})
        Progress: 5/10
        
        >>> # Unsubscribe
        >>> emitter.off("progress", on_progress)
    """
    
    def __init__(self):
        """Initialize the event emitter with no listeners."""
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
    
    def on(self, event_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to an event.
        
        Args:
            event_name: Name of the event to listen for
            callback: Function to call when event is emitted. 
                     Should accept a dict of event data.
        
        Example:
            >>> def my_handler(data):
            ...     print(f"Received: {data}")
            >>> emitter.on("my_event", my_handler)
        """
        if callback not in self._listeners[event_name]:
            self._listeners[event_name].append(callback)
            logger.debug(f"Added listener for event '{event_name}'")
    
    def off(self, event_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unsubscribe from an event.
        
        Args:
            event_name: Name of the event to stop listening for
            callback: The callback function to remove
        
        Example:
            >>> emitter.off("my_event", my_handler)
        """
        # A once() listener is registered through a wrapper; match either
        # the wrapper itself or the callback it carries, so a pending once
        # can be removed before it fires (A32).
        for registered in list(self._listeners[event_name]):
            if registered is callback or getattr(registered, '_once_callback', None) is callback:
                self._listeners[event_name].remove(registered)
                logger.debug(f"Removed listener for event '{event_name}'")
    
    def once(self, event_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to an event but only fire once, then auto-unsubscribe.
        
        Args:
            event_name: Name of the event to listen for
            callback: Function to call when event is emitted (only once)
        
        Example:
            >>> emitter.once("complete", lambda data: print("Done!"))
        """
        def wrapper(data):
            self.off(event_name, wrapper)
            callback(data)

        wrapper._once_callback = callback
        self.on(event_name, wrapper)
    
    def emit(self, event_name: str, data: Dict[str, Any] = None) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            event_name: Name of the event to emit
            data: Optional dictionary of data to pass to listeners
        
        Example:
            >>> emitter.emit("progress", {"current": 1, "total": 10})
            >>> emitter.emit("error", {"message": "Something went wrong"})
        """
        event_data = data or {}
        
        # Call all registered listeners for this event. Iterate over a copy:
        # a listener that unsubscribes itself (once() does) must not shift
        # the next listener out of this emit (A32).
        for callback in list(self._listeners[event_name]):
            try:
                callback(event_data)
            except Exception as e:
                # Don't let listener errors break the emitter
                logger.error(f"Error in event listener for '{event_name}': {e}", exc_info=True)
    
    def remove_all_listeners(self, event_name: str = None) -> None:
        """
        Remove all listeners for a specific event, or all events if no name given.
        
        Args:
            event_name: Optional event name. If None, removes all listeners for all events.
        
        Example:
            >>> emitter.remove_all_listeners("progress")  # Remove all progress listeners
            >>> emitter.remove_all_listeners()  # Remove all listeners
        """
        if event_name is None:
            self._listeners.clear()
            logger.debug("Removed all event listeners")
        else:
            self._listeners[event_name].clear()
            logger.debug(f"Removed all listeners for event '{event_name}'")
    
    def listener_count(self, event_name: str) -> int:
        """
        Get the number of listeners for a specific event.
        
        Args:
            event_name: Name of the event
        
        Returns:
            Number of listeners registered for this event
        
        Example:
            >>> count = emitter.listener_count("progress")
        """
        return len(self._listeners[event_name])
    
    def event_names(self) -> List[str]:
        """
        Get list of all event names that have listeners.
        
        Returns:
            List of event names
        
        Example:
            >>> events = emitter.event_names()
            >>> print(events)  # ['progress', 'complete', 'error']
        """
        return [name for name, listeners in self._listeners.items() if listeners]


# Common event data schemas (for documentation/type hints)

class ProgressEventData:
    """
    Standard schema for progress events.
    
    Fields:
        current (int): Current step/iteration
        total (int): Total steps/iterations
        message (str): Optional progress message
        percentage (float): Optional percentage complete (0-100)
    """
    pass


class ErrorEventData:
    """
    Standard schema for error events.
    
    Fields:
        message (str): Error message
        exception (Exception): Optional exception object
        recoverable (bool): Whether the error is recoverable
    """
    pass


class ModelTrainedEventData:
    """
    Standard schema for model trained events.
    
    Fields:
        backend (str): Backend used (sklearn, botorch, etc.)
        hyperparameters (dict): Model hyperparameters
        training_time (float): Training time in seconds
    """
    pass


class NextPointSuggestedEventData:
    """
    Standard schema for next point suggested events.
    
    Fields:
        point (dict or DataFrame): Suggested experiment point(s)
        strategy (str): Strategy used (acquisition, EMOC, etc.)
        acq_func (str): Acquisition function used
        acq_value (float): Acquisition function value at suggested point
    """
    pass
