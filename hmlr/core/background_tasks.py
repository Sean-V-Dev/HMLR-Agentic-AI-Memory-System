
"""
Background Task Manager for HMLR.

This module provides a safe way to run background tasks (like Scribe)
without silent failures. It holds strong references to tasks to prevent
garbage collection and logs exceptions upon completion.
"""

import asyncio
import logging
import traceback
from typing import Set, Callable, Optional, Any

logger = logging.getLogger(__name__)

class BackgroundTaskManager:
    """
    Manages background asyncio tasks to prevent silent failures and GC issues.
    """
    
    def __init__(self):
        # Hold strong references to prevent garbage collection of running tasks
        self._active_tasks: Set[asyncio.Task] = set()
        
    def add_task(self, coro, name: str = "background_task") -> asyncio.Task:
        """
        Schedule a coroutine as a background task.
        
        Args:
            coro: The coroutine to run (e.g. self.scribe.run_agent(...))
            name: Human-readable name for logging
            
        Returns:
            The created asyncio.Task
        """
        task = asyncio.create_task(coro, name=name)
        
        # Add to set to maintain strong reference
        self._active_tasks.add(task)
        
        # Add callback to handle completion (success or failure)
        task.add_done_callback(self._create_done_callback(name))
        
        return task
        
    def _create_done_callback(self, name: str) -> Callable[[asyncio.Task], None]:
        """Create a closure for the done callback."""
        def done_callback(t: asyncio.Task):
            try:
                # Remove from active set
                self._active_tasks.discard(t)
                
                # Check for exceptions
                exc = t.exception()
                if exc:
                    logger.error(
                        f"Background task '{name}' failed with error: {exc}",
                        exc_info=exc
                    )
                elif t.cancelled():
                    logger.warning(f"Background task '{name}' was cancelled")
                else:
                    # Success case - debug log only
                    logger.debug(f"Background task '{name}' completed successfully")
                    
            except asyncio.CancelledError:
                pass
            except Exception as e:
                # Failing in the callback is bad - log definitively
                logger.critical(f"Critical error in task callback for '{name}': {e}", exc_info=True)
                
        return done_callback
    
    async def shutdown(self, timeout: float = 5.0):
        """
        Wait for all background tasks to complete.
        
        Args:
            timeout: Max seconds to wait
        """
        if not self._active_tasks:
            return
            
        logger.info(f"Waiting for {len(self._active_tasks)} background tasks to complete...")
        
        # Wait for all tasks
        # We wrap in wait_for to enforce timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._active_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timed out waiting for background tasks ({len(self._active_tasks)} remaining)")
            # Optional: cancel remaining?
            # for t in self._active_tasks: t.cancel()
