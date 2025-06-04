from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from .idto import IDTO

class INotificationAttempt(IDTO):
    """Record of a notification attempt."""
    notification: Dict
    timestamp: datetime
    successful: bool
    had_active_connection: bool
    error: Optional[str] = None

    def dict(self, *args, **kwargs) -> dict:
        """Convert to dictionary with serializable datetime."""
        d = super().dict(*args, **kwargs)
        d['timestamp'] = d['timestamp'].isoformat()
        return d

class INotificationState(IDTO):
    """Active notification state."""
    pending_notifications: List[Dict] = Field(default_factory=list)
    notification_history: List[INotificationAttempt] = Field(default_factory=list)
    notifications: List[Dict] = Field(default_factory=list)  # Actual notifications sent
    last_notification_time: Optional[datetime] = None
    notification_count: int = 0
    successful_notifications: int = 0
    failed_notifications: int = 0

    def dict(self, *args, **kwargs) -> dict:
        """Convert to dictionary with serializable datetime."""
        d = super().dict(*args, **kwargs)
        if d['last_notification_time']:
            d['last_notification_time'] = d['last_notification_time'].isoformat()
        d['notification_history'] = [nh.dict() for nh in self.notification_history]
        return d
    
    def record_attempt(
        self, 
        notification: Dict,
        had_active_connection: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Record a notification attempt.
        
        Args:
            notification: The notification that was attempted
            had_active_connection: Whether there was an active connection
            error: Optional error message if the attempt failed
        """
        successful = error is None
        timestamp = datetime.utcnow()
        
        # Create attempt record
        attempt = INotificationAttempt(
            notification=notification,
            timestamp=timestamp,
            successful=successful,
            had_active_connection=had_active_connection,
            error=error
        )
        
        # Update state
        self.notification_history.append(attempt)
        self.last_notification_time = timestamp
        self.notification_count += 1
        
        if successful:
            self.successful_notifications += 1
            # Remove from pending if it was there
            for i, pending in enumerate(self.pending_notifications):
                if pending == notification:
                    self.pending_notifications.pop(i)
                    break
        else:
            self.failed_notifications += 1
            # Add to pending if not already there
            if notification not in self.pending_notifications:
                self.pending_notifications.append(notification)
       
