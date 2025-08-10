"""
Performance monitoring module for the Arshai framework.

This module provides comprehensive performance monitoring and metrics collection
for async operations, memory usage, and system health.
"""

import asyncio
import logging
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Deque
from contextlib import asynccontextmanager
import threading

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = "ms"


@dataclass
class OperationStats:
    """Statistics for a specific operation type."""
    total_count: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update(self, duration: float, is_error: bool = False):
        """Update stats with new measurement."""
        self.total_count += 1
        if is_error:
            self.error_count += 1
        else:
            self.total_duration += duration
            self.avg_duration = self.total_duration / (self.total_count - self.error_count)
            self.min_duration = min(self.min_duration, duration)
            self.max_duration = max(self.max_duration, duration)
        self.last_updated = datetime.now()


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for Arshai.
    
    Tracks async operations, memory usage, system metrics, and provides
    performance insights for optimization.
    """
    
    def __init__(self, max_history: int = 10000, cleanup_interval: int = 300):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_history = max_history
        self.cleanup_interval = cleanup_interval
        
        # Metrics storage
        self.metrics: Deque[PerformanceMetric] = deque(maxlen=max_history)
        self.operation_stats: Dict[str, OperationStats] = defaultdict(OperationStats)
        
        # System metrics
        self.system_metrics: Dict[str, Any] = {}
        self.last_system_check = datetime.now()
        
        # Background tasks and monitoring
        self._monitoring_active = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._system_monitor_task: Optional[asyncio.Task] = None
        
        # Thread-safe operations
        self._lock = threading.RLock()
        
        logger.info(f"Initialized PerformanceMonitor with max_history: {max_history}")
    
    def start_monitoring(self):
        """Start background monitoring tasks."""
        if self._monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self._monitoring_active = True
        
        # Start background tasks if event loop is available
        try:
            loop = asyncio.get_event_loop()
            if not self._cleanup_task or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            if not self._system_monitor_task or self._system_monitor_task.done():
                self._system_monitor_task = asyncio.create_task(self._system_monitoring())
        except RuntimeError:
            logger.info("No event loop available, monitoring will start when one is created")
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring tasks."""
        self._monitoring_active = False
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        if self._system_monitor_task and not self._system_monitor_task.done():
            self._system_monitor_task.cancel()
        
        logger.info("Performance monitoring stopped")
    
    def record_metric(self, name: str, value: float, unit: str = "ms", tags: Dict[str, str] = None):
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            tags: Additional tags for the metric
        """
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                unit=unit
            )
            self.metrics.append(metric)
            logger.debug(f"Recorded metric: {name}={value}{unit}")
    
    def record_operation(self, operation_name: str, duration: float, is_error: bool = False, tags: Dict[str, str] = None):
        """
        Record operation performance.
        
        Args:
            operation_name: Name of the operation
            duration: Duration in milliseconds
            is_error: Whether the operation failed
            tags: Additional tags
        """
        with self._lock:
            # Update operation stats
            self.operation_stats[operation_name].update(duration, is_error)
            
            # Record as metric
            metric_tags = {"operation": operation_name, "status": "error" if is_error else "success"}
            if tags:
                metric_tags.update(tags)
            
            self.record_metric(f"operation.{operation_name}", duration, "ms", metric_tags)
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str, tags: Dict[str, str] = None):
        """
        Context manager for monitoring async operations.
        
        Args:
            operation_name: Name of the operation to monitor
            tags: Additional tags for the operation
            
        Usage:
            async with monitor.monitor_operation("vector_search"):
                result = await vector_db.search(...)
        """
        start_time = time.perf_counter()
        is_error = False
        
        try:
            yield
        except Exception as e:
            is_error = True
            logger.error(f"Operation {operation_name} failed: {str(e)}")
            raise
        finally:
            duration = (time.perf_counter() - start_time) * 1000  # Convert to ms
            self.record_operation(operation_name, duration, is_error, tags)
    
    def monitor_function(self, operation_name: str = None, tags: Dict[str, str] = None):
        """
        Decorator for monitoring function performance.
        
        Args:
            operation_name: Custom operation name (defaults to function name)
            tags: Additional tags
            
        Usage:
            @monitor.monitor_function("custom_operation")
            async def my_async_function():
                pass
        """
        def decorator(func: Callable):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    async with self.monitor_operation(op_name, tags):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    start_time = time.perf_counter()
                    is_error = False
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        is_error = True
                        raise
                    finally:
                        duration = (time.perf_counter() - start_time) * 1000
                        self.record_operation(op_name, duration, is_error, tags)
                return sync_wrapper
        
        return decorator
    
    def get_operation_stats(self, operation_name: str = None) -> Dict[str, OperationStats]:
        """
        Get operation statistics.
        
        Args:
            operation_name: Specific operation name, or None for all operations
            
        Returns:
            Dictionary of operation statistics
        """
        with self._lock:
            if operation_name:
                return {operation_name: self.operation_stats.get(operation_name, OperationStats())}
            return dict(self.operation_stats)
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """
        Get metrics summary for the last N hours.
        
        Args:
            hours: Number of hours to include in summary
            
        Returns:
            Summary statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {"total_metrics": 0, "time_range_hours": hours}
            
            # Group by metric name
            by_name = defaultdict(list)
            for metric in recent_metrics:
                by_name[metric.name].append(metric.value)
            
            summary = {
                "total_metrics": len(recent_metrics),
                "time_range_hours": hours,
                "metrics": {}
            }
            
            for name, values in by_name.items():
                summary["metrics"][name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "total": sum(values)
                }
            
            return summary
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get current system health metrics.
        
        Returns:
            System health information
        """
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Network (if available)
            network = {}
            try:
                net_io = psutil.net_io_counters()
                network = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            except:
                pass
            
            health = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                },
                "process": {
                    "memory_rss": process_memory.rss,
                    "memory_vms": process_memory.vms,
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads()
                },
                "network": network
            }
            
            return health
            
        except Exception as e:
            logger.error(f"Error collecting system health: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Complete performance report
        """
        with self._lock:
            report = {
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": self._monitoring_active,
                "metrics_count": len(self.metrics),
                "operations_tracked": len(self.operation_stats),
                "system_health": self.get_system_health(),
                "metrics_summary_1h": self.get_metrics_summary(1),
                "metrics_summary_24h": self.get_metrics_summary(24),
                "top_operations": {}
            }
            
            # Top operations by different metrics
            if self.operation_stats:
                # Sort by average duration
                sorted_by_avg = sorted(
                    self.operation_stats.items(),
                    key=lambda x: x[1].avg_duration,
                    reverse=True
                )
                report["top_operations"]["slowest_avg"] = [
                    {
                        "name": name,
                        "avg_duration": stats.avg_duration,
                        "total_count": stats.total_count
                    }
                    for name, stats in sorted_by_avg[:5]
                ]
                
                # Sort by error rate
                sorted_by_errors = sorted(
                    self.operation_stats.items(),
                    key=lambda x: x[1].error_count / max(x[1].total_count, 1),
                    reverse=True
                )
                report["top_operations"]["highest_error_rate"] = [
                    {
                        "name": name,
                        "error_rate": stats.error_count / max(stats.total_count, 1),
                        "error_count": stats.error_count,
                        "total_count": stats.total_count
                    }
                    for name, stats in sorted_by_errors[:5]
                ]
            
            return report
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old metrics and stats."""
        while self._monitoring_active:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                with self._lock:
                    # Clean up old operation stats (older than 24 hours)
                    cutoff = datetime.now() - timedelta(hours=24)
                    ops_to_remove = [
                        name for name, stats in self.operation_stats.items()
                        if stats.last_updated < cutoff and stats.total_count == 0
                    ]
                    
                    for op_name in ops_to_remove:
                        del self.operation_stats[op_name]
                    
                    if ops_to_remove:
                        logger.debug(f"Cleaned up {len(ops_to_remove)} old operation stats")
                
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {str(e)}")
    
    async def _system_monitoring(self):
        """Periodic system metrics collection."""
        while self._monitoring_active:
            try:
                # Collect system metrics every 60 seconds
                await asyncio.sleep(60)
                
                health = self.get_system_health()
                if "error" not in health:
                    # Record system metrics
                    self.record_metric("system.cpu.percent", health["cpu"]["percent"], "%")
                    self.record_metric("system.memory.percent", health["memory"]["percent"], "%")
                    self.record_metric("system.disk.percent", health["disk"]["percent"], "%")
                    self.record_metric("process.memory.rss", health["process"]["memory_rss"], "bytes")
                    self.record_metric("process.cpu.percent", health["process"]["cpu_percent"], "%")
                    self.record_metric("process.threads", health["process"]["num_threads"], "count")
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {str(e)}")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_monitoring()


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor


# Convenience decorators and functions
def monitor_async_operation(operation_name: str = None, tags: Dict[str, str] = None):
    """Convenience decorator for monitoring async operations."""
    return get_performance_monitor().monitor_function(operation_name, tags)


@asynccontextmanager
async def monitor_operation(operation_name: str, tags: Dict[str, str] = None):
    """Convenience context manager for monitoring operations."""
    async with get_performance_monitor().monitor_operation(operation_name, tags):
        yield


def record_metric(name: str, value: float, unit: str = "ms", tags: Dict[str, str] = None):
    """Convenience function to record a metric."""
    get_performance_monitor().record_metric(name, value, unit, tags)


def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report."""
    return get_performance_monitor().get_performance_report()