#!/usr/bin/env python3
"""
Monitoring and Metrics Module
Handles external logging, health metrics, and monitoring integration
"""

import time
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import psutil
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    """Health metric data structure"""
    timestamp: str
    service: str
    status: str
    model_loaded: bool
    device_type: str
    memory_usage_mb: float
    gpu_memory_gb: Optional[float]
    active_requests: int
    total_requests: int
    error_count: int
    avg_response_time_ms: float

@dataclass
class ErrorMetric:
    """Error metric data structure"""
    timestamp: str
    service: str
    error_type: str
    error_message: str
    endpoint: str
    user_agent: Optional[str]
    request_id: Optional[str]

class MetricsCollector:
    """Collects and manages metrics for monitoring"""
    
    def __init__(self, service_name: str = "ultravox_api"):
        self.service_name = service_name
        self.start_time = time.time()
        self.total_requests = 0
        self.error_count = 0
        self.active_requests = 0
        self.response_times: List[float] = []
        self.max_response_times = 100  # Keep last 100 response times
        
        # External monitoring configuration
        self.external_logging_enabled = os.getenv("EXTERNAL_LOGGING_ENABLED", "false").lower() == "true"
        self.metrics_endpoint = os.getenv("METRICS_ENDPOINT")
        self.log_file_path = os.getenv("LOG_FILE_PATH", "/var/log/ultravox/metrics.log")
        
        # Ensure log directory exists
        if self.external_logging_enabled:
            Path(self.log_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    def start_request(self) -> str:
        """Start tracking a request"""
        self.active_requests += 1
        self.total_requests += 1
        return str(int(time.time() * 1000))  # Request ID
    
    def end_request(self, request_id: str, response_time_ms: float, success: bool = True):
        """End tracking a request"""
        self.active_requests = max(0, self.active_requests - 1)
        
        if not success:
            self.error_count += 1
        
        # Track response time
        self.response_times.append(response_time_ms)
        if len(self.response_times) > self.max_response_times:
            self.response_times.pop(0)
    
    def get_health_metric(self, model_loaded: bool, device_type: str, gpu_memory_gb: Optional[float] = None) -> HealthMetric:
        """Get current health metric"""
        return HealthMetric(
            timestamp=datetime.now(timezone.utc).isoformat(),
            service=self.service_name,
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            device_type=device_type,
            memory_usage_mb=self.get_memory_usage(),
            gpu_memory_gb=gpu_memory_gb,
            active_requests=self.active_requests,
            total_requests=self.total_requests,
            error_count=self.error_count,
            avg_response_time_ms=self.get_avg_response_time()
        )
    
    def log_error(self, error_type: str, error_message: str, endpoint: str, 
                  user_agent: Optional[str] = None, request_id: Optional[str] = None):
        """Log an error metric"""
        error_metric = ErrorMetric(
            timestamp=datetime.now(timezone.utc).isoformat(),
            service=self.service_name,
            error_type=error_type,
            error_message=error_message,
            endpoint=endpoint,
            user_agent=user_agent,
            request_id=request_id
        )
        
        self.error_count += 1
        
        if self.external_logging_enabled:
            self._write_metric_to_file("error", asdict(error_metric))
        
        logger.error(f"Error [{error_type}] in {endpoint}: {error_message}")
    
    def log_health_metric(self, health_metric: HealthMetric):
        """Log health metric"""
        if self.external_logging_enabled:
            self._write_metric_to_file("health", asdict(health_metric))
        
        logger.info(f"Health check - Status: {health_metric.status}, "
                   f"Memory: {health_metric.memory_usage_mb:.1f}MB, "
                   f"Active requests: {health_metric.active_requests}")
    
    def _write_metric_to_file(self, metric_type: str, data: Dict[str, Any]):
        """Write metric to external log file"""
        try:
            log_entry = {
                "type": metric_type,
                "data": data
            }
            
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write metric to file: {e}")
    
    async def send_metrics_to_endpoint(self, metrics: Dict[str, Any]):
        """Send metrics to external endpoint (if configured)"""
        if not self.metrics_endpoint:
            return
        
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.metrics_endpoint,
                    json=metrics,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to send metrics to endpoint: {response.status}")
        except Exception as e:
            logger.warning(f"Error sending metrics to endpoint: {e}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_avg_response_time(self) -> float:
        """Get average response time in milliseconds"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_uptime_seconds(self) -> float:
        """Get service uptime in seconds"""
        return time.time() - self.start_time
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            "service": self.service_name,
            "uptime_seconds": self.get_uptime_seconds(),
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.total_requests),
            "avg_response_time_ms": self.get_avg_response_time(),
            "memory_usage_mb": self.get_memory_usage(),
            "external_logging_enabled": self.external_logging_enabled
        }
    
    def cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info("Cleaning up metrics collector...")
        # Reset counters
        self.total_requests = 0
        self.error_count = 0
        self.active_requests = 0
        self.response_times.clear()
        logger.info("Metrics collector cleanup completed")

class RequestContext:
    """Context manager for request tracking"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.request_id = None
        self.start_time = None
    
    async def __aenter__(self):
        self.request_id = self.metrics_collector.start_request()
        self.start_time = time.time()
        return self.request_id
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            response_time_ms = (time.time() - self.start_time) * 1000
            success = exc_type is None
            self.metrics_collector.end_request(self.request_id, response_time_ms, success)

# Global metrics collector
metrics_collector = MetricsCollector()

