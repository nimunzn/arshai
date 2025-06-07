================
Monitoring Arshai
================

Guide for monitoring Arshai-based applications in production.

.. note::
   This section covers monitoring applications **built with** Arshai.

Overview
========

Effective monitoring of Arshai applications includes:

- **Application Metrics**: Performance and usage statistics
- **Health Monitoring**: Service availability and health checks
- **Logging**: Structured logging for debugging and analysis
- **Alerting**: Proactive notifications for issues

Application Metrics
===================

Key Metrics to Track
-------------------

**Request Metrics**:
- Request count and rate
- Response time percentiles (p50, p95, p99)
- Error rates and types
- Queue depth and wait times

**Agent Metrics**:
- Agent response time
- Token usage and costs
- Memory utilization
- Tool execution frequency

**System Metrics**:
- CPU and memory usage
- Network I/O
- Redis connection pool status
- External API response times

Prometheus Integration
---------------------

Example metrics implementation:

.. code-block:: python

   from prometheus_client import Counter, Histogram, Gauge, start_http_server
   import time

   # Define metrics
   request_count = Counter(
       'arshai_requests_total', 
       'Total requests processed',
       ['agent_type', 'status']
   )

   request_duration = Histogram(
       'arshai_request_duration_seconds',
       'Request processing time',
       ['agent_type']
   )

   active_conversations = Gauge(
       'arshai_active_conversations',
       'Number of active conversations'
   )

   token_usage = Counter(
       'arshai_tokens_used_total',
       'Total tokens consumed',
       ['model', 'type']
   )

   # Instrument your agent
   class MonitoredAgent:
       def __init__(self, agent, agent_type):
           self.agent = agent
           self.agent_type = agent_type

       async def process_message(self, input_data):
           start_time = time.time()
           
           try:
               result = await self.agent.aprocess_message(input_data)
               request_count.labels(
                   agent_type=self.agent_type, 
                   status='success'
               ).inc()
               
               # Track token usage if available
               if hasattr(result, 'usage'):
                   token_usage.labels(
                       model=result.model,
                       type='completion'
                   ).inc(result.usage.completion_tokens)
               
               return result
               
           except Exception as e:
               request_count.labels(
                   agent_type=self.agent_type, 
                   status='error'
               ).inc()
               raise
               
           finally:
               duration = time.time() - start_time
               request_duration.labels(
                   agent_type=self.agent_type
               ).observe(duration)

   # Start metrics server
   start_http_server(8001)

Health Monitoring
=================

Health Check Endpoints
----------------------

Implement comprehensive health checks:

.. code-block:: python

   from fastapi import FastAPI, HTTPException
   from arshai import Settings
   import asyncio

   app = FastAPI()

   @app.get("/health")
   async def health_check():
       """Basic health check"""
       return {"status": "healthy", "timestamp": time.time()}

   @app.get("/health/detailed")
   async def detailed_health_check():
       """Detailed health check with dependencies"""
       checks = {}
       overall_status = "healthy"
       
       # Check Redis connection
       try:
           settings = Settings()
           memory = settings.create_memory_manager()
           await memory.ping()
           checks["redis"] = {"status": "healthy"}
       except Exception as e:
           checks["redis"] = {"status": "unhealthy", "error": str(e)}
           overall_status = "unhealthy"
       
       # Check LLM provider
       try:
           llm = settings.create_llm()
           # Simple test call
           await llm.test_connection()
           checks["llm"] = {"status": "healthy"}
       except Exception as e:
           checks["llm"] = {"status": "degraded", "error": str(e)}
           if overall_status == "healthy":
               overall_status = "degraded"
       
       if overall_status != "healthy":
           raise HTTPException(status_code=503, detail=checks)
       
       return {"status": overall_status, "checks": checks}

   @app.get("/ready")
   async def readiness_check():
       """Kubernetes readiness probe"""
       try:
           # Ensure all critical services are available
           settings = Settings()
           memory = settings.create_memory_manager()
           await memory.ping()
           return {"status": "ready"}
       except Exception:
           raise HTTPException(status_code=503, detail="Not ready")

Structured Logging
==================

Logging Configuration
--------------------

Configure structured logging:

.. code-block:: python

   import logging
   import json
   from datetime import datetime
   from typing import Dict, Any

   class JSONFormatter(logging.Formatter):
       def format(self, record):
           log_entry = {
               "timestamp": datetime.utcnow().isoformat(),
               "level": record.levelname,
               "logger": record.name,
               "message": record.getMessage(),
               "module": record.module,
               "function": record.funcName,
               "line": record.lineno
           }
           
           # Add extra fields
           if hasattr(record, 'conversation_id'):
               log_entry['conversation_id'] = record.conversation_id
           
           if hasattr(record, 'user_id'):
               log_entry['user_id'] = record.user_id
           
           if hasattr(record, 'duration'):
               log_entry['duration'] = record.duration
           
           return json.dumps(log_entry)

   # Configure logging
   def setup_logging():
       logger = logging.getLogger("arshai")
       logger.setLevel(logging.INFO)
       
       handler = logging.StreamHandler()
       handler.setFormatter(JSONFormatter())
       logger.addHandler(handler)
       
       return logger

Application Logging
-------------------

Add contextual logging to your agents:

.. code-block:: python

   import logging
   from arshai import Settings, IAgentInput, IAgentOutput

   logger = logging.getLogger("arshai.agent")

   class LoggedAgent:
       def __init__(self, agent):
           self.agent = agent

       async def process_message(self, input_data: IAgentInput) -> IAgentOutput:
           start_time = time.time()
           
           logger.info(
               "Processing message",
               extra={
                   'conversation_id': input_data.conversation_id,
                   'user_id': input_data.user_id,
                   'message_length': len(input_data.message)
               }
           )
           
           try:
               result = await self.agent.aprocess_message(input_data)
               
               duration = time.time() - start_time
               logger.info(
                   "Message processed successfully",
                   extra={
                       'conversation_id': input_data.conversation_id,
                       'duration': duration,
                       'response_length': len(result.response),
                       'tokens_used': getattr(result, 'usage', {}).get('total_tokens', 0)
                   }
               )
               
               return result
               
           except Exception as e:
               duration = time.time() - start_time
               logger.error(
                   "Message processing failed",
                   extra={
                       'conversation_id': input_data.conversation_id,
                       'duration': duration,
                       'error': str(e),
                       'error_type': type(e).__name__
                   }
               )
               raise

Alerting
========

Alert Rules
-----------

Define alerting rules for critical issues:

.. code-block:: yaml

   # Prometheus alerting rules
   groups:
   - name: arshai_alerts
     rules:
     - alert: HighErrorRate
       expr: rate(arshai_requests_total{status="error"}[5m]) > 0.1
       for: 2m
       labels:
         severity: warning
       annotations:
         summary: "High error rate detected"
         description: "Error rate is {{ $value }} errors per second"

     - alert: SlowResponseTime
       expr: histogram_quantile(0.95, rate(arshai_request_duration_seconds_bucket[5m])) > 10
       for: 5m
       labels:
         severity: warning
       annotations:
         summary: "Slow response times detected"
         description: "95th percentile response time is {{ $value }} seconds"

     - alert: ServiceDown
       expr: up{job="arshai"} == 0
       for: 1m
       labels:
         severity: critical
       annotations:
         summary: "Arshai service is down"
         description: "Service has been down for more than 1 minute"

     - alert: HighMemoryUsage
       expr: arshai_active_conversations > 1000
       for: 5m
       labels:
         severity: warning
       annotations:
         summary: "High memory usage"
         description: "Active conversations: {{ $value }}"

Notification Channels
--------------------

Configure notification channels:

.. code-block:: python

   import smtplib
   import json
   from email.mime.text import MIMEText
   from slack_sdk.webhook import WebhookClient

   class AlertManager:
       def __init__(self, slack_webhook_url=None, smtp_config=None):
           self.slack_client = WebhookClient(slack_webhook_url) if slack_webhook_url else None
           self.smtp_config = smtp_config

       def send_alert(self, alert_type: str, message: str, severity: str = "warning"):
           """Send alert via configured channels"""
           
           if self.slack_client:
               self._send_slack_alert(alert_type, message, severity)
           
           if self.smtp_config:
               self._send_email_alert(alert_type, message, severity)

       def _send_slack_alert(self, alert_type: str, message: str, severity: str):
           color = {"critical": "danger", "warning": "warning", "info": "good"}.get(severity, "warning")
           
           self.slack_client.send(
               text=f"Arshai Alert: {alert_type}",
               attachments=[{
                   "color": color,
                   "fields": [
                       {"title": "Alert Type", "value": alert_type, "short": True},
                       {"title": "Severity", "value": severity, "short": True},
                       {"title": "Message", "value": message, "short": False}
                   ]
               }]
           )

Dashboard Examples
==================

Grafana Dashboard
----------------

Example Grafana dashboard configuration:

.. code-block:: json

   {
     "dashboard": {
       "title": "Arshai Application Monitoring",
       "panels": [
         {
           "title": "Request Rate",
           "type": "graph",
           "targets": [
             {
               "expr": "rate(arshai_requests_total[5m])",
               "legendFormat": "{{agent_type}} - {{status}}"
             }
           ]
         },
         {
           "title": "Response Time Percentiles",
           "type": "graph",
           "targets": [
             {
               "expr": "histogram_quantile(0.50, rate(arshai_request_duration_seconds_bucket[5m]))",
               "legendFormat": "50th percentile"
             },
             {
               "expr": "histogram_quantile(0.95, rate(arshai_request_duration_seconds_bucket[5m]))",
               "legendFormat": "95th percentile"
             }
           ]
         },
         {
           "title": "Active Conversations",
           "type": "singlestat",
           "targets": [
             {
               "expr": "arshai_active_conversations"
             }
           ]
         }
       ]
     }
   }

Best Practices
==============

Monitoring Strategy
-------------------

1. **Start Simple**: Begin with basic metrics and health checks
2. **Add Context**: Include correlation IDs and user context in logs
3. **Monitor Dependencies**: Track external service health and performance
4. **Set SLOs**: Define Service Level Objectives for your application
5. **Regular Reviews**: Regularly review and update monitoring setup

Security Considerations
----------------------

- **Sanitize Logs**: Never log sensitive information (API keys, personal data)
- **Access Control**: Restrict access to monitoring endpoints
- **Encryption**: Use TLS for metric and log transmission
- **Retention**: Set appropriate retention policies for logs and metrics

This section provides a foundation for monitoring Arshai applications. Specific implementations may vary based on your infrastructure and requirements.