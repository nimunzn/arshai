===============
Scaling Arshai
===============

Guide for scaling Arshai-based applications to handle increased load.

.. note::
   This section covers scaling applications **built with** Arshai.

Overview
========

Arshai applications can be scaled using various strategies:

- **Horizontal Scaling**: Multiple application instances
- **Vertical Scaling**: Increased resources per instance
- **Caching**: Redis-based caching and session management
- **Load Balancing**: Distributing traffic across instances

Horizontal Scaling
==================

Multi-Instance Deployment
-------------------------

Deploy multiple instances of your Arshai application:

.. code-block:: yaml

   # Kubernetes deployment example
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: arshai-app
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: arshai-app
     template:
       metadata:
         labels:
           app: arshai-app
       spec:
         containers:
         - name: app
           image: your-arshai-app:latest
           env:
           - name: REDIS_URL
             value: "redis://redis-service:6379"

Load Balancing
--------------

Use a load balancer to distribute requests:

.. code-block:: yaml

   # Kubernetes service example
   apiVersion: v1
   kind: Service
   metadata:
     name: arshai-service
   spec:
     selector:
       app: arshai-app
     ports:
     - port: 80
       targetPort: 8000
     type: LoadBalancer

Caching Strategies
==================

Redis Configuration
-------------------

Configure Redis for optimal performance:

.. code-block:: python

   from arshai import Settings

   settings = Settings(
       memory_provider="redis",
       redis_host="redis-cluster.example.com",
       redis_port=6379,
       redis_db=0,
       redis_max_connections=20,
       redis_retry_on_timeout=True
   )

Memory Management
-----------------

Implement effective memory management:

.. code-block:: python

   from arshai.memory import MemoryManager

   # Configure memory with TTL
   memory_manager = MemoryManager(
       provider="redis",
       ttl_seconds=3600,  # 1 hour
       max_conversations=1000
   )

Performance Optimization
========================

Async Operations
----------------

Use async operations for better concurrency:

.. code-block:: python

   import asyncio
   from arshai import Settings, IAgentConfig, IAgentInput

   async def process_multiple_requests(requests):
       settings = Settings()
       agent = settings.create_agent("conversation", config)
       
       tasks = [
           agent.aprocess_message(request) 
           for request in requests
       ]
       
       results = await asyncio.gather(*tasks)
       return results

Connection Pooling
------------------

Implement connection pooling for external services:

.. code-block:: python

   # Configure HTTP client with connection pooling
   import aiohttp

   async def create_http_session():
       connector = aiohttp.TCPConnector(
           limit=100,  # Total connection limit
           limit_per_host=20,  # Per-host connection limit
           ttl_dns_cache=300,  # DNS cache TTL
           use_dns_cache=True
       )
       
       return aiohttp.ClientSession(connector=connector)

Monitoring and Metrics
======================

Health Checks
-------------

Implement health check endpoints:

.. code-block:: python

   from fastapi import FastAPI
   from arshai import Settings

   app = FastAPI()

   @app.get("/health")
   async def health_check():
       try:
           # Test Redis connection
           settings = Settings()
           memory = settings.create_memory_manager()
           await memory.health_check()
           
           return {"status": "healthy"}
       except Exception as e:
           return {"status": "unhealthy", "error": str(e)}

Metrics Collection
------------------

Collect performance metrics:

.. code-block:: python

   import time
   from prometheus_client import Counter, Histogram

   # Metrics
   request_count = Counter('requests_total', 'Total requests')
   request_duration = Histogram('request_duration_seconds', 'Request duration')

   async def process_with_metrics(agent_input):
       request_count.inc()
       
       start_time = time.time()
       try:
           result = await agent.aprocess_message(agent_input)
           return result
       finally:
           duration = time.time() - start_time
           request_duration.observe(duration)

Auto-Scaling
============

Kubernetes HPA
--------------

Configure Horizontal Pod Autoscaler:

.. code-block:: yaml

   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: arshai-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: arshai-app
     minReplicas: 2
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70

Cloud Auto-Scaling
-------------------

Configure cloud provider auto-scaling groups based on metrics like:

- CPU utilization
- Memory usage
- Request queue length
- Response time

This section will be expanded with more detailed scaling patterns and cloud-specific examples.