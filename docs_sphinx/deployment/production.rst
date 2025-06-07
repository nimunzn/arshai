=================
Production Deployment
=================

Guide for deploying Arshai-based applications in production environments.

.. note::
   This section covers deploying applications **built with** Arshai, not deploying the Arshai package itself.

Overview
========

When deploying applications built with Arshai, consider these key areas:

- **Environment Configuration**: Managing secrets and settings
- **Scaling**: Handling concurrent requests and load
- **Monitoring**: Observability and health checks
- **Security**: Authentication, authorization, and data protection

Environment Setup
=================

Configuration Management
------------------------

Use environment variables for configuration:

.. code-block:: python

   import os
   from arshai import Settings

   # Configure via environment variables
   settings = Settings(
       openai_api_key=os.getenv("OPENAI_API_KEY"),
       azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
       redis_url=os.getenv("REDIS_URL"),
       # ... other settings
   )

Container Deployment
--------------------

Example Dockerfile:

.. code-block:: dockerfile

   FROM python:3.11-slim

   WORKDIR /app

   # Install dependencies
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   # Copy application
   COPY . .

   # Run application
   CMD ["python", "app.py"]

Example docker-compose.yml:

.. code-block:: yaml

   version: '3.8'
   services:
     app:
       build: .
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - REDIS_URL=redis://redis:6379
       depends_on:
         - redis
       ports:
         - "8000:8000"
     
     redis:
       image: redis:7-alpine
       volumes:
         - redis_data:/data

   volumes:
     redis_data:

Best Practices
==============

Security
--------

- Store API keys in environment variables or secret managers
- Use HTTPS for all external communications
- Implement proper authentication and authorization
- Validate all inputs and sanitize outputs

Performance
-----------

- Use Redis for caching and session storage
- Implement connection pooling for external APIs
- Set appropriate timeouts for LLM calls
- Use async/await for concurrent operations

Reliability
-----------

- Implement retry logic with exponential backoff
- Use circuit breakers for external service calls
- Set up health checks and readiness probes
- Implement graceful shutdown handling

This section will be expanded with more detailed production deployment patterns and examples.