# Arshai Deployment Guide

## Infrastructure Requirements

### Compute Resources

#### Minimum Requirements
| Component | CPU | RAM | Storage | Network | Notes |
|-----------|-----|-----|---------|---------|-------|
| Arshai Core | 2 vCPU | 4 GB | 20 GB SSD | 100 Mbps | Basic setup without document processing |
| Document Processing | 4 vCPU | 8 GB | 40 GB SSD | 100 Mbps | PDF processing is CPU/memory intensive |
| Vector Database | 2 vCPU | 8 GB | 100 GB SSD | 100 Mbps | For small to medium document collections |
| Redis Cache | 1 vCPU | 2 GB | 10 GB SSD | 100 Mbps | For memory and caching |

#### Recommended Production Environment
| Component | CPU | RAM | Storage | Network | Notes |
|-----------|-----|-----|---------|---------|-------|
| Arshai Core | 4 vCPU | 8 GB | 40 GB SSD | 1 Gbps | Handles multiple concurrent workflows |
| Document Processing | 8 vCPU | 16 GB | 100 GB SSD | 1 Gbps | Fast document processing at scale |
| Vector Database | 8 vCPU | 32 GB | 500 GB SSD | 1 Gbps | For large document collections with frequent queries |
| Redis Cache | 2 vCPU | 8 GB | 50 GB SSD | 1 Gbps | For distributed deployments |
| Database | 4 vCPU | 16 GB | 200 GB SSD | 1 Gbps | For workflow state and configuration |

### Cloud Provider Options

| Provider | Services | Advantages | Considerations |
|----------|----------|------------|----------------|
| AWS | EC2, ECS, Lambda, RDS, ElastiCache, OpenSearch | Comprehensive ecosystem, scalability | Cost can be higher, complex setup |
| Azure | Azure Container Apps, AKS, Cosmos DB, Azure Cache | Strong integration with Azure OpenAI | Best if already using Azure ecosystem |
| GCP | GKE, Cloud Run, Cloud SQL, Memorystore | Good network performance, ML focus | Fewer managed services specific to vector DBs |
| Digital Ocean | Droplets, Managed Databases, App Platform | Simplicity, predictable pricing | Fewer advanced features |

### Self-Hosted Requirements
- Linux-based operating system (Ubuntu 20.04+ recommended)
- Docker and Docker Compose for containerized deployment
- Python 3.11+ for non-containerized deployment
- Nginx or similar for HTTP proxy and TLS termination
- Firewall configuration for security

## Containerization Strategy

### Container Architecture

```mermaid
graph TD
    subgraph "Arshai Containers"
        API[API Service] --> Core[Arshai Core]
        Core --> DocProc[Document Processor]
        API --> HealthCheck[Health Monitor]
        WorkRunner[Workflow Runner] --> Core
    end
    
    subgraph "Infrastructure Containers"
        Core --> Redis[Redis]
        Core --> VectorDB[Vector Database]
        Core --> DB[PostgreSQL]
    end
    
    subgraph "External Services"
        Core --> LLM[LLM Provider API]
        DocProc --> Storage[Object Storage]
    end
    
    Client --> API
    Client --> WorkRunner
```

### Containerization Approach
1. **Microservices Architecture**:
   - API Service container
   - Workflow Runner container
   - Document Processing container
   - Health Check container

2. **Base Images**:
   - `python:3.11-slim` for primary services
   - Specialized base images for infrastructure components

3. **Container Configuration**:
   - Environment variables for service configuration
   - Volume mounts for persistent storage
   - Health check endpoints for all services
   - Resource limits and requests

### Example Docker Compose Configuration
```yaml
version: '3.8'

services:
  arshai-api:
    image: arshai/api:latest
    ports:
      - "8000:8000"
    environment:
      - ARSHAI_ENV=production
      - ARSHAI_CONFIG_PATH=/app/config/config.yaml
      - ARSHAI_DB_URL=postgresql://user:password@postgres:5432/arshai
      - ARSHAI_REDIS_URL=redis://redis:6379/0
      - ARSHAI_VECTOR_DB_URL=http://milvus:19530
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
      - milvus
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  workflow-runner:
    image: arshai/workflow-runner:latest
    environment:
      - ARSHAI_ENV=production
      - ARSHAI_CONFIG_PATH=/app/config/config.yaml
      - ARSHAI_DB_URL=postgresql://user:password@postgres:5432/arshai
      - ARSHAI_REDIS_URL=redis://redis:6379/0
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  document-processor:
    image: arshai/document-processor:latest
    environment:
      - ARSHAI_ENV=production
      - ARSHAI_CONFIG_PATH=/app/config/config.yaml
      - ARSHAI_VECTOR_DB_URL=http://milvus:19530
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
      - ./documents:/app/documents
    depends_on:
      - milvus
    restart: unless-stopped

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=arshai
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d arshai"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7
    volumes:
      - redis-data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  milvus:
    image: milvusdb/milvus:v2.3.0
    volumes:
      - milvus-data:/var/lib/milvus
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s

volumes:
  postgres-data:
  redis-data:
  milvus-data:
```

## Orchestration Mechanisms

### Kubernetes Deployment

Key components:
- **Deployments** for stateless services
- **StatefulSets** for database components
- **Services** for internal communication
- **Ingress** for external access
- **ConfigMaps** and **Secrets** for configuration
- **PersistentVolumeClaims** for storage
- **HorizontalPodAutoscalers** for scaling

Example deployment manifest structure:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arshai-api
  namespace: arshai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: arshai-api
  template:
    metadata:
      labels:
        app: arshai-api
    spec:
      containers:
      - name: api
        image: arshai/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ARSHAI_ENV
          value: "production"
        - name: ARSHAI_DB_URL
          valueFrom:
            secretKeyRef:
              name: arshai-secrets
              key: db-url
        - name: ARSHAI_REDIS_URL
          valueFrom:
            secretKeyRef:
              name: arshai-secrets
              key: redis-url
        - name: ARSHAI_OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: arshai-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
```

### Serverless Options

For event-driven or request-based usage patterns:
- AWS Lambda with API Gateway
- Azure Functions
- Google Cloud Functions
- Serverless Framework for deployment

Considerations:
- Cold start latency
- Memory constraints
- Execution time limits
- State management between invocations

## Scaling Considerations

### Horizontal Scaling
- API and Workflow services are stateless and horizontally scalable
- Use load balancers to distribute traffic
- Session affinity not required due to external state storage

### Vertical Scaling
- Document Processing benefits from additional CPU/RAM
- Vector databases benefit from higher memory and faster storage
- Database read replicas for scaling read operations

### Scaling Limits
| Component | Scaling Pattern | Practical Limits | Bottlenecks |
|-----------|-----------------|------------------|-------------|
| API Service | Horizontal | 100+ instances | Rate limits on LLM APIs |
| Workflow Runner | Horizontal | 50+ instances | Database connections |
| Document Processor | Both | CPU/memory bound | Processing speed and embedding generation |
| Vector Database | Cluster | Depends on provider | Query latency at scale |
| Redis | Cluster | Memory bound | Network latency |
| PostgreSQL | Primary + Replicas | Write throughput | Transaction volume |

## Zero-Downtime Deployment

### Blue-Green Deployment
1. Deploy new version alongside existing version
2. Run health checks on new deployment
3. Switch traffic gradually from old to new
4. Monitor for errors
5. Keep old deployment as fallback
6. Remove old deployment after confirmation

### Canary Releases
1. Deploy new version to a small subset of infrastructure
2. Route a percentage of traffic to new version
3. Monitor for errors and performance issues
4. Gradually increase traffic to new version
5. Complete transition when confident

### Database Migrations
- Use migration frameworks with versioning (Alembic)
- Ensure backward compatibility for one version
- Run migrations before code deployment
- Have rollback plans for failed migrations
- Use database transaction features for safety

## Monitoring and Observability

### Monitoring Components

```mermaid
graph TD
    subgraph "Arshai System"
        API[API Service]
        WF[Workflow System]
        DP[Document Processing]
        VDB[Vector Database]
    end
    
    subgraph "Monitoring Stack"
        Prom[Prometheus]
        Graf[Grafana]
        Loki[Loki]
        Alert[AlertManager]
    end
    
    API --> Prom
    WF --> Prom
    DP --> Prom
    VDB --> Prom
    
    API --> Loki
    WF --> Loki
    DP --> Loki
    
    Prom --> Graf
    Loki --> Graf
    Prom --> Alert
```

### Key Metrics
| Metric | Type | Description | Alert Threshold |
|--------|------|-------------|----------------|
| `api_request_count` | Counter | API request volume | N/A (trending) |
| `api_request_duration_seconds` | Histogram | API response time | p95 > 2s |
| `workflow_execution_count` | Counter | Workflow executions | N/A (trending) |
| `workflow_execution_duration_seconds` | Histogram | Workflow execution time | p95 > 30s |
| `llm_request_count` | Counter | LLM API calls | N/A (trending) |
| `llm_token_usage` | Counter | Token consumption | Cost thresholds |
| `llm_request_failures` | Counter | Failed LLM calls | > 5% error rate |
| `document_processing_duration_seconds` | Histogram | Document processing time | p95 > 60s |
| `vector_query_duration_seconds` | Histogram | Vector DB query time | p95 > 1s |
| `memory_usage_bytes` | Gauge | Memory consumption | > 80% capacity |
| `cpu_usage_percent` | Gauge | CPU utilization | > 80% utilization |
| `disk_free_bytes` | Gauge | Available disk space | < 20% free |

### Logging Strategy
- Structured JSON logging
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Contextual information (request ID, user ID)
- Sensitive data filtering
- Log aggregation with Loki or ELK stack
- Log retention policies

### Tracing
- Distributed tracing with OpenTelemetry
- Trace context propagation
- Span collection for performance analysis
- Integration with Jaeger or Zipkin
- Sampling strategies for high-volume systems

## Disaster Recovery

### Backup Procedures
| Component | Backup Method | Frequency | Retention |
|-----------|---------------|-----------|-----------|
| Configuration | Git repository | On change | Indefinite |
| Database | Automated dumps | Hourly | 7 days |
| Vector Database | Export collections | Daily | 30 days |
| Document Storage | Object storage replication | Continuous | 90 days |
| Logs | Log archiving | Daily | 90 days |

### Recovery Procedures
1. **Infrastructure Failure**:
   - Deploy to alternative region/zone
   - Restore from infrastructure-as-code templates
   - Verify configuration

2. **Data Corruption**:
   - Identify corruption scope
   - Stop affected services
   - Restore from last known good backup
   - Verify data integrity
   - Resume services

3. **Complete Disaster**:
   - Deploy infrastructure from templates
   - Restore databases from backups
   - Verify system functionality
   - Update DNS or load balancer configuration
   - Resume operation in new environment

### Business Continuity
- Recovery Time Objective (RTO): 1 hour for critical functions
- Recovery Point Objective (RPO): 15 minutes for transaction data
- Multi-region deployment options for high availability
- Regular disaster recovery testing

## Security Hardening

### Network Security
- Private subnets for internal services
- Public endpoints only for API gateway
- Network ACLs and security groups
- TLS for all communications
- API rate limiting
- DDoS protection

### Container Security
- Minimal base images
- Non-root users
- Read-only file systems where possible
- Resource limits
- Regular security updates
- Container image scanning

### Authentication and Authorization
- API key rotation policies
- OAuth2 for user authentication
- Role-based access control
- Principle of least privilege
- Audit logging for security events

### Data Security
- Encryption at rest for all storage
- Encryption in transit for all communications
- Data minimization practices
- Sensitive data identification and special handling
- Regular security audits

## Compliance Considerations

### Data Privacy
- GDPR compliance capabilities
- Data residency options
- Data retention policies
- User data export and deletion capabilities
- Privacy impact assessments

### Industry Standards
- SOC 2 compliance readiness
- HIPAA compatibility (with additional controls)
- PCI DSS considerations for payment information
- Documentation for compliance audits

### Security Assessments
- Regular penetration testing
- Vulnerability scanning
- Security code reviews
- Third-party security audits
- Responsible disclosure program 