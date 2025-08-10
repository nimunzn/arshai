# Layer 3: Agentic Systems

Agentic Systems represent the top layer of Arshai's three-layer architecture. This is where you have **maximum developer authority** - complete control over orchestration, composition, and system-level patterns.

## What Are Agentic Systems?

Agentic Systems are complex applications that orchestrate multiple agents, manage state, coordinate workflows, and integrate external services. You're the architect - the framework provides building blocks, but you control the design.

```python
# Layer 1: LLM Clients (minimal authority)
llm_client = OpenAIClient(config)

# Layer 2: Agents (moderate authority)  
analyzer = AnalysisAgent(llm_client)
summarizer = SummaryAgent(llm_client)

# Layer 3: Systems (maximum authority - YOU control everything)
class DocumentProcessingSystem:
    def __init__(self):
        self.analyzer = analyzer
        self.summarizer = summarizer  
        self.memory = RedisMemory()
        self.workflow = self._design_workflow()  # YOU design this
        
    async def process(self, document):
        # YOU control the orchestration
        return await self._orchestrate_processing(document)
```

## Core Principles

### 1. You Control the Architecture

Unlike Layer 1 (standardized) and Layer 2 (structured), Layer 3 gives you complete freedom:

```python
# You choose the patterns
class YourSystem:
    def __init__(self):
        # Pipeline pattern?
        self.pipeline = [agent1, agent2, agent3]
        
        # Event-driven?
        self.event_bus = EventBus()
        
        # Actor model?
        self.actors = ActorSystem()
        
        # Or something completely custom?
        self.custom_orchestrator = YourCustomDesign()
```

### 2. State Management is Your Choice

```python
class StatefulSystem:
    """Example: You manage all state."""
    
    def __init__(self):
        # You choose storage
        self.conversation_state = {}
        self.user_sessions = RedisStore()
        self.long_term_memory = PostgresStore()
        self.cache = MemcachedStore()
    
    async def process(self, input_data, user_id):
        # You control state flow
        session = await self.user_sessions.get(user_id)
        context = self.conversation_state.get(user_id, {})
        
        # Your processing logic
        result = await self._your_processing(input_data, session, context)
        
        # You update state
        self.conversation_state[user_id] = result.context
        await self.user_sessions.update(user_id, result.session)
        
        return result
```

### 3. Integration Patterns Are Up to You

```python
class IntegratedSystem:
    """Example: Your integration choices."""
    
    def __init__(self):
        # You choose external services
        self.webhook_handler = WebhookProcessor()
        self.message_queue = RabbitMQHandler()
        self.monitoring = PrometheusMetrics()
        self.alerts = SlackNotifier()
    
    async def handle_request(self, request):
        # You design the flow
        await self.monitoring.track("request_received")
        
        try:
            result = await self._process(request)
            await self.webhook_handler.notify_success(result)
            return result
        except Exception as e:
            await self.alerts.send_error(e)
            raise
```

## System Patterns

### Pattern 1: Pipeline System

Sequential processing through multiple agents:

```python
class ProcessingPipeline:
    """Sequential agent processing pipeline."""
    
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
    
    async def process(self, input_data: Any) -> Any:
        """Process through pipeline sequentially."""
        current_data = input_data
        
        for i, agent in enumerate(self.agents):
            try:
                # Each agent processes the output of the previous
                agent_input = IAgentInput(message=str(current_data))
                current_data = await agent.process(agent_input)
                
                print(f"Stage {i+1} completed: {type(current_data).__name__}")
                
            except Exception as e:
                print(f"Pipeline failed at stage {i+1}: {e}")
                raise
        
        return current_data

# Usage
llm_client = OpenAIClient(config)
pipeline = ProcessingPipeline([
    ExtractionAgent(llm_client, "Extract key information"),
    AnalysisAgent(llm_client, "Analyze extracted data"),
    SummaryAgent(llm_client, "Create summary")
])

result = await pipeline.process("Large document text...")
```

### Pattern 2: Parallel Processing System

Concurrent processing with result aggregation:

```python
class ParallelProcessor:
    """Process input through multiple agents in parallel."""
    
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
    
    async def process(self, input_data: str) -> Dict[str, Any]:
        """Process input through all agents concurrently."""
        
        # Create tasks for all agents
        tasks = {
            name: agent.process(IAgentInput(message=input_data))
            for name, agent in self.agents.items()
        }
        
        # Execute in parallel
        results = await asyncio.gather(
            *tasks.values(), 
            return_exceptions=True
        )
        
        # Combine results
        combined_results = {}
        for (name, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                combined_results[name] = f"Error: {result}"
            else:
                combined_results[name] = result
        
        return combined_results

# Usage
llm_client = OpenAIClient(config)
processor = ParallelProcessor({
    "sentiment": SentimentAgent(llm_client),
    "summary": SummaryAgent(llm_client),
    "keywords": KeywordAgent(llm_client),
    "category": CategoryAgent(llm_client)
})

results = await processor.process("Customer feedback text...")
# Returns: {"sentiment": "positive", "summary": "...", ...}
```

### Pattern 3: Router System

Route requests to appropriate agents based on criteria:

```python
class IntelligentRouter:
    """Route requests to the most appropriate agent."""
    
    def __init__(self):
        self.llm_client = OpenAIClient(config)
        
        # Different specialized agents
        self.agents = {
            "technical": TechnicalAgent(self.llm_client),
            "customer_service": ServiceAgent(self.llm_client),
            "sales": SalesAgent(self.llm_client),
            "general": GeneralAgent(self.llm_client)
        }
        
        self.router_agent = RouterAgent(self.llm_client)
    
    async def process(self, input_data: str, context: Dict = None) -> Dict[str, Any]:
        """Route input to appropriate agent."""
        
        # Analyze input to determine routing
        routing_input = IAgentInput(
            message=input_data,
            metadata={"context": context or {}}
        )
        
        route_decision = await self.router_agent.process(routing_input)
        agent_type = route_decision.get("agent_type", "general")
        
        # Route to selected agent
        selected_agent = self.agents.get(agent_type, self.agents["general"])
        result = await selected_agent.process(routing_input)
        
        return {
            "response": result,
            "routed_to": agent_type,
            "confidence": route_decision.get("confidence", 0.5)
        }

# Custom routing agent
class RouterAgent(BaseAgent):
    async def process(self, input: IAgentInput) -> Dict[str, Any]:
        llm_input = ILLMInput(
            system_prompt="Route requests to: technical, customer_service, sales, or general",
            user_message=input.message,
            structure_type=RoutingDecision  # Pydantic model
        )
        
        result = await self.llm_client.chat(llm_input)
        return result.get("llm_response", {"agent_type": "general"})
```

### Pattern 4: State Machine System

Complex state management with transitions:

```python
from enum import Enum

class ConversationState(Enum):
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    PROCESSING = "processing"
    CONFIRMING = "confirming"
    COMPLETED = "completed"

class StateMachineSystem:
    """Conversation system with state management."""
    
    def __init__(self):
        self.llm_client = OpenAIClient(config)
        self.states = {}  # user_id -> state
        self.data = {}    # user_id -> collected data
        
        # State-specific agents
        self.agents = {
            ConversationState.GREETING: GreetingAgent(self.llm_client),
            ConversationState.COLLECTING_INFO: InfoCollectorAgent(self.llm_client),
            ConversationState.PROCESSING: ProcessingAgent(self.llm_client),
            ConversationState.CONFIRMING: ConfirmationAgent(self.llm_client),
            ConversationState.COMPLETED: CompletionAgent(self.llm_client)
        }
    
    async def process(self, user_id: str, message: str) -> Dict[str, Any]:
        """Process message based on current state."""
        
        # Get current state
        current_state = self.states.get(user_id, ConversationState.GREETING)
        user_data = self.data.get(user_id, {})
        
        # Get appropriate agent
        agent = self.agents[current_state]
        
        # Process with context
        input_data = IAgentInput(
            message=message,
            metadata={
                "state": current_state.value,
                "user_data": user_data,
                "user_id": user_id
            }
        )
        
        result = await agent.process(input_data)
        
        # Handle state transitions
        next_state = self._determine_next_state(current_state, result, user_data)
        
        # Update state and data
        self.states[user_id] = next_state
        if "updated_data" in result:
            self.data[user_id] = result["updated_data"]
        
        return {
            "response": result.get("response", ""),
            "current_state": current_state.value,
            "next_state": next_state.value,
            "progress": self._calculate_progress(next_state)
        }
    
    def _determine_next_state(self, current: ConversationState, result: Dict, data: Dict) -> ConversationState:
        """Your state transition logic."""
        if "next_state" in result:
            return ConversationState(result["next_state"])
        
        # Default transitions
        transitions = {
            ConversationState.GREETING: ConversationState.COLLECTING_INFO,
            ConversationState.COLLECTING_INFO: ConversationState.PROCESSING if self._info_complete(data) else ConversationState.COLLECTING_INFO,
            ConversationState.PROCESSING: ConversationState.CONFIRMING,
            ConversationState.CONFIRMING: ConversationState.COMPLETED if result.get("confirmed") else ConversationState.PROCESSING,
            ConversationState.COMPLETED: ConversationState.GREETING
        }
        
        return transitions.get(current, current)
```

## Advanced Patterns

### Pattern: Event-Driven System

```python
class Event:
    def __init__(self, type: str, data: Any, user_id: str = None):
        self.type = type
        self.data = data
        self.user_id = user_id
        self.timestamp = datetime.utcnow()

class EventDrivenSystem:
    def __init__(self):
        self.handlers = {}  # event_type -> List[handler_func]
        self.agents = {}
        self.event_queue = asyncio.Queue()
    
    def register_handler(self, event_type: str, handler):
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    async def emit_event(self, event: Event):
        await self.event_queue.put(event)
    
    async def process_events(self):
        """Event processing loop."""
        while True:
            event = await self.event_queue.get()
            
            handlers = self.handlers.get(event.type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    print(f"Handler failed: {e}")
            
            self.event_queue.task_done()

# Usage
system = EventDrivenSystem()

@system.register_handler("user_message")
async def handle_user_message(event):
    response = await process_with_agent(event.data)
    await system.emit_event(Event("agent_response", response, event.user_id))

@system.register_handler("agent_response") 
async def handle_agent_response(event):
    await send_to_user(event.user_id, event.data)
```

### Pattern: Multi-Agent Collaboration

```python
class CollaborativeSystem:
    """Multiple agents working together on complex tasks."""
    
    def __init__(self):
        self.llm_client = OpenAIClient(config)
        
        # Specialized agents
        self.researcher = ResearchAgent(self.llm_client)
        self.analyst = AnalysisAgent(self.llm_client)
        self.writer = WriterAgent(self.llm_client)
        self.reviewer = ReviewerAgent(self.llm_client)
        
        # Shared memory for collaboration
        self.shared_memory = {}
    
    async def create_report(self, topic: str) -> Dict[str, Any]:
        """Collaborative report creation."""
        
        session_id = f"report_{int(datetime.now().timestamp())}"
        self.shared_memory[session_id] = {"topic": topic}
        
        # Phase 1: Research
        print("🔍 Research phase...")
        research_data = await self.researcher.process(
            IAgentInput(
                message=f"Research topic: {topic}",
                metadata={"session_id": session_id}
            )
        )
        self.shared_memory[session_id]["research"] = research_data
        
        # Phase 2: Analysis
        print("📊 Analysis phase...")
        analysis = await self.analyst.process(
            IAgentInput(
                message="Analyze the research data",
                metadata={
                    "session_id": session_id,
                    "research_data": research_data
                }
            )
        )
        self.shared_memory[session_id]["analysis"] = analysis
        
        # Phase 3: Writing
        print("✍️ Writing phase...")
        draft = await self.writer.process(
            IAgentInput(
                message="Create report based on research and analysis",
                metadata={
                    "session_id": session_id,
                    "research_data": research_data,
                    "analysis": analysis
                }
            )
        )
        self.shared_memory[session_id]["draft"] = draft
        
        # Phase 4: Review and refinement
        print("👁️ Review phase...")
        final_report = await self.reviewer.process(
            IAgentInput(
                message="Review and improve the draft report",
                metadata={
                    "session_id": session_id,
                    "draft": draft,
                    "research_data": research_data,
                    "analysis": analysis
                }
            )
        )
        
        return {
            "report": final_report,
            "metadata": {
                "session_id": session_id,
                "phases_completed": ["research", "analysis", "writing", "review"],
                "collaboration_data": self.shared_memory[session_id]
            }
        }
```

## Workflow Integration

Arshai provides optional workflow utilities, but you control how to use them:

```python
from arshai.workflows import WorkflowRunner, WorkflowConfig

class CustomWorkflowSystem(WorkflowConfig):
    """Your custom workflow configuration."""
    
    def __init__(self, llm_client: ILLM, memory_manager: IMemory):
        # Direct dependency injection
        super().__init__(debug_mode=True)
        self.llm_client = llm_client
        self.memory = memory_manager
        
        # Your agents
        self.agents = {
            "extract": ExtractionAgent(llm_client, "Extract data"),
            "analyze": AnalysisAgent(llm_client, "Analyze data"),
            "report": ReportAgent(llm_client, "Generate report")
        }
    
    def _create_nodes(self):
        """You define the workflow nodes."""
        return {
            "extraction": AgentNode(self.agents["extract"]),
            "analysis": AgentNode(self.agents["analyze"]),
            "reporting": AgentNode(self.agents["report"])
        }
    
    def _define_edges(self):
        """You define the workflow connections."""
        return {
            "extraction": "analysis",
            "analysis": "reporting"
        }
    
    def _route_input(self, input_data):
        """You define how inputs enter the workflow."""
        return "extraction"  # Start at extraction

# Usage
workflow_config = CustomWorkflowSystem(llm_client, memory_manager)
workflow_runner = WorkflowRunner(workflow_config)

result = await workflow_runner.run({
    "message": "Process this document...",
    "user_id": "user123"
})
```

## Memory and State Management

You control all memory and state decisions:

```python
class MemoryManagedSystem:
    """System with sophisticated memory management."""
    
    def __init__(self):
        self.llm_client = OpenAIClient(config)
        
        # Your memory choices
        self.short_term = InMemoryStore()  # Fast access
        self.working_memory = RedisStore() # Distributed
        self.long_term = PostgresStore()   # Persistent
        self.vector_store = MilvusStore()  # Semantic search
        
        self.agents = self._create_agents()
    
    async def process_with_memory(self, user_id: str, message: str) -> Dict[str, Any]:
        """Process with full memory context."""
        
        # Retrieve different types of memory
        recent_context = await self.short_term.get(f"recent:{user_id}")
        session_data = await self.working_memory.get(f"session:{user_id}")
        user_profile = await self.long_term.get(f"profile:{user_id}")
        
        # Semantic similarity search
        similar_conversations = await self.vector_store.search(
            query=message,
            filters={"user_id": user_id},
            limit=3
        )
        
        # Create rich context
        context = {
            "recent": recent_context,
            "session": session_data,
            "profile": user_profile,
            "similar": similar_conversations
        }
        
        # Process with context
        agent_input = IAgentInput(
            message=message,
            metadata={
                "user_id": user_id,
                "context": context
            }
        )
        
        result = await self.agents["main"].process(agent_input)
        
        # Update memory systems
        await self._update_memories(user_id, message, result, context)
        
        return result
    
    async def _update_memories(self, user_id: str, message: str, result: Any, context: Dict):
        """Your memory update strategy."""
        
        # Update short-term (recent conversation)
        recent = context.get("recent", [])
        recent.append({"user": message, "agent": str(result)})
        if len(recent) > 10:  # Keep last 10 exchanges
            recent = recent[-10:]
        await self.short_term.set(f"recent:{user_id}", recent, ttl=3600)
        
        # Update working memory (session data)
        session = context.get("session", {})
        session["last_interaction"] = datetime.utcnow().isoformat()
        session["interaction_count"] = session.get("interaction_count", 0) + 1
        await self.working_memory.set(f"session:{user_id}", session, ttl=86400)
        
        # Update long-term (user profile)
        profile = context.get("profile", {})
        # Your profile update logic here
        await self.long_term.set(f"profile:{user_id}", profile)
        
        # Add to vector store for semantic search
        conversation = {
            "message": message,
            "response": str(result),
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id
        }
        await self.vector_store.add(conversation)
```

## Integration Patterns

### REST API Integration

```python
from fastapi import FastAPI
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None

class APIIntegratedSystem:
    def __init__(self):
        self.app = FastAPI()
        self.agent_system = YourAgentSystem()
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.post("/chat")
        async def chat_endpoint(request: ChatRequest):
            try:
                result = await self.agent_system.process(
                    user_id=request.user_id,
                    message=request.message,
                    session_id=request.session_id
                )
                return {"success": True, "data": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# Run: uvicorn your_system:APIIntegratedSystem().app --host 0.0.0.0 --port 8000
```

### WebSocket Integration

```python
import websockets
import json

class WebSocketSystem:
    def __init__(self):
        self.agent_system = YourAgentSystem()
        self.connections = {}  # connection_id -> websocket
    
    async def handle_connection(self, websocket, path):
        connection_id = str(uuid.uuid4())
        self.connections[connection_id] = websocket
        
        try:
            async for message in websocket:
                data = json.loads(message)
                
                # Process with your system
                result = await self.agent_system.process(
                    user_id=data.get("user_id"),
                    message=data.get("message")
                )
                
                # Send response
                await websocket.send(json.dumps({
                    "type": "response",
                    "data": result
                }))
                
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            del self.connections[connection_id]

# Run WebSocket server
system = WebSocketSystem()
start_server = websockets.serve(system.handle_connection, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
```

## Testing System-Level Components

```python
import pytest
from unittest.mock import AsyncMock, Mock

class TestAgentSystem:
    @pytest.fixture
    def mock_agents(self):
        return {
            "agent1": AsyncMock(),
            "agent2": AsyncMock(),
            "agent3": AsyncMock()
        }
    
    @pytest.fixture
    def system(self, mock_agents):
        return YourSystem(agents=mock_agents)
    
    @pytest.mark.asyncio
    async def test_pipeline_processing(self, system, mock_agents):
        # Configure mocks
        mock_agents["agent1"].process.return_value = "stage1_result"
        mock_agents["agent2"].process.return_value = "stage2_result"
        mock_agents["agent3"].process.return_value = "final_result"
        
        # Test pipeline
        result = await system.process_pipeline("input_data")
        
        # Verify calls
        assert result == "final_result"
        mock_agents["agent1"].process.assert_called_once()
        mock_agents["agent2"].process.assert_called_once()
        mock_agents["agent3"].process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, system, mock_agents):
        # Configure error
        mock_agents["agent2"].process.side_effect = Exception("Test error")
        
        # Test error handling
        with pytest.raises(Exception, match="Test error"):
            await system.process_pipeline("input_data")
        
        # Verify first agent was called, third was not
        mock_agents["agent1"].process.assert_called_once()
        mock_agents["agent3"].process.assert_not_called()
```

## Best Practices for Layer 3

### 1. **Design for Testability**
```python
# Good: Testable system
class System:
    def __init__(self, agents: Dict[str, IAgent], storage: IStorage):
        self.agents = agents  # Injectable
        self.storage = storage  # Injectable

# Bad: Hard to test
class System:
    def __init__(self):
        self.agents = create_all_agents()  # Not injectable
        self.storage = DatabaseConnection()  # Not injectable
```

### 2. **Handle Failures Gracefully**
```python
async def robust_processing(self, input_data):
    results = {}
    
    for name, agent in self.agents.items():
        try:
            results[name] = await agent.process(input_data)
        except Exception as e:
            results[name] = f"Failed: {str(e)}"
            # Continue processing other agents
    
    return self.combine_results(results)
```

### 3. **Monitor and Observe**
```python
class ObservableSystem:
    def __init__(self):
        self.metrics = MetricsCollector()
        self.logger = LoggerFactory.get_logger(__name__)
    
    async def process(self, input_data):
        start_time = time.time()
        
        try:
            result = await self._internal_process(input_data)
            self.metrics.increment("processing.success")
            return result
        except Exception as e:
            self.metrics.increment("processing.error")
            self.logger.error(f"Processing failed: {e}")
            raise
        finally:
            processing_time = time.time() - start_time
            self.metrics.histogram("processing.duration", processing_time)
```

### 4. **Design for Scalability**
```python
class ScalableSystem:
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.agents = self._create_agent_pool()
    
    async def process(self, input_data):
        async with self.semaphore:
            # Limit concurrent processing
            return await self._internal_process(input_data)
```

## Summary

Layer 3 is where your creativity and system design skills shine:

- **🎨 Complete Creative Control** - Design your own patterns
- **🔧 Choose Your Tools** - Integrate any services you need
- **📊 Manage State** - Handle complexity as you see fit
- **🚀 Scale Your Way** - Design for your specific needs
- **🧪 Test Your Design** - Full control enables great testing

## Next Steps

- **[Patterns](../../03-patterns/)** - Learn advanced patterns
- **[Components](../../04-components/)** - Explore available building blocks  
- **[Tutorials](../../05-tutorials/)** - Build complete systems
- **[Deployment](../../09-deployment/)** - Deploy to production

---

*Layer 3 is where you become the architect. The framework gives you the building blocks - you design the masterpiece.* 🏗️