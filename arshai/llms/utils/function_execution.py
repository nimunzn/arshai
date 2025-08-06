"""
Function execution utilities for LLM tool calling - Version 2.

Provides structured function orchestration with clear input/output contracts
and resilient error handling for the Arshai agentic framework.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Set, Optional
from dataclasses import field

logger = logging.getLogger(__name__)


@dataclass
class FunctionCall:
    """Represents a single function call with metadata."""
    name: str
    args: Dict[str, Any]
    call_id: Optional[str] = None
    is_background: bool = False


@dataclass
class FunctionExecutionInput:
    """Structured input for function execution orchestrator."""
    function_calls: List[FunctionCall]
    available_functions: Dict[str, Callable]  # Maps function names to actual callables
    available_background_tasks: Dict[str, Callable]  # Maps background task names to callables


@dataclass
class FunctionExecutionResult:
    """Structured result from function execution orchestrator."""
    regular_results: List[Dict[str, Any]]  # [{"name": "func", "args": {...}, "result": 8, "error": None}]
    background_initiated: List[str]        # ["task1 started", "task2 started"]
    failed_functions: List[Dict[str, Any]] # [{"name": "broken_func", "args": {...}, "error": "ValueError: ..."}]


@dataclass
class StreamingExecutionState:
    """
    Tracks progressive execution state during streaming.
    
    This class maintains the state of function executions that happen
    progressively during streaming, preventing duplicate executions and
    tracking active tasks.
    """
    active_function_tasks: List[asyncio.Task] = None
    executed_functions: Set[str] = None
    completed_functions: List[Dict[str, Any]] = None
    background_initiated: List[str] = None
    _execution_counter: int = field(default=0, init=False)
    _start_time: float = field(default_factory=time.time, init=False)
    
    def __post_init__(self):
        """Initialize mutable defaults properly."""
        if self.active_function_tasks is None:
            self.active_function_tasks = []
        if self.executed_functions is None:
            self.executed_functions = set()
        if self.completed_functions is None:
            self.completed_functions = []
        if self.background_initiated is None:
            self.background_initiated = []
    
    def _generate_execution_key(self, function_call: FunctionCall) -> str:
        """Generate a deterministic execution key for function call tracking."""
        if function_call.call_id:
            return f"{function_call.name}_{function_call.call_id}"
        else:
            # Fallback: use timestamp + counter for deterministic key
            self._execution_counter += 1
            timestamp_ms = int((time.time() - self._start_time) * 1000)
            return f"{function_call.name}_{timestamp_ms}_{self._execution_counter}"
    
    def add_function_task(self, task: asyncio.Task, function_call: FunctionCall):
        """Track a new function execution."""
        self.active_function_tasks.append(task)
        # Mark as executed to prevent duplicates using deterministic key
        execution_key = self._generate_execution_key(function_call)
        self.executed_functions.add(execution_key)
    
    def is_already_executed(self, function_call: FunctionCall) -> bool:
        """Check if function was already executed."""
        execution_key = self._generate_execution_key(function_call)
        return execution_key in self.executed_functions
    
    def clear_for_next_turn(self):
        """Clear state for the next streaming turn."""
        self.active_function_tasks.clear()
        # Keep executed_functions to prevent re-execution across turns


class FunctionOrchestrator:
    """
    Simplified function execution orchestrator for LLM tool calling.
    
    Handles parallel execution of regular functions and background tasks
    with resilient error handling and structured input/output.
    
    Supports both batch execution (original) and progressive execution (new)
    for real-time streaming scenarios.
    """
    
    def __init__(self):
        self._background_tasks: Set[asyncio.Task] = set()
        # NEW: Track progressive execution tasks
        self._progressive_tasks: Dict[str, asyncio.Task] = {}
    
    async def execute_functions(self, execution_input: FunctionExecutionInput) -> FunctionExecutionResult:
        """
        Execute function calls based on structured input and return structured results.
        
        Args:
            execution_input: Structured input containing function calls and available functions
            
        Returns:
            FunctionExecutionResult with results, background task status, and any errors
        """
        # Separate regular functions from background tasks
        regular_calls = [call for call in execution_input.function_calls if not call.is_background]
        background_calls = [call for call in execution_input.function_calls if call.is_background]
        
        logger.debug(f"Executing {len(regular_calls)} regular functions and {len(background_calls)} background tasks")
        
        # Initialize result containers
        regular_results = []
        background_initiated = []
        failed_functions = []
        
        # Execute regular functions in parallel with error resilience
        if regular_calls:
            regular_results, regular_failures = await self._execute_function_calls(
                regular_calls, execution_input.available_functions, is_background=False
            )
            failed_functions.extend(regular_failures)
        
        # Execute background tasks (fire-and-forget)
        if background_calls:
            bg_messages, bg_failures = await self._execute_function_calls(
                background_calls, execution_input.available_background_tasks, is_background=True
            )
            background_initiated.extend(bg_messages)
            failed_functions.extend(bg_failures)
        
        logger.debug(f"Execution complete: {len(regular_results)} regular results, {len(background_initiated)} background tasks, {len(failed_functions)} failures")
        
        return FunctionExecutionResult(
            regular_results=regular_results,
            background_initiated=background_initiated,
            failed_functions=failed_functions
        )
    
    async def _execute_function_calls(
        self, 
        function_calls: List[FunctionCall], 
        available_functions: Dict[str, Callable],
        is_background: bool = False
    ) -> tuple[List[Any], List[Dict[str, Any]]]:
        """
        Execute function calls in parallel with error resilience.
        
        Args:
            function_calls: List of function calls to execute
            available_functions: Dict mapping function names to callable functions
            is_background: Whether these are background tasks
        
        Returns:
            Tuple of (results_or_messages, failed_results)
        """
        if not function_calls:
            return [], []
        
        successful_results = []
        failed_results = []
        
        if is_background:
            # Handle background tasks - create fire-and-forget tasks
            for call in function_calls:
                if call.name not in available_functions:
                    # Function not found - add to failures
                    failed_results.append({
                        "name": call.name,
                        "args": call.args,
                        "error": f"Function '{call.name}' not found in available background tasks",
                        "call_id": call.call_id
                    })
                    continue
                
                try:
                    func = available_functions[call.name]
                    success_msg = await self._create_background_task(call, func)
                    successful_results.append(success_msg)
                except Exception as e:
                    failed_results.append({
                        "name": call.name,
                        "args": call.args,
                        "error": f"Failed to create background task: {str(e)}",
                        "call_id": call.call_id
                    })
        else:
            # Handle regular functions - execute in parallel
            tasks = []
            call_metadata = []
            
            # Prepare tasks for parallel execution
            for call in function_calls:
                if call.name not in available_functions:
                    # Function not found - add to failures immediately
                    failed_results.append({
                        "name": call.name,
                        "args": call.args,
                        "error": f"Function '{call.name}' not found in available functions",
                        "call_id": call.call_id
                    })
                    continue
                
                func = available_functions[call.name]
                call_metadata.append(call)
                
                if asyncio.iscoroutinefunction(func):
                    # Async function
                    task = self._safe_execute_async(call.name, func(**call.args), call.args, call.call_id)
                else:
                    # Sync function - wrap in async
                    task = self._safe_execute_sync(call.name, func, call.args, call.call_id)
                
                tasks.append(task)
            
            if tasks:
                # Execute all tasks in parallel
                logger.debug(f"Executing {len(tasks)} regular functions in parallel")
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Separate successful results from failures
                for i, result in enumerate(results):
                    call = call_metadata[i]
                    
                    if isinstance(result, Exception):
                        # Task failed
                        failed_results.append({
                            "name": call.name,
                            "args": call.args,
                            "error": str(result),
                            "call_id": call.call_id
                        })
                        logger.warning(f"Function {call.name} failed: {str(result)}")
                    else:
                        # Task succeeded - return structured results
                        successful_results.append({
                            "name": call.name,
                            "args": call.args,
                            "result": result,
                            "call_id": call.call_id
                        })
                        logger.debug(f"Function {call.name} completed successfully")
        
        return successful_results, failed_results
    
    # ========================================================================
    # PROGRESSIVE EXECUTION METHODS (NEW)
    # ========================================================================
    
    async def execute_function_progressively(
        self,
        function_call: FunctionCall,
        available_functions: Dict[str, Callable],
        available_background_tasks: Dict[str, Callable]
    ) -> asyncio.Task:
        """
        Execute a single function immediately and return the task.
        
        This enables real-time progressive execution during streaming,
        where functions execute as soon as they're complete rather than
        waiting for the entire stream to finish.
        
        Args:
            function_call: Single function call to execute
            available_functions: Dict of regular callable functions
            available_background_tasks: Dict of background task callables
        
        Returns:
            asyncio.Task that can be awaited later for the result
        """
        if function_call.is_background:
            # Background task - fire and forget
            if function_call.name not in available_background_tasks:
                # Create a failed task for missing function
                async def missing_bg_task():
                    raise ValueError(f"Background task '{function_call.name}' not found")
                task = asyncio.create_task(missing_bg_task())
                task.function_call = function_call  # Attach metadata
                return task
            
            func = available_background_tasks[function_call.name]
            task = await self._create_progressive_background_task(function_call, func)
            return task
        else:
            # Regular function - track for later gathering
            if function_call.name not in available_functions:
                # Create a failed task for missing function
                async def missing_func():
                    raise ValueError(f"Function '{function_call.name}' not found")
                task = asyncio.create_task(missing_func())
                task.function_call = function_call  # Attach metadata
                return task
            
            func = available_functions[function_call.name]
            task = await self._create_progressive_regular_task(function_call, func)
            
            # Track the task for management
            if function_call.call_id:
                self._progressive_tasks[function_call.call_id] = task
            
            return task
    
    async def _create_progressive_regular_task(self, function_call: FunctionCall, func: Callable) -> asyncio.Task:
        """Create a task for progressive execution of a regular function."""
        async def execute_with_metadata():
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(**function_call.args)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: func(**function_call.args))
                
                # Return structured result
                return {
                    "name": function_call.name,
                    "args": function_call.args,
                    "result": result,
                    "call_id": function_call.call_id,
                    "error": None
                }
            except Exception as e:
                # Return structured error
                return {
                    "name": function_call.name,
                    "args": function_call.args,
                    "result": None,
                    "call_id": function_call.call_id,
                    "error": str(e)
                }
        
        task = asyncio.create_task(execute_with_metadata())
        task.function_call = function_call  # Attach metadata for tracking
        return task
    
    async def _create_progressive_background_task(self, function_call: FunctionCall, func: Callable) -> asyncio.Task:
        """Create a task for progressive execution of a background task."""
        async def execute_background():
            try:
                logger.debug(f"Starting progressive background task: {function_call.name}")
                
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    await func(**function_call.args)
                else:
                    # Run sync function
                    func(**function_call.args)
                
                logger.debug(f"Background task {function_call.name} completed successfully")
                return f"Background task '{function_call.name}' initiated with args {function_call.args}"
            except Exception as e:
                logger.warning(f"Background task '{function_call.name}' failed: {str(e)}")
                # Background tasks don't return errors to the caller
                return f"Background task '{function_call.name}' failed: {str(e)}"
        
        task = asyncio.create_task(execute_background())
        
        # Add to background task set for reference management
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        task.function_call = function_call  # Attach metadata
        return task
    
    async def gather_progressive_results(
        self,
        function_tasks: List[asyncio.Task]
    ) -> FunctionExecutionResult:
        """
        Gather results from progressively executed functions.
        
        This method waits for all progressive tasks to complete and
        consolidates their results into a structured format.
        
        Args:
            function_tasks: List of tasks from progressive execution
        
        Returns:
            FunctionExecutionResult with consolidated results
        """
        if not function_tasks:
            return FunctionExecutionResult(
                regular_results=[],
                background_initiated=[],
                failed_functions=[]
            )
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*function_tasks, return_exceptions=True)
        
        regular_results = []
        failed_functions = []
        background_initiated = []
        
        for task, result in zip(function_tasks, results):
            # Extract function metadata from task
            function_call = getattr(task, 'function_call', None)
            if not function_call:
                logger.warning("Task missing function_call metadata")
                continue
            
            if isinstance(result, Exception):
                # Task raised an exception
                failed_functions.append({
                    "name": function_call.name,
                    "args": function_call.args,
                    "error": str(result),
                    "call_id": function_call.call_id
                })
            elif function_call.is_background:
                # Background task message
                if isinstance(result, str):
                    background_initiated.append(result)
                else:
                    background_initiated.append(
                        f"Background task '{function_call.name}' completed"
                    )
            elif isinstance(result, dict) and "error" in result and result["error"]:
                # Regular function returned error
                failed_functions.append({
                    "name": result.get("name", function_call.name),
                    "args": result.get("args", function_call.args),
                    "error": result["error"],
                    "call_id": result.get("call_id", function_call.call_id)
                })
            elif isinstance(result, dict):
                # Regular function success
                regular_results.append(result)
            else:
                # Unexpected result format
                regular_results.append({
                    "name": function_call.name,
                    "args": function_call.args,
                    "result": result,
                    "call_id": function_call.call_id,
                    "error": None
                })
        
        logger.debug(
            f"Progressive execution complete: {len(regular_results)} regular results, "
            f"{len(background_initiated)} background tasks, {len(failed_functions)} failures"
        )
        
        # Clear tracked progressive tasks
        self._progressive_tasks.clear()
        
        return FunctionExecutionResult(
            regular_results=regular_results,
            background_initiated=background_initiated,
            failed_functions=failed_functions
        )
    
    async def _create_background_task(self, call: FunctionCall, func: Callable) -> str:
        """
        Create a background task for fire-and-forget execution.
        
        Args:
            call: Function call information
            func: Callable function to execute
        
        Returns:
            Success message for the initiated background task
        """
        logger.debug(f"Starting background task: {call.name}")
        
        # Create background task
        if asyncio.iscoroutinefunction(func):
            task = asyncio.create_task(func(**call.args))
        else:
            # Wrap sync function in async task
            async def sync_background_wrapper():
                return func(**call.args)
            task = asyncio.create_task(sync_background_wrapper())
        
        # Add to background task set for reference management
        self._background_tasks.add(task)
        
        # Add error handling callback for background tasks
        def handle_background_completion(completed_task):
            self._background_tasks.discard(completed_task)
            if completed_task.exception() is not None:
                logger.warning(f"Background task '{call.name}' failed: {completed_task.exception()}")
        
        task.add_done_callback(handle_background_completion)
        
        # Create success message
        success_msg = f"Background task '{call.name}' initiated with args {call.args}"
        logger.debug(f"Background task {call.name} started successfully")
        
        return success_msg
    
    async def _safe_execute_async(self, func_name: str, coro, args: Dict[str, Any], call_id: Optional[str] = None):
        """Safely execute an async function with error handling."""
        try:
            result = await coro
            return result
        except Exception as e:
            logger.warning(f"Async function {func_name} (call_id: {call_id}) failed: {str(e)}")
            raise e
    
    async def _safe_execute_sync(self, func_name: str, func: Callable, args: Dict[str, Any], call_id: Optional[str] = None):
        """Safely execute a sync function with error handling."""
        try:
            # Run sync function in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: func(**args))
            return result
        except Exception as e:
            logger.warning(f"Sync function {func_name} (call_id: {call_id}) failed: {str(e)}")
            raise e
    
    def get_active_background_tasks_count(self) -> int:
        """
        Get the number of currently active background tasks.
        
        Returns:
            Number of active background tasks
        """
        return len(self._background_tasks)
    
    async def wait_for_background_tasks(self, timeout: float = None) -> None:
        """
        Wait for all background tasks to complete (useful for testing).
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        if self._background_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._background_tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Background tasks did not complete within {timeout} seconds")
            except Exception as e:
                logger.warning(f"Error waiting for background tasks: {str(e)}")