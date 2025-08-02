"""
Function execution utilities for LLM tool calling.

Provides generic orchestration patterns for executing functions
and background tasks in LLM tool calling scenarios.
"""

import asyncio
import logging
from typing import List, Dict, Any, Callable, Tuple, Set

logger = logging.getLogger(__name__)


class FunctionOrchestrator:
    """
    Generic function execution orchestrator for LLM tool calling.
    
    Handles parallel execution of regular functions and background tasks
    while maintaining proper task lifecycle management.
    """
    
    def __init__(self):
        self._background_tasks: Set[asyncio.Task] = set()
    
    async def execute_parallel_functions(
        self, 
        functions: List[Callable], 
        args_list: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Execute multiple functions in parallel.
        
        Args:
            functions: List of callable functions to execute
            args_list: List of argument dictionaries for each function
            
        Returns:
            List of function results in the same order as input
        """
        if len(functions) != len(args_list):
            raise ValueError("Functions and args_list must have the same length")
        
        tasks = []
        for func, args in zip(functions, args_list):
            if asyncio.iscoroutinefunction(func):
                # Async function - create coroutine
                task = func(**args)
            else:
                # Sync function - wrap in async task
                async def sync_wrapper(f=func, a=args):
                    return f(**a)
                task = sync_wrapper()
            tasks.append(task)
        
        logger.info(f"Executing {len(tasks)} functions in parallel")
        results = await asyncio.gather(*tasks)
        logger.info(f"Completed {len(results)} function calls")
        
        return results
    
    async def execute_background_tasks(
        self, 
        tasks: Dict[str, Callable], 
        args_dict: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Execute background tasks in fire-and-forget mode.
        
        Args:
            tasks: Dictionary mapping task names to callable functions
            args_dict: Dictionary mapping task names to their arguments
            
        Returns:
            List of context messages describing initiated tasks
        """
        context_messages = []
        
        for task_name, func in tasks.items():
            args = args_dict.get(task_name, {})
            
            logger.info(f"Executing background task: {task_name}")
            
            # Create background task
            if asyncio.iscoroutinefunction(func):
                task = asyncio.create_task(func(**args))
            else:
                async def sync_background_wrapper(f=func, a=args):
                    return f(**a)
                task = asyncio.create_task(sync_background_wrapper())
            
            # Add to background task set for reference management
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            
            # Create context message
            context_msg = (
                f"Background task '{task_name}' initiated and running independently. "
                "This task will execute in the background without returning results to this conversation."
            )
            context_messages.append(context_msg)
            logger.debug(f"Background task {task_name} started in fire-and-forget mode")
        
        return context_messages
    
    def build_function_context_messages(
        self, 
        function_names: List[str], 
        function_results: List[Any], 
        function_args_list: List[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Build enhanced context messages with function arguments and results.
        
        Args:
            function_names: List of function names that were called
            function_results: List of function results
            function_args_list: Optional list of function arguments
            
        Returns:
            List of formatted context messages
        """
        context_messages = []
        
        for i, (function_name, function_response) in enumerate(zip(function_names, function_results)):
            # Get function arguments if provided
            function_args = {}
            if function_args_list and i < len(function_args_list):
                function_args = function_args_list[i]
            
            # Build enhanced context: Function_name(arg1=value1, arg2=value2) → result
            if function_args:
                args_str = ", ".join([f"{k}={v}" for k, v in function_args.items()])
                context_message = f"Function {function_name}({args_str}) → {function_response}"
            else:
                context_message = f"Function {function_name}() → {function_response}"
            
            context_messages.append(context_message)
            logger.debug(f"Enhanced function context: {context_message}")
        
        return context_messages
    
    def get_completion_message(self, num_functions: int) -> str:
        """
        Generate completion message for function execution.
        
        Args:
            num_functions: Number of functions that were executed
            
        Returns:
            Completion message string to guide the model's next response
        """
        if num_functions > 1:
            return f"All {num_functions} requested calculations have been completed. Please provide a summary of the results."
        elif num_functions == 1:
            return "The requested calculation has been completed. Please Do your next step based on the result."
        return ""
    
    def get_active_background_tasks_count(self) -> int:
        """
        Get the number of currently active background tasks.
        
        Returns:
            Number of active background tasks
        """
        return len(self._background_tasks)
    
    async def process_function_calls_from_response(
        self, 
        function_calls, 
        input_functions: Dict[str, Callable],
        input_background_tasks: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """
        Process function calls from LLM response and return execution results.
        
        This is a generic method that can be used by any LLM client to process
        function calls from model responses. Returns raw results that clients
        can format according to their SDK requirements.
        
        Args:
            function_calls: Function calls from the LLM response
            input_functions: Regular callable functions dictionary
            input_background_tasks: Background tasks dictionary
            
        Returns:
            Dict containing:
            - 'function_results': List of regular function results
            - 'function_names': List of function names that were executed
            - 'function_args': List of arguments used for each function
            - 'background_initiated': List of background task descriptions
        """
        # Prepare tasks for parallel execution
        function_tasks = []
        function_names = []
        function_args_list = []
        background_tasks_to_execute = {}
        background_args_dict = {}
    
        for function_call in function_calls:
            function_name = function_call.name
            function_args = dict(function_call.args) if function_call.args else {}
            
            logger.debug(f"Preparing function: {function_name}")
            
            # Check if it's a background task (fire-and-forget)
            if function_name in input_background_tasks:
                background_tasks_to_execute[function_name] = input_background_tasks[function_name]
                background_args_dict[function_name] = function_args
            # Check if it's a regular tool
            elif function_name in input_functions:
                function_tasks.append(input_functions[function_name])
                function_names.append(function_name)
                function_args_list.append(function_args)
            else:
                raise ValueError(f"Function {function_name} not found in available functions or background tasks")
        
        # Execute background tasks
        background_initiated = []
        if background_tasks_to_execute:
            background_messages = await self.execute_background_tasks(
                background_tasks_to_execute, background_args_dict
            )
            background_initiated = background_messages
        
        # Execute regular functions in parallel
        function_results = []
        if function_tasks:
            logger.info(f"Executing {len(function_tasks)} functions in parallel")
            function_results = await self.execute_parallel_functions(
                function_tasks, function_args_list
            )
            logger.info(f"Completed {len(function_results)} function calls")
        
        # Return structured results for client-specific formatting
        return {
            'function_results': function_results,
            'function_names': function_names,
            'function_args': function_args_list,
            'background_initiated': background_initiated
        }
    
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