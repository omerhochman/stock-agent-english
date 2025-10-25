import functools
import io
import sys
import logging
from contextvars import ContextVar
from typing import Any, Callable, Optional
from datetime import datetime, timezone

UTC = timezone.utc


# --- Context Variables ---
# These variables hold state specific to the current execution context (e.g., a single agent run within a workflow).

# Holds the name of the agent currently being executed. Set by the decorator.
current_agent_name_context: ContextVar[Optional[str]] = ContextVar(
    "current_agent_name_context", default=None
)

# Holds the unique ID for the entire workflow run. Set in main.py and passed via state.
current_run_id_context: ContextVar[Optional[str]] = ContextVar(
    "current_run_id_context", default=None
)

# Simple in-memory storage class for log information
class SimpleLogStorage:
    def __init__(self):
        self.llm_logs = []
        self.agent_logs = []
    
    def add_log(self, log_entry):
        """Add LLM interaction log"""
        self.llm_logs.append(log_entry)
    
    def add_agent_log(self, log_entry):
        """Add Agent execution log"""
        self.agent_logs.append(log_entry)
    
    def get_logs(self, agent_name=None, run_id=None):
        """Get LLM interaction logs matching criteria"""
        if not agent_name and not run_id:
            return self.llm_logs
        
        filtered_logs = []
        for log in self.llm_logs:
            if (agent_name is None or log.get("agent_name") == agent_name) and \
               (run_id is None or log.get("run_id") == run_id):
                filtered_logs.append(log)
        
        return filtered_logs
    
    def get_agent_logs(self, agent_name=None, run_id=None):
        """Get Agent execution logs matching criteria"""
        if not agent_name and not run_id:
            return self.agent_logs
        
        filtered_logs = []
        for log in self.agent_logs:
            if (agent_name is None or log.get("agent_name") == agent_name) and \
               (run_id is None or log.get("run_id") == run_id):
                filtered_logs.append(log)
        
        return filtered_logs

# Global log storage instance
_log_storage = SimpleLogStorage()

# --- Output Capture Utility ---
class OutputCapture:
    """Utility class for capturing standard output and logs"""

    def __init__(self):
        self.outputs = []
        self.stdout_buffer = io.StringIO()
        self.old_stdout = None
        self.log_handler = None
        self.old_log_level = None

    def __enter__(self):
        # Capture standard output
        self.old_stdout = sys.stdout
        sys.stdout = self.stdout_buffer

        # Capture logs
        self.log_handler = logging.StreamHandler(io.StringIO())
        self.log_handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        self.old_log_level = root_logger.level
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self.log_handler)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore standard output and capture content
        sys.stdout = self.old_stdout
        stdout_content = self.stdout_buffer.getvalue()
        if stdout_content.strip():
            self.outputs.append(stdout_content)

        # Restore logs and capture content
        log_content = self.log_handler.stream.getvalue()
        if log_content.strip():
            self.outputs.append(log_content)

        # Clean up log handler
        root_logger = logging.getLogger()
        root_logger.removeHandler(self.log_handler)
        root_logger.setLevel(self.old_log_level)


# --- Wrapper for LLM Calls ---
def wrap_llm_call(original_llm_func: Callable) -> Callable:
    """Wraps an LLM call function (like get_chat_completion) to log interactions.

    Reads context variables set by the agent decorator to get agent name,
    run ID, and the storage instance.

    Args:
        original_llm_func: The original function that makes the LLM call.

    Returns:
        A wrapped function that logs the interaction before returning the original result.
    """

    @functools.wraps(original_llm_func)
    def wrapper(*args, **kwargs) -> Any:
        # Retrieve context information
        agent_name = current_agent_name_context.get()
        run_id = current_run_id_context.get()

        # Proceed with the original call even if context is missing, but don't log
        if not agent_name:
            # Maybe log a warning here if desired
            return original_llm_func(*args, **kwargs)

        # Assume the first argument is usually the list of messages or prompt
        # This might need adjustment if the wrapped function signature varies
        request_data = args[0] if args else kwargs.get(
            'messages', kwargs)  # Adapt based on common usage

        # Execute the original LLM call
        response_data = original_llm_func(*args, **kwargs)

        # Create and store the log entry
        log_entry = {
            "agent_name": agent_name,
            "run_id": run_id,  # run_id can be None if not set
            "request_data": request_data,  # Consider serializing complex objects if needed
            "response_data": response_data,  # Consider serializing complex objects if needed
            "timestamp": datetime.now(UTC).isoformat()  # Explicit timestamp
        }
        _log_storage.add_log(log_entry)

        return response_data

    return wrapper


# --- Decorator for Agent Functions ---
def log_agent_execution(agent_name: str):
    """Decorator for agent functions to set logging context variables.

    Retrieves the run_id from the agent state's metadata.

    Args:
        agent_name: The name of the agent being decorated.
    """

    def decorator(agent_func: Callable[[dict], dict]):
        @functools.wraps(agent_func)
        def wrapper(state: dict) -> dict:
            # Retrieve run_id from state metadata (set in main.py)
            run_id = state.get("metadata", {}).get("run_id")

            # Set context variables
            agent_token = current_agent_name_context.set(agent_name)
            run_id_token = current_run_id_context.set(run_id)

            # Capture start time and input state
            timestamp_start = datetime.now(UTC)
            
            # Serialize input state
            serialized_input = state.copy() if isinstance(state, dict) else state

            # Prepare output capture
            output_capture = OutputCapture()
            result_state = None
            error = None

            try:
                # Use output capture
                with output_capture:
                    # Execute original Agent function
                    result_state = agent_func(state)

                # Successful execution, record log
                timestamp_end = datetime.now(UTC)
                terminal_outputs = output_capture.outputs

                # Serialize output state
                serialized_output = result_state.copy() if isinstance(result_state, dict) else result_state

                # Extract reasoning details (if any)
                reasoning_details = None
                if result_state and isinstance(result_state, dict):
                    if result_state.get("metadata", {}).get("show_reasoning", False):
                        if "agent_reasoning" in result_state.get("metadata", {}):
                            reasoning_details = result_state["metadata"]["agent_reasoning"]

                # Create log entry
                log_entry = {
                    "agent_name": agent_name,
                    "run_id": run_id,
                    "timestamp_start": timestamp_start.isoformat(),
                    "timestamp_end": timestamp_end.isoformat(),
                    "input_state": serialized_input,
                    "output_state": serialized_output,
                    "reasoning_details": reasoning_details,
                    "terminal_outputs": terminal_outputs
                }

                # Store log
                _log_storage.add_agent_log(log_entry)
            except Exception as e:
                # Record error
                error = str(e)
                # If error occurs, record error log
                timestamp_end = datetime.now(UTC)
                log_entry = {
                    "agent_name": agent_name,
                    "run_id": run_id,
                    "timestamp_start": timestamp_start.isoformat(),
                    "timestamp_end": timestamp_end.isoformat(),
                    "input_state": serialized_input,
                    "output_state": {"error": error},
                    "reasoning_details": None,
                    "terminal_outputs": output_capture.outputs
                }
                _log_storage.add_agent_log(log_entry)
                
                # Re-raise exception for upper level handling
                raise
            finally:
                # Clean up context variables
                current_agent_name_context.reset(agent_token)
                current_run_id_context.reset(run_id_token)

            return result_state
        return wrapper
    return decorator


# Helper to set the global storage instance (called from main.py)
def set_global_log_storage(storage):
    """Set global log storage instance if custom storage is provided"""
    global _log_storage
    if storage:
        _log_storage = storage


# Function to get log storage
def get_log_storage():
    """Get current log storage instance"""
    return _log_storage