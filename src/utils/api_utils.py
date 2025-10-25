"""
API utility module - provides shared API functionality components for Agents

This module defines API functionality components, providing unified API exposure methods for various Agents.
"""

import functools
import inspect
import io
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, TypeVar

UTC = timezone.utc

import uvicorn
from fastapi import APIRouter, FastAPI

# Type definitions
T = TypeVar("T")

# Create FastAPI application instance
app = FastAPI(
    title="A_Share_Investment_Agent API",
    description="API service for A_Share_Investment_Agent",
    version="0.1.0",
)

# Create routers
agents_router = APIRouter(tags=["Agents"])
runs_router = APIRouter(tags=["Runs"])
workflow_router = APIRouter(tags=["Workflow"])

# Include routers
app.include_router(agents_router)
app.include_router(runs_router)
app.include_router(workflow_router)

# Add a global dictionary to track LLM calls for each agent
_agent_llm_calls = {}


# Create global state manager
class APIState:
    def __init__(self):
        self.current_run_id = None
        self.current_agent_name = None
        self.agents = {}
        self.runs = {}
        self.agent_data = {}

    def register_agent(self, agent_name: str, description: str = ""):
        """Register Agent to state manager"""
        self.agents[agent_name] = {
            "name": agent_name,
            "description": description,
            "state": "idle",
        }
        self.agent_data[agent_name] = {}

    def update_agent_state(self, agent_name: str, state: str):
        """Update Agent state"""
        if agent_name in self.agents:
            self.agents[agent_name]["state"] = state
            self.current_agent_name = agent_name

    def update_agent_data(self, agent_name: str, key: str, value: Any):
        """Update Agent data"""
        if agent_name not in self.agent_data:
            self.agent_data[agent_name] = {}
        self.agent_data[agent_name][key] = value

    def register_run(self, run_id: str, run_info: Dict[str, Any]):
        """Register run ID and information"""
        self.runs[run_id] = run_info
        self.current_run_id = run_id


# Instantiate global state manager
api_state = APIState()

# Define logger uniformly here
logger = logging.getLogger("api_utils")


# Utility functions
def safe_parse_json(json_str):
    """Safely parse JSON string, return None on error"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None


def format_llm_request(request_data):
    """Format LLM request data for logging"""
    try:
        # If it's a dictionary, return directly
        if isinstance(request_data, dict):
            return request_data
        # If it's a list, try to process message format
        if isinstance(request_data, list):
            # Simple check if it's message format
            if all(isinstance(item, dict) and "role" in item for item in request_data):
                return {"messages": request_data}
            return {"data": request_data}
        # Other types, try to convert to string
        return {"data": str(request_data)}
    except Exception as e:
        logger.warning(f"Error formatting LLM request: {str(e)}")
        return {"data": "Unable to format request"}


def format_llm_response(response_data):
    """Format LLM response data for logging"""
    try:
        # Handle common response formats
        if isinstance(response_data, dict):
            # Check if it has typical OpenAI response structure
            if "choices" in response_data:
                return response_data
            return response_data
        # Other types, try to convert to string
        return {"content": str(response_data)}
    except Exception as e:
        logger.warning(f"Error formatting LLM response: {str(e)}")
        return {"content": "Unable to format response"}


# -----------------------------------------------------------------------------
# Decorators and utility functions
# -----------------------------------------------------------------------------
def log_llm_interaction(state):
    """Decorator function for logging LLM interactions

    This function can be used in two ways:
    1. As a decorator factory: log_llm_interaction(state)(llm_func)
    2. As a direct call function: for existing log_llm_interaction compatibility mode
    """
    # Check if it's direct function call mode (backward compatibility)
    if isinstance(state, str) and len(state) > 0:
        # Compatible with original direct call method
        agent_name = state  # First parameter is agent_name

        def direct_logger(request_data, response_data):
            # Save formatted request and response
            formatted_request = format_llm_request(request_data)
            formatted_response = format_llm_response(response_data)

            timestamp = datetime.now(UTC)

            # Get current run ID
            run_id = api_state.current_run_id

            api_state.update_agent_data(agent_name, "llm_request", formatted_request)
            api_state.update_agent_data(agent_name, "llm_response", formatted_response)

            # Record interaction timestamp
            api_state.update_agent_data(
                agent_name, "llm_timestamp", timestamp.isoformat()
            )

            return response_data

        return direct_logger

    # Decorator factory mode
    def decorator(llm_func):
        @functools.wraps(llm_func)
        def wrapper(*args, **kwargs):
            # Get function call information for better request logging
            caller_frame = inspect.currentframe().f_back
            caller_info = {
                "function": llm_func.__name__,
                "file": caller_frame.f_code.co_filename,
                "line": caller_frame.f_lineno,
            }

            # Execute original function to get result
            result = llm_func(*args, **kwargs)

            # Extract agent_name and run_id from state
            agent_name = None
            run_id = None

            # Try to extract from state parameter
            if isinstance(state, dict):
                agent_name = state.get("metadata", {}).get("current_agent_name")
                run_id = state.get("metadata", {}).get("run_id")

            # Try to get from context variables
            if not agent_name:
                try:
                    from src.utils.llm_interaction_logger import (
                        current_agent_name_context,
                        current_run_id_context,
                    )

                    agent_name = current_agent_name_context.get()
                    run_id = current_run_id_context.get()
                except (ImportError, AttributeError):
                    pass

            # If still not found, try to get current running agent from api_state
            if not agent_name and hasattr(api_state, "current_agent_name"):
                agent_name = api_state.current_agent_name
                run_id = api_state.current_run_id

            if agent_name:
                timestamp = datetime.now(UTC)

                # Extract messages parameter
                messages = None
                if "messages" in kwargs:
                    messages = kwargs["messages"]
                elif args and len(args) > 0:
                    messages = args[0]

                # Extract other parameters
                model = kwargs.get("model")
                client_type = kwargs.get("client_type", "auto")

                # Prepare formatted request data
                formatted_request = {
                    "caller": caller_info,
                    "messages": messages,
                    "model": model,
                    "client_type": client_type,
                    "arguments": format_llm_request(args),
                    "kwargs": format_llm_request(kwargs) if kwargs else {},
                }

                # Prepare formatted response data
                formatted_response = format_llm_response(result)

                # Record to API state
                api_state.update_agent_data(
                    agent_name, "llm_request", formatted_request
                )
                api_state.update_agent_data(
                    agent_name, "llm_response", formatted_response
                )
                api_state.update_agent_data(
                    agent_name, "llm_timestamp", timestamp.isoformat()
                )

            return result

        return wrapper

    return decorator


def agent_endpoint(agent_name: str, description: str = ""):
    """
    Decorator for creating API endpoints for Agents

    Usage:
    @agent_endpoint("sentiment")
    def sentiment_agent(state: AgentState) -> AgentState:
        ...
    """

    def decorator(agent_func):
        # Register Agent
        api_state.register_agent(agent_name, description)

        # Initialize LLM call tracking for this agent
        _agent_llm_calls[agent_name] = False

        @functools.wraps(agent_func)
        def wrapper(state):
            # Update Agent state to running
            api_state.update_agent_state(agent_name, "running")

            # Add current agent name to state metadata
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"]["current_agent_name"] = agent_name

            # Ensure run_id is in metadata, crucial for logging
            run_id = state.get("metadata", {}).get("run_id")
            # Record input state
            timestamp_start = datetime.now(UTC)

            # Serialize state function
            def serialize_agent_state(state):
                """Serialize Agent state"""
                try:
                    # Simply use shallow copy to prevent modifying original state
                    state_copy = state.copy() if isinstance(state, dict) else state
                    return state_copy
                except Exception as e:
                    logger.warning(f"Failed to serialize Agent state: {str(e)}")
                    return {"error": "Unable to serialize state"}

            serialized_input = serialize_agent_state(state)
            api_state.update_agent_data(agent_name, "input_state", serialized_input)

            result = None
            error = None
            terminal_outputs = []  # Capture terminal output

            # Capture stdout/stderr and logs during agent execution
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            log_stream = io.StringIO()
            log_handler = logging.StreamHandler(log_stream)
            log_handler.setLevel(logging.INFO)
            root_logger = logging.getLogger()
            root_logger.addHandler(log_handler)

            redirect_stdout = io.StringIO()
            redirect_stderr = io.StringIO()
            sys.stdout = redirect_stdout
            sys.stderr = redirect_stderr

            try:
                # --- Execute Agent core logic ---
                # Directly call original agent_func
                result = agent_func(state)
                # --------------------------

                timestamp_end = datetime.now(UTC)

                # Restore standard output/error
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                root_logger.removeHandler(log_handler)

                # Get captured output
                stdout_content = redirect_stdout.getvalue()
                stderr_content = redirect_stderr.getvalue()
                log_content = log_stream.getvalue()
                if stdout_content:
                    terminal_outputs.append(stdout_content)
                if stderr_content:
                    terminal_outputs.append(stderr_content)
                if log_content:
                    terminal_outputs.append(log_content)

                # Serialize output state
                serialized_output = serialize_agent_state(result)
                api_state.update_agent_data(
                    agent_name, "output_state", serialized_output
                )

                # Extract reasoning details from state (if any)
                reasoning_details = None
                if result.get("metadata", {}).get("show_reasoning", False):
                    if "agent_reasoning" in result.get("metadata", {}):
                        reasoning_details = result["metadata"]["agent_reasoning"]
                        api_state.update_agent_data(
                            agent_name, "reasoning", reasoning_details
                        )

                # Update Agent state to completed
                api_state.update_agent_state(agent_name, "completed")

                return result
            except Exception as e:
                # Record end time even on error
                timestamp_end = datetime.now(UTC)
                error = str(e)
                # Restore standard output/error
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                root_logger.removeHandler(log_handler)
                # Get captured output
                stdout_content = redirect_stdout.getvalue()
                stderr_content = redirect_stderr.getvalue()
                log_content = log_stream.getvalue()
                if stdout_content:
                    terminal_outputs.append(stdout_content)
                if stderr_content:
                    terminal_outputs.append(stderr_content)
                if log_content:
                    terminal_outputs.append(log_content)

                # Update Agent state to error
                api_state.update_agent_state(agent_name, "error")
                # Record error information
                api_state.update_agent_data(agent_name, "error", error)

                # Re-raise exception
                raise

        return wrapper

    return decorator


# Function to start API server
def start_api_server(host="0.0.0.0", port=8000, stop_event=None):
    """Start API server in separate thread"""
    if stop_event:
        # Use configuration that supports graceful shutdown
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_config=None,
            # Enable ctrl+c handling
            use_colors=True,
        )
        server = uvicorn.Server(config)

        # Run server and listen for stop_event in separate thread
        def check_stop_event():
            # Check stop_event in background
            while not stop_event.is_set():
                time.sleep(0.5)
            # When stop_event is set, request server to exit
            logger.info("Received stop signal, shutting down API server...")
            server.should_exit = True

        # Start stop_event monitoring thread
        stop_monitor = threading.Thread(target=check_stop_event, daemon=True)
        stop_monitor.start()

        # Run server (blocking call, but responds to should_exit flag)
        try:
            server.run()
        except KeyboardInterrupt:
            # If KeyboardInterrupt is still received, ensure our stop_event is also set
            stop_event.set()
        logger.info("API server has been shut down")
    else:
        # Default startup method, no external stop control but still responds to Ctrl+C
        uvicorn.run(app, host=host, port=port, log_config=None)


# Add basic API routes
@app.get("/", tags=["Root"])
def read_root():
    """API root path"""
    return {"message": "A_Share_Investment_Agent API", "version": "0.1.0"}


@app.get("/agents/", tags=["Agents"])
def list_agents():
    """List all registered Agents"""
    return {"agents": list(api_state.agents.values())}


@app.get("/status/", tags=["Status"])
def get_status():
    """Get system status"""
    return {
        "current_run_id": api_state.current_run_id,
        "current_agent": api_state.current_agent_name,
        "agents_count": len(api_state.agents),
        "runs_count": len(api_state.runs),
    }
