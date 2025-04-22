"""
API工具模块 - 提供Agent共享的API功能组件

此模块定义了API功能组件，为各个Agent提供统一的API暴露方式。
"""

import json
import logging
import functools
import threading
import time
import inspect
import sys
import io
from typing import Dict, List, Any, Optional, Callable, TypeVar
from datetime import datetime, timezone

UTC = timezone.utc

import uvicorn
from fastapi import FastAPI, APIRouter

# 类型定义
T = TypeVar('T')

# 创建FastAPI应用实例
app = FastAPI(
    title="A_Share_Investment_Agent API",
    description="A_Share_Investment_Agent的API服务",
    version="0.1.0"
)

# 创建路由器
agents_router = APIRouter(tags=["Agents"])
runs_router = APIRouter(tags=["Runs"])
workflow_router = APIRouter(tags=["Workflow"])

# 应用路由器
app.include_router(agents_router)
app.include_router(runs_router)
app.include_router(workflow_router)

# 增加一个全局字典用于跟踪每个agent的LLM调用
_agent_llm_calls = {}

# 创建全局状态管理器
class APIState:
    def __init__(self):
        self.current_run_id = None
        self.current_agent_name = None
        self.agents = {}
        self.runs = {}
        self.agent_data = {}
    
    def register_agent(self, agent_name: str, description: str = ""):
        """注册Agent到状态管理器"""
        self.agents[agent_name] = {
            "name": agent_name,
            "description": description,
            "state": "idle"
        }
        self.agent_data[agent_name] = {}
    
    def update_agent_state(self, agent_name: str, state: str):
        """更新Agent的状态"""
        if agent_name in self.agents:
            self.agents[agent_name]["state"] = state
            self.current_agent_name = agent_name
    
    def update_agent_data(self, agent_name: str, key: str, value: Any):
        """更新Agent的数据"""
        if agent_name not in self.agent_data:
            self.agent_data[agent_name] = {}
        self.agent_data[agent_name][key] = value
    
    def register_run(self, run_id: str, run_info: Dict[str, Any]):
        """注册运行ID和信息"""
        self.runs[run_id] = run_info
        self.current_run_id = run_id

# 实例化全局状态管理器
api_state = APIState()

# 统一在此处定义 logger
logger = logging.getLogger("api_utils")

# 工具函数
def safe_parse_json(json_str):
    """安全解析JSON字符串，出错时返回None"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None

def format_llm_request(request_data):
    """格式化LLM请求数据以便记录"""
    try:
        # 如果是字典，直接返回
        if isinstance(request_data, dict):
            return request_data
        # 如果是列表，尝试处理消息格式
        if isinstance(request_data, list):
            # 简单检查是否为消息格式
            if all(isinstance(item, dict) and "role" in item for item in request_data):
                return {"messages": request_data}
            return {"data": request_data}
        # 其他类型，尝试转换为字符串
        return {"data": str(request_data)}
    except Exception as e:
        logger.warning(f"格式化LLM请求时出错: {str(e)}")
        return {"data": "无法格式化的请求"}

def format_llm_response(response_data):
    """格式化LLM响应数据以便记录"""
    try:
        # 处理常见的响应格式
        if isinstance(response_data, dict):
            # 检查是否有典型的OpenAI响应结构
            if "choices" in response_data:
                return response_data
            return response_data
        # 其他类型，尝试转换为字符串
        return {"content": str(response_data)}
    except Exception as e:
        logger.warning(f"格式化LLM响应时出错: {str(e)}")
        return {"content": "无法格式化的响应"}

# -----------------------------------------------------------------------------
# 装饰器和工具函数
# -----------------------------------------------------------------------------
def log_llm_interaction(state):
    """记录LLM交互的装饰器函数

    这个函数可以以两种方式使用：
    1. 作为装饰器工厂：log_llm_interaction(state)(llm_func)
    2. 作为直接调用函数：用于已有的log_llm_interaction兼容模式
    """
    # 检查是否是直接函数调用模式（向后兼容）
    if isinstance(state, str) and len(state) > 0:
        # 兼容原有直接调用方式
        agent_name = state  # 第一个参数是agent_name

        def direct_logger(request_data, response_data):
            # 保存格式化的请求和响应
            formatted_request = format_llm_request(request_data)
            formatted_response = format_llm_response(response_data)

            timestamp = datetime.now(UTC)

            # 获取当前运行ID
            run_id = api_state.current_run_id

            api_state.update_agent_data(
                agent_name, "llm_request", formatted_request)
            api_state.update_agent_data(
                agent_name, "llm_response", formatted_response)

            # 记录交互的时间戳
            api_state.update_agent_data(
                agent_name, "llm_timestamp", timestamp.isoformat())

            return response_data

        return direct_logger

    # 装饰器工厂模式
    def decorator(llm_func):
        @functools.wraps(llm_func)
        def wrapper(*args, **kwargs):
            # 获取函数调用信息，以便更好地记录请求
            caller_frame = inspect.currentframe().f_back
            caller_info = {
                "function": llm_func.__name__,
                "file": caller_frame.f_code.co_filename,
                "line": caller_frame.f_lineno
            }

            # 执行原始函数获取结果
            result = llm_func(*args, **kwargs)

            # 从state中提取agent_name和run_id
            agent_name = None
            run_id = None

            # 尝试从state参数中提取
            if isinstance(state, dict):
                agent_name = state.get("metadata", {}).get(
                    "current_agent_name")
                run_id = state.get("metadata", {}).get("run_id")

            # 尝试从上下文变量中获取
            if not agent_name:
                try:
                    from src.utils.llm_interaction_logger import current_agent_name_context, current_run_id_context
                    agent_name = current_agent_name_context.get()
                    run_id = current_run_id_context.get()
                except (ImportError, AttributeError):
                    pass

            # 如果仍然没有，尝试从api_state中获取当前运行的agent
            if not agent_name and hasattr(api_state, "current_agent_name"):
                agent_name = api_state.current_agent_name
                run_id = api_state.current_run_id

            if agent_name:
                timestamp = datetime.now(UTC)

                # 提取messages参数
                messages = None
                if "messages" in kwargs:
                    messages = kwargs["messages"]
                elif args and len(args) > 0:
                    messages = args[0]

                # 提取其他参数
                model = kwargs.get("model")
                client_type = kwargs.get("client_type", "auto")

                # 准备格式化的请求数据
                formatted_request = {
                    "caller": caller_info,
                    "messages": messages,
                    "model": model,
                    "client_type": client_type,
                    "arguments": format_llm_request(args),
                    "kwargs": format_llm_request(kwargs) if kwargs else {}
                }

                # 准备格式化的响应数据
                formatted_response = format_llm_response(result)

                # 记录到API状态
                api_state.update_agent_data(
                    agent_name, "llm_request", formatted_request)
                api_state.update_agent_data(
                    agent_name, "llm_response", formatted_response)
                api_state.update_agent_data(
                    agent_name, "llm_timestamp", timestamp.isoformat())

            return result
        return wrapper
    return decorator

def agent_endpoint(agent_name: str, description: str = ""):
    """
    为Agent创建API端点的装饰器

    用法:
    @agent_endpoint("sentiment")
    def sentiment_agent(state: AgentState) -> AgentState:
        ...
    """
    def decorator(agent_func):
        # 注册Agent
        api_state.register_agent(agent_name, description)

        # 初始化此agent的LLM调用跟踪
        _agent_llm_calls[agent_name] = False

        @functools.wraps(agent_func)
        def wrapper(state):
            # 更新Agent状态为运行中
            api_state.update_agent_state(agent_name, "running")

            # 添加当前agent名称到状态元数据
            if "metadata" not in state:
                state["metadata"] = {}
            state["metadata"]["current_agent_name"] = agent_name

            # 确保run_id在元数据中，这对日志记录至关重要
            run_id = state.get("metadata", {}).get("run_id")
            # 记录输入状态
            timestamp_start = datetime.now(UTC)
            
            # 序列化状态函数
            def serialize_agent_state(state):
                """序列化Agent状态"""
                try:
                    # 简单地使用浅拷贝防止修改原始状态
                    state_copy = state.copy() if isinstance(state, dict) else state
                    return state_copy
                except Exception as e:
                    logger.warning(f"序列化Agent状态失败: {str(e)}")
                    return {"error": "无法序列化的状态"}
                
            serialized_input = serialize_agent_state(state)
            api_state.update_agent_data(
                agent_name, "input_state", serialized_input)

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
                # --- 执行Agent核心逻辑 ---
                # 直接调用原始 agent_func
                result = agent_func(state)
                # --------------------------

                timestamp_end = datetime.now(UTC)

                # 恢复标准输出/错误
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                root_logger.removeHandler(log_handler)

                # 获取捕获的输出
                stdout_content = redirect_stdout.getvalue()
                stderr_content = redirect_stderr.getvalue()
                log_content = log_stream.getvalue()
                if stdout_content:
                    terminal_outputs.append(stdout_content)
                if stderr_content:
                    terminal_outputs.append(stderr_content)
                if log_content:
                    terminal_outputs.append(log_content)

                # 序列化输出状态
                serialized_output = serialize_agent_state(result)
                api_state.update_agent_data(
                    agent_name, "output_state", serialized_output)

                # 从状态中提取推理细节（如果有）
                reasoning_details = None
                if result.get("metadata", {}).get("show_reasoning", False):
                    if "agent_reasoning" in result.get("metadata", {}):
                        reasoning_details = result["metadata"]["agent_reasoning"]
                        api_state.update_agent_data(
                            agent_name,
                            "reasoning",
                            reasoning_details
                        )

                # 更新Agent状态为已完成
                api_state.update_agent_state(agent_name, "completed")

                return result
            except Exception as e:
                # Record end time even on error
                timestamp_end = datetime.now(UTC)
                error = str(e)
                # 恢复标准输出/错误
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                root_logger.removeHandler(log_handler)
                # 获取捕获的输出
                stdout_content = redirect_stdout.getvalue()
                stderr_content = redirect_stderr.getvalue()
                log_content = log_stream.getvalue()
                if stdout_content:
                    terminal_outputs.append(stdout_content)
                if stderr_content:
                    terminal_outputs.append(stderr_content)
                if log_content:
                    terminal_outputs.append(log_content)

                # 更新Agent状态为错误
                api_state.update_agent_state(agent_name, "error")
                # 记录错误信息
                api_state.update_agent_data(agent_name, "error", error)

                # 重新抛出异常
                raise

        return wrapper
    return decorator

# 启动API服务器的函数
def start_api_server(host="0.0.0.0", port=8000, stop_event=None):
    """在独立线程中启动API服务器"""
    if stop_event:
        # 使用支持优雅关闭的配置
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_config=None,
            # 开启ctrl+c处理
            use_colors=True
        )
        server = uvicorn.Server(config)

        # 运行服务器并在单独线程中监听stop_event
        def check_stop_event():
            # 在后台检查stop_event
            while not stop_event.is_set():
                time.sleep(0.5)
            # 当stop_event被设置时，请求服务器退出
            logger.info("收到停止信号，正在关闭API服务器...")
            server.should_exit = True

        # 启动stop_event监听线程
        stop_monitor = threading.Thread(
            target=check_stop_event,
            daemon=True
        )
        stop_monitor.start()

        # 运行服务器（阻塞调用，但会响应should_exit标志）
        try:
            server.run()
        except KeyboardInterrupt:
            # 如果还是收到了KeyboardInterrupt，确保我们的stop_event也被设置
            stop_event.set()
        logger.info("API服务器已关闭")
    else:
        # 默认方式启动，不支持外部停止控制但仍响应Ctrl+C
        uvicorn.run(app, host=host, port=port, log_config=None)

# 添加基本的API路由
@app.get("/", tags=["Root"])
def read_root():
    """API根路径"""
    return {
        "message": "A_Share_Investment_Agent API",
        "version": "0.1.0"
    }

@app.get("/agents/", tags=["Agents"])
def list_agents():
    """列出所有注册的Agents"""
    return {"agents": list(api_state.agents.values())}

@app.get("/status/", tags=["Status"])
def get_status():
    """获取系统状态"""
    return {
        "current_run_id": api_state.current_run_id,
        "current_agent": api_state.current_agent_name,
        "agents_count": len(api_state.agents),
        "runs_count": len(api_state.runs)
    }