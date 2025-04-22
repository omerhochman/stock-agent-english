import os
import time
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from openai import OpenAI
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
from src.utils.llm_clients import LLMClientFactory

# 设置日志记录
logger = setup_logger('api_calls')


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

# 验证环境变量
api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
model = os.getenv("OPENAI_COMPATIBLE_MODEL")

if not api_key:
    logger.error(f"{ERROR_ICON} 未找到 OPENAI_COMPATIBLE_API_KEY 环境变量")
    raise ValueError("OPENAI_COMPATIBLE_API_KEY not found in environment variables")
if not base_url:
    logger.error(f"{ERROR_ICON} 未找到 OPENAI_COMPATIBLE_BASE_URL 环境变量")
    raise ValueError("OPENAI_COMPATIBLE_BASE_URL not found in environment variables")
if not model:
    logger.error(f"{ERROR_ICON} 未找到 OPENAI_COMPATIBLE_MODEL 环境变量")
    raise ValueError("OPENAI_COMPATIBLE_MODEL not found in environment variables")

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)
logger.info(f"{SUCCESS_ICON} OpenAI Compatible 客户端初始化成功")


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300
)
def call_api_with_retry(model, messages):
    """带重试机制的 API 调用函数"""
    try:
        logger.info(f"{WAIT_ICON} 正在调用 OpenAI Compatible API...")
        logger.debug(f"请求内容: {messages}")
        logger.debug(f"模型: {model}")

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        logger.info(f"{SUCCESS_ICON} API 调用成功")
        content = response.choices[0].message.content
        logger.debug(f"响应内容: {content[:500]}...")
        return response
    except Exception as e:
        error_msg = str(e)
        logger.error(f"{ERROR_ICON} API 调用失败: {error_msg}")
        raise e


def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1,
                        api_key=None, base_url=None):
    """
    获取聊天完成结果，包含重试逻辑

    Args:
        messages: 消息列表，OpenAI 格式
        model: 模型名称（可选）
        max_retries: 最大重试次数
        initial_retry_delay: 初始重试延迟（秒）
        api_key: API 密钥（可选，用于覆盖环境变量）
        base_url: API 基础 URL（可选，用于覆盖环境变量）

    Returns:
        str: 模型回答内容或 None（如果出错）
    """
    try:
        # 使用参数提供的值或默认值
        use_api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        use_base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        use_model = model or os.getenv("OPENAI_COMPATIBLE_MODEL")

        # 创建客户端
        client = LLMClientFactory.create_client(
            api_key=use_api_key,
            base_url=use_base_url,
            model=use_model
        )

        # 获取回答
        return client.get_completion(
            messages=messages,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay
        )
    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None