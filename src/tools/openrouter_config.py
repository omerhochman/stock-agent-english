import os
import time
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from openai import OpenAI
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
from src.utils.llm_clients import LLMClientFactory

# Setup logging
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


# Get project root directory
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# Load environment variables
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} Environment variables loaded: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} Environment variables file not found: {env_path}")

# Validate environment variables
api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
model = os.getenv("OPENAI_COMPATIBLE_MODEL")

if not api_key:
    logger.error(f"{ERROR_ICON} OPENAI_COMPATIBLE_API_KEY environment variable not found")
    raise ValueError("OPENAI_COMPATIBLE_API_KEY not found in environment variables")
if not base_url:
    logger.error(f"{ERROR_ICON} OPENAI_COMPATIBLE_BASE_URL environment variable not found")
    raise ValueError("OPENAI_COMPATIBLE_BASE_URL not found in environment variables")
if not model:
    logger.error(f"{ERROR_ICON} OPENAI_COMPATIBLE_MODEL environment variable not found")
    raise ValueError("OPENAI_COMPATIBLE_MODEL not found in environment variables")

# Initialize OpenAI client
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)
logger.info(f"{SUCCESS_ICON} OpenAI Compatible client initialized successfully")


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300
)
def call_api_with_retry(model, messages):
    """API call function with retry mechanism"""
    try:
        logger.info(f"{WAIT_ICON} Calling OpenAI Compatible API...")
        logger.debug(f"Request content: {messages}")
        logger.debug(f"Model: {model}")

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        logger.info(f"{SUCCESS_ICON} API call successful")
        content = response.choices[0].message.content
        logger.debug(f"Response content: {content[:500]}...")
        return response
    except Exception as e:
        error_msg = str(e)
        logger.error(f"{ERROR_ICON} API call failed: {error_msg}")
        raise e


def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1,
                        api_key=None, base_url=None):
    """
    Get chat completion result with retry logic

    Args:
        messages: Message list in OpenAI format
        model: Model name (optional)
        max_retries: Maximum number of retries
        initial_retry_delay: Initial retry delay (seconds)
        api_key: API key (optional, to override environment variables)
        base_url: API base URL (optional, to override environment variables)

    Returns:
        str: Model response content or None (if error occurs)
    """
    try:
        # Use provided parameter values or default values
        use_api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        use_base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        use_model = model or os.getenv("OPENAI_COMPATIBLE_MODEL")

        # Create client
        client = LLMClientFactory.create_client(
            api_key=use_api_key,
            base_url=use_base_url,
            model=use_model
        )

        # Get response
        return client.get_completion(
            messages=messages,
            max_retries=max_retries,
            initial_retry_delay=initial_retry_delay
        )
    except Exception as e:
        logger.error(f"{ERROR_ICON} Error in get_chat_completion: {str(e)}")
        return None