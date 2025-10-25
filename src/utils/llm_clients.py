import os
import time
import backoff
from abc import ABC, abstractmethod
from openai import OpenAI
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

# Setup logging
logger = setup_logger('llm_clients')


class LLMClient(ABC):
    """LLM client abstract base class"""

    @abstractmethod
    def get_completion(self, messages, **kwargs):
        """Get model response"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI Compatible API client"""

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        self.model = model or os.getenv("OPENAI_COMPATIBLE_MODEL")

        if not self.api_key:
            logger.error(f"{ERROR_ICON} OPENAI_COMPATIBLE_API_KEY environment variable not found")
            raise ValueError(
                "OPENAI_COMPATIBLE_API_KEY not found in environment variables")

        if not self.base_url:
            logger.error(f"{ERROR_ICON} OPENAI_COMPATIBLE_BASE_URL environment variable not found")
            raise ValueError(
                "OPENAI_COMPATIBLE_BASE_URL not found in environment variables")

        if not self.model:
            logger.error(f"{ERROR_ICON} OPENAI_COMPATIBLE_MODEL environment variable not found")
            raise ValueError(
                "OPENAI_COMPATIBLE_MODEL not found in environment variables")

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        logger.info(f"{SUCCESS_ICON} OpenAI Compatible client initialized successfully")

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=5,
        max_time=300
    )
    def call_api_with_retry(self, messages, stream=False):
        """API call function with retry mechanism"""
        try:
            logger.info(f"{WAIT_ICON} Calling OpenAI API...")
            logger.debug(f"Request content: {messages}")
            logger.debug(f"Model: {self.model}, Stream: {stream}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream
            )

            logger.info(f"{SUCCESS_ICON} API call successful")
            return response
        except Exception as e:
            error_msg = str(e)
            logger.error(f"{ERROR_ICON} API call failed: {error_msg}")
            raise e

    def get_completion(self, messages, max_retries=3, initial_retry_delay=1, **kwargs):
        """Get chat completion result with retry logic"""
        try:
            logger.info(f"{WAIT_ICON} Using OpenAI model: {self.model}")
            logger.debug(f"Message content: {messages}")

            for attempt in range(max_retries):
                try:
                    # Call API
                    response = self.call_api_with_retry(messages)

                    if response is None:
                        logger.warning(
                            f"{ERROR_ICON} Attempt {attempt + 1}/{max_retries}: API returned null value")
                        if attempt < max_retries - 1:
                            retry_delay = initial_retry_delay * (2 ** attempt)
                            logger.info(
                                f"{WAIT_ICON} Waiting {retry_delay} seconds before retry...")
                            time.sleep(retry_delay)
                            continue
                        return None

                    # Print debug information
                    content = response.choices[0].message.content
                    logger.debug(f"API raw response: {content[:500]}...")
                    logger.info(f"{SUCCESS_ICON} Successfully obtained OpenAI response")

                    # Return text content directly
                    return content

                except Exception as e:
                    logger.error(
                        f"{ERROR_ICON} Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"{ERROR_ICON} Final error: {str(e)}")
                        return None

        except Exception as e:
            logger.error(f"{ERROR_ICON} Error in get_completion: {str(e)}")
            return None


class LLMClientFactory:
    """LLM client factory class"""

    @staticmethod
    def create_client(**kwargs):
        """
        Create OpenAI Compatible client

        Args:
            **kwargs: Client configuration parameters, including api_key, base_url and model

        Returns:
            LLMClient: Instantiated OpenAI Compatible client
        """
        return OpenAIClient(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
            model=kwargs.get("model")
        )