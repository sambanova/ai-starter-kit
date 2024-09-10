from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, generate_from_stream
from langchain_core.messages import (
    AIMessageChunk, BaseMessage, AIMessage, ChatMessage,
    HumanMessage, ToolMessage, SystemMessage
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import get_from_dict_or_env, pre_init
from langchain_core.pydantic_v1 import Extra
import requests
import json


class SambaNovaCloud(BaseChatModel):
    """
    SambaNova Cloud chat model.
    To use, you should have the environment variables
    ``SAMBANOVA_URL`` set with your SambaNova Cloud URL.
    ``SAMBANOVA_API_KEY`` set with your SambaNova Cloud API Key.
    http://cloud.sambanova.ai/
    Example:
    .. code-block:: python
        SambaNovaCloud(
            base_url = SambaNova cloud endpoint URL,
            api_key = set with your SambaNova cloud API key,
            model = model name,
            streaming = set True for streaming
            max_tokens = max number of tokens to generate,
            temperature = model temperature,
            top_p = model top p,
            top_k = model top k,
            stream_options = include usage to get generation metrics
        )
    """

    base_url: str = ''
    """SambaNova Cloud Url"""

    api_key: str = ''
    """SambaNova Cloud api key"""

    model: str = 'llama3-8b'
    """The name of the model"""

    streaming: bool = False
    """Whether to use streaming or not"""

    max_tokens: int = 1024
    """max tokens to generate"""

    temperature: float = 0.7
    """model temperature"""

    top_p: float = 0.0
    """model top p"""

    top_k: int = 1
    """model top k"""

    stream_options: dict = {'include_usage': True}
    """stream options, include usage to get generation metrics"""

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model": self.model,
            "streaming": self.streaming,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream_options": self.stream_options
        }
    
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "sambanovacloud-chatmodel"

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['base_url'] = get_from_dict_or_env(
            values, 'base_url', 'SAMBANOVA_URL', default='https://api.sambanova.ai/v1/chat/completions'
        )
        values['api_key'] = get_from_dict_or_env(values, 'api_key', 'SAMBANOVA_API_KEY')
        return values

    def _handle_request(self, messages_dicts: List[Dict], stop: Optional[List[str]] = None):
        data = {
            'messages': messages_dicts,
            'max_tokens': self.max_tokens,
            'stop': stop,
            'model': self.model,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
        }
        http_session = requests.Session()
        response = http_session.post(
            self.base_url,
            headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
            json=data
        )
        if response.status_code != 200:
            raise RuntimeError(
                f'Sambanova /complete call failed with status code ' f'{response.status_code}.' f'{response.text}.'
            )
        response_dict = response.json()
        if response_dict.get("error"):
            raise RuntimeError(
                f"Sambanova /complete call failed with status code " f"{response['status_code']}." f"{response_dict}."
            )
        return response_dict


    def _handle_streaming_request(self):
        pass

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        elif isinstance(message, ToolMessage):
            message_dict = {"role": "tool", "content": message.content}
        else:
            raise TypeError(f"Got unknown type {message}")
        return message_dict

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        SambaNovaCloud chat model logic.

        Call SambaNovaCloud API.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            if stream_iter:
                return generate_from_stream(stream_iter)
        messages_dicts = self._create_message_dicts(messages)
        response = self._handle_request(messages_dicts, stop)
        message = AIMessage(
            content=response["choices"][0]["message"]["content"],
            additional_kwargs={},
            response_metadata={
                "finish_reason": response["choices"][0]["finish_reason"],
                "usage": response.get("usage"),
                "model_name": response["model"],
                "system_fingerprint": response["system_fingerprint"],
                "created": response["created"]
            },
            id=response["id"]
        )

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the output of the SambaNovaCloud chat model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        last_message = messages[-1]
        tokens = last_message.content[: self.n]

        for token in tokens:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))

            if run_manager:
                # This is optional in newer versions of LangChain
                # The on_llm_new_token will be called automatically
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

        # Let's add some other information (e.g., response metadata)
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
        )
        if run_manager:
            # This is optional in newer versions of LangChain
            # The on_llm_new_token will be called automatically
            run_manager.on_llm_new_token(token, chunk=chunk)
        yield chunk

