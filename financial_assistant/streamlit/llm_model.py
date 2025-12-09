from typing import Any
import streamlit

from financial_assistant.constants import CONFIG_PATH
from financial_assistant.src.llm import SambaNovaLLM


def get_sambanova_llm() -> SambaNovaLLM:
    api_key = streamlit.session_state.get('SAMBANOVA_API_KEY', '')
    api_base = streamlit.session_state.get('SAMBANOVA_API_BASE', '')
    cache_key = (api_key, api_base)
    if streamlit.session_state.get('_llm_cache_key') != cache_key:
        streamlit.session_state._llm_cache_key = cache_key
        streamlit.session_state._sambanova_llm = SambaNovaLLM(
            config_path=CONFIG_PATH,
            sambanova_api_key=api_key,
            sambanova_api_base=api_base,
        )
    return streamlit.session_state._sambanova_llm  # type: ignore[attr-defined]


class _SambaNovaLLMProxy:
    def __getattr__(self, attr: str) -> Any:
        return getattr(get_sambanova_llm(), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        return setattr(get_sambanova_llm(), attr, value)


sambanova_llm = _SambaNovaLLMProxy()
