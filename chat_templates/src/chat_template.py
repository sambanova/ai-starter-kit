"""
chat_template.py
Core logic for Custom Chat Templates kit.
Handles tokenizer loading, template rendering, completions invocation, and model-specific output parsing.
"""

import os
import json
import uuid
import logging
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer
from sambanova import SambaNova
from pydantic import BaseModel
from json import JSONDecodeError
from jinja2 import Template, Environment, TemplateSyntaxError
import re

# Paths and environment setup
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

load_dotenv(os.path.join(repo_dir, ".env"), override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = "data"
BASE_URL = "https://api.sambanova.ai/v1"

class ToolCallModel(BaseModel):
    """Schema validator for a tool-call output."""
    name: str
    arguments: dict

class ChatTemplateManager:
    """Session-scoped chat template manager."""
    def __init__(self, hf_token: str | None = None, sambanova_api_key: str | None = None, sambanova_api_base: str | None = None,  cache_dir: str | None = None):
        self.hf_token = hf_token
        self.sambanova_api_key = sambanova_api_key or os.environ.get("SAMBANOVA_API_KEY")
        self.sambanova_api_base = sambanova_api_base or os.environ.get("SAMBANOVA_API_BASE", BASE_URL)
        self.cache_dir = cache_dir or os.path.join(kit_dir, CACHE_DIR)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.tokenizers: dict[str, AutoTokenizer] = {}
        self.chat_templates: dict[str, str | None] = {}
        self.additional_chat_templates_context: dict[str, str | None] = {}
        self.parsers: dict[str, callable] = {}
        self.register_parser("llama_json_parser", self.llama3_parser)
        self.register_parser("deepseek_xml_parser", self.deepseek_v3_parser)

    # Tokenizer and chat-template handlers
    def load_tokenizer(self, model_name: str):
        """
        Load a Hugging Face tokenizer for the given model.

        Args:
            model_name: HF model identifier (e.g. "sambanova/Meta-Llama-3-8B-Instruct").

        Returns:
            AutoTokenizer instance ready for chat-template use.
        """  
        if model_name in self.tokenizers:
            return self.tokenizers[model_name]
        
        tokenizer_path = os.path.join(self.cache_dir, model_name.replace("/", "_"))
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=self.hf_token,
                cache_dir=self.cache_dir
            )
        self.tokenizers[model_name]=tokenizer
        return tokenizer
    
    def get_chat_template(self, model_name: str) -> str | None:
        """
        Retrieve a model’s built-in chat template, downloading tokenizer if needed.

        Args:
            model_name: HF model name.
            hf_token: Optional auth token.
            cache_dir: Optional path to cache directory (defaults to kit/data).

        Returns:
            Template string or None if not available.
        """
        
        if model_name in self.chat_templates:
            return self.chat_templates[model_name]
        
        tokenizer = self.load_tokenizer(model_name)
        chat_template = getattr(tokenizer, "chat_template", None)
        if not chat_template:
            error_msg = f"No chat template found for model {model_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.chat_templates[model_name]=chat_template
        return self.chat_templates[model_name]
        
    def _validate_jinja_template(self, template: str) -> bool:
        """
        Validate that a Jinja2 template compiles.

        Raises:
            ValueError: If syntax errors are found.

        Returns:
            True if template is syntactically valid.
        """
        try:
            env = Environment()
            env.parse(template)
            return True
        except TemplateSyntaxError as e:
            error_msg = f"Invalid Jinja template syntax at line {e.lineno}: {e.message}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        
    def set_chat_template(self, model_name: str, template: str, additional_context: dict[str, str]=None) -> None:
        """Validate a chat template and add it to chat templates dict"""
        self._validate_jinja_template(template)
        self.chat_templates[model_name]=template
        if additional_context is not None:
            self.additional_chat_templates_context[model_name]=additional_context
        logger.info(f"Chat template set for {model_name}")
        
    def _extract_tokenizer_context(self, tokenizer) -> dict:
        """
        Collect all simple (JSON-serializable) attributes from the tokenizer
        that may be referenced by the Jinja chat template.
        """
        context = {}
        for key, value in vars(tokenizer).items():
            # Skip private/internal and complex objects
            if key.startswith("_"):
                continue
            if isinstance(value, (str, int, float, list, dict, tuple, type(None))):
                context[key] = value
        return context
    
    def apply_chat_template(self, model_name: str, messages: list, tools: list | None = None, add_generation_prompt: bool = True, extra_context: dict | None = None) -> str:
        """
        Render a Jinja2 chat template with given messages and optional tools.

        Args:
            model_name: The model chat template to apply.
            messages: List of dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            add_generation_prompt: Whether to append model’s assistant prefix.
            extra_context: Extra variables to inject into the Jinja context.

        Returns:
            str: Rendered chat prompt string ready for the completions API.
        """
        template_str = self.get_chat_template(model_name)
        tokenizer = self.tokenizers.get(model_name)
        
        
        context = {
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools is not None:
            context["tools"]=tools
        
        # Add tokenizer tokens if available
        if tokenizer:
            context.update(self._extract_tokenizer_context(tokenizer))
        
        # Merge any extra context provides when setting a custom chat template  
        if self.additional_chat_templates_context.get(model_name) is not None:
            context.update(self.additional_chat_templates_context[model_name])
                            
        # Merge any user-provided extra context
        if extra_context:
            context.update(extra_context)
        
        try:
            template = Template(template_str)
            rendered = template.render(**context)
            return rendered.strip()
        except Exception as e:
            error_msg = f"Error rendering custom Jinja template: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Completions API invocation
    def completions_invoke(self, prompt: str, model: str, **kwargs: dict[str, any] | None):
        """
        Send a raw prompt to the SambaNova Completions API.

        Args:
            prompt: Text input to model.
            model: Model name to invoke.
            kwargs: additional params.

        Returns:
            Raw text output from the model.
        """
        client = SambaNova(api_key=self.sambanova_api_key, base_url=self.sambanova_api_base)
        try:
             response = client.completions.create(model=model, prompt=prompt, **kwargs)
             return response.choices[0].text
        except Exception as e:
             error_msg = f"SambaNova API error: {e}"
             logger.error(error_msg)
             raise ValueError(error_msg)
        
    #  Tool-call models
    def _generate_random_id(self, length=18):
        """Generate random ID prefix 'call_' of given length."""
        return "call_" + str(uuid.uuid4()).replace("-", "")[:length]

    def instantiate_function_calling_model(self, model_name: str, parameters: dict):
        """
        Validate and format a function-call into OpenAI tool-call schema.
        """
        ToolCallModel(name=model_name, arguments=parameters)
        return {
            "id": self._generate_random_id(),
            "type": "function",
            "function": {"name": model_name, "arguments": json.dumps(parameters)},
        }

    def register_parser(self, name: str, func):
        """Register a new parser in the global registry."""
        self.parsers[name] = func

    ### Llama 
    def _extract_llama_json_strings(self, response: str):
        """Extract every top-level {...} JSON object from text."""
        fc_strings, brace_count, start = [], 0, None
        for i, ch in enumerate(response):
            if ch == "{":
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif ch == "}" and start is not None:
                brace_count -= 1
                if brace_count == 0:
                    fc_strings.append(response[start : i + 1])
                    start = None
        return fc_strings

    def llama3_parser(self, response: str):
        """Parse JSON-style tool calls (Llama3)."""
        calls = []
        for js in self._extract_llama_json_strings(response):
            try:
                obj = json.loads(js)
                name, params = obj["name"], obj["parameters"]
                calls.append(self.instantiate_function_calling_model(name, params))
            except Exception as e:
                error_msg = f"Invalid tool call in block {js[:120]}...: {e}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        return calls

    ### deepseek
    def _extract_deepseek_v3_xml_strings(self, response: str):
        """Extract tool-call pairs from DeepSeek XML markers."""
        fc_strings = []
        pattern = r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>"
        for match in re.findall(pattern, response, re.DOTALL):
            name, args = match
            try:
                args = json.loads(args)
                fc_strings.append({"name": name.strip(), "parameters": args})
            except JSONDecodeError as e:
                error_msg = (
                    f"Invalid JSON in DeepSeek tool-call arguments for '{name.strip()}': {e.msg}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        return fc_strings

    def deepseek_v3_parser(self, response: str):
        """Parse DeepSeek XML-style tool calls into OpenAI format."""
        calls = []
        for fc in self._extract_deepseek_v3_xml_strings(response):
            calls.append(self.instantiate_function_calling_model(fc["name"], fc["parameters"]))
        return calls


    def add_custom_tool_parser(self, name: str, code_str: str):
        """
        Dynamically register a parser from user code.

        The provided code must define:
            def parse(response: str) -> list'
            
        Args:
            name: Name to register the parser under.
            code_str: Python source code defining a function `parse(response: str) -> list`.

        Raises:
            ValueError: If the code is invalid or does not define a callable named `parse`.
            def parse(response: str):
            
        Sample Dummy custom parser
            return [{"id": "call_custom", "type": "function", "function": {"name": "hello", "arguments": "{}"}}]
        """

        local_env = {}
        try:
            compiled = compile(code_str, "<custom_parser>", "exec")
            exec(compiled, {}, local_env)
            
            user_fn = local_env.get("parse")
            if not callable(user_fn):
                raise ValueError("Custom code must define a function named `parse`.")
            # Optional safety: check signature
            if user_fn.__code__.co_argcount != 1:
                raise ValueError("Custom code `parse` method must accept exactly one argument: response.")
            
            def validated_parser(response: str):
                """Wrapper to enforce output schema validation."""
                raw_output = user_fn(response)
                if not isinstance(raw_output, list):
                    raise ValueError("Custom code parser must return a list of tool-call dicts.")
                validated = []
                for i, item in enumerate(raw_output):
                    if not isinstance(item, dict):
                        raise ValueError(f"Item {i} in custom code parser output must be a dict, got {type(item)}.")

                    fn = item.get("function")
                    if not isinstance(fn, dict):
                        raise ValueError(f"Item {i} in custom code missing 'function' key or not a dict.")

                    name = fn.get("name")
                    args = fn.get("arguments")
                    if not isinstance(name, str):
                        raise ValueError(f"Function name must be a string, got {type(name)} in item {i} of custom code parser output.")
                    if not isinstance(args, str):
                        raise ValueError(f"Function arguments must be a dumped json in str, got {type(args)} in item {i} of custom code parser output.")
                    try: 
                        args_dict = json.loads(args)
                    except Exception as e:
                        raise ValueError(f"Function arguments couldn't be loaded as json {type(args)} in item {i} of custom code parser output. \n{e}")

                    # Enforce proper OpenAI tool-call schema via Pydantic
                    validated.append(self.instantiate_function_calling_model(name, args_dict))
                return validated
            
            self.parsers[name] = validated_parser
            logger.info(f"Custom parser '{name}' registered successfully.")
            
        except SyntaxError as e:
            msg = f"Syntax error in custom parser '{name}' at line {e.lineno}: {e.msg}"
            logger.error(msg)
            raise ValueError(msg)
        except Exception as e:
            msg = f"Failed to register custom parser '{name}': {e}"
            logger.error(msg)
            raise ValueError(msg)

    ## parse to message structure
    def parse_to_message(self, response: str, parser_name: str):
        """
        Convert a raw model response to an assistant message object.

        Args:
            response: Raw text from completions API.
            name: Parser name (e.g., 'llama', 'deepseek', or custom).

        Returns:
            dict: OpenAI-style assistant message with role/content/tool_calls.
        """
        parser = self.parsers.get(parser_name)
        tool_calls = parser(response) if parser else []

        if tool_calls:
            return {"role": "assistant", "content": None, "tool_calls": tool_calls}
        return {"role": "assistant", "content": response.strip(), "tool_calls": []}

