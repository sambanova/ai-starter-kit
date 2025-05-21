import base64
import json
from typing import Any, Dict, List, Optional

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI


def function_calling(
    messages: List[Dict[str, Any]],
    client: OpenAI,
    model: str,
    tools: List[Dict[str, Any]] = None,
    tool_choice: str = 'auto',
    parallel_tool_calls: bool = True,
    response_format: Dict[str, Any] = None,
    stream: bool = False,
) -> Any:
    """
    Sends a chat completion request to the OpenAI API with optional tool calling configuration.

    This function handles sending a list of messages to the OpenAI chat model, with optional tool usage,
    parallel tool calls, and streaming of responses. It captures the response or any error that may occur.

    Parameters:
        messages (List[Dict[str, Any]]): The conversation history/messages to pass to the model.
        client (OpenAI): An instance of the OpenAI client to make API calls.
        model (str): The model name to use for generating completions.
        tools (List[Dict[str, Any]], optional): A list of tool specifications (functions) that the model can call.
        tool_choice (str, optional): Determines how the model chooses a tool. Defaults to 'auto'.
        parallel_tool_calls (bool, optional): Whether the model can call multiple tools in parallel. Defaults to False.
        response_format (Dict[str, Any], optional): Optional format specification for the API response
          (e.g., {'type': 'json'}).
        stream (bool, optional): Whether to stream the response chunks as they are generated. Defaults to False.

    Returns:
        Any: The response from the OpenAI API. If `stream` is True, a list of streamed response chunks (choices) is
        returned. If `stream` is False, the final assistant message (or an error message) is returned.

    Raises:
        Exception: Propagates any exception that occurs during the API call.
    """
    if tools is not None:
        tools_args = {'tools': tools, 'tool_choice': tool_choice, 'parallel_tool_calls': parallel_tool_calls}
    else:
        tools_args = {}

    results = []

    try:
        completion = client.chat.completions.create(
            model=model, messages=messages, stream=stream, response_format=response_format, **tools_args
        )
        if stream:
            for chunk in completion:
                results.append(chunk.choices)
        else:
            if completion and hasattr(completion, 'error'):
                results = f'Error: {completion.error}'
            else:
                results = completion.choices[0].message
    except Exception as e:
        raise e

    return results


def mcp_to_json_schema(mcp_tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts an MCP tool definition to a JSON schema for tool calling.

    Args:
        mcp_tool: A dictionary representing the MCP tool definition.

    Returns:
        A dictionary representing the JSON schema.
    """
    json_schema = {
        'type': 'function',
        'function': {
            'name': mcp_tool['name'],
            'description': mcp_tool['description'],
            'parameters': {'type': 'object', 'properties': {}},
        },
    }

    if 'inputSchema' in mcp_tool and mcp_tool['inputSchema']:
        input_schema = mcp_tool['inputSchema']
        if 'properties' in input_schema:
            for param_name, param_details in input_schema['properties'].items():
                json_schema['function']['parameters']['properties'][param_name] = {
                    'type': param_details['type'],
                    'description': param_details.get('description', ''),
                }

                if 'required' in input_schema and param_name in input_schema['required']:
                    if 'required' not in json_schema['function']['parameters']:
                        json_schema['function']['parameters']['required'] = []
                    json_schema['function']['parameters']['required'].append(param_name)

                if 'default' in param_details:
                    json_schema['function']['parameters']['properties'][param_name]['default'] = param_details[
                        'default'
                    ]

    if '$defs' in mcp_tool:
        json_schema['function']['parameters']['$defs'] = mcp_tool['$defs']

    return json_schema


async def mcp_client(
    available_tools: List[str], client: OpenAI, model: str, messages: List[Dict[str, Any]], stream: bool
) -> Any:
    """
    Interfaces with an MCP (Multi-Component Protocol) server to retrieve tool schemas, filter them based on
    availability, and send a chat completion request to the OpenAI API using those tools.

    This function:
    - Starts a subprocess running an MCP-compatible tool server.
    - Connects to it using a stdio client.
    - Initializes a session and retrieves the list of available tools from the server.
    - Converts each tool to the appropriate JSON schema format if it's included in `available_tools`.
    - Calls the `function_calling` function to send the chat completion request using the filtered tools.

    Parameters:
        available_tools (List[str]): A list of tool names that are allowed for use in the completion request.
        client (OpenAI): An instance of the OpenAI client to make the API call.
        model (str): The name of the OpenAI model to use.
        messages (List[Dict[str, Any]]): The chat messages to send to the model.
        stream (bool): Whether to stream the model's response incrementally.

    Returns:
        Any: The response from the OpenAI API via the `function_calling` method.
             If `stream` is True, returns a list of streamed chunks; otherwise, returns the final message.

    Raises:
        Exception: Any exceptions raised during server communication or the OpenAI API call are propagated.
    """

    server_params = StdioServerParameters(
        command='python',
        args=['tests/mcp_example_server.py'],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()

            tools_schemas = []

            for tool in tools.tools:
                if tool.name in available_tools:
                    tool_dict = json.loads(tool.model_dump_json())
                    final_tool = mcp_to_json_schema(tool_dict)
                    tools_schemas.append(final_tool)
            response = function_calling(messages, client, model, tools_schemas, stream=stream)
            return response


def read_json_file(file_path: str) -> Any:
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary. If the file does not exist or the JSON is invalid,
          returns None.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        json.JSONDecodeError: If the JSON data is invalid.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f'File not found at {file_path}')
        return None
    except json.JSONDecodeError as e:
        print(f'Error parsing JSON: {e}')
        return None


def image_to_base64(image_path: str) -> str:
    """
    Convert an image to base64.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded image.
    """
    with open(image_path, 'rb') as img_file:
        base64_img = base64.b64encode(img_file.read()).decode('utf-8')
        return base64_img


def encode_to_base64(content: bytes) -> str:
    """Encode audio file to base64"""
    return base64.b64encode(content).decode('utf-8')


def load_encode_audio(path: str) -> str:
    with open(path, 'rb') as file:
        audio = file.read()
    b64_audio = encode_to_base64(content=audio)
    return b64_audio


def audio_requests(
    url: str,
    api_key: str,
    file_path: str,
    language: str = 'english',
    model: str = 'Qwen2-Audio-7B-Instruct',
    prompt: Optional[str] = None,
    temperature: float = 0.01,
) -> Dict[str, Any] | str:
    headers = {
        'Authorization': f'Bearer {api_key}',
    }
    with open(file_path, 'rb') as file:
        files: Dict[str, Any] = {
            'file': file,
        }
        data = {
            'model': model,
            'language': language,
            'response_format': 'json',
            'temperature': str(temperature),
        }
        if prompt:
            files['prompt'] = prompt

        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        return response.text
