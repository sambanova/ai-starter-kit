import abc
import json
import os
import sys
import time
from datetime import datetime
from math import isclose
from typing import Any, Dict, List, Tuple

import requests
import sseclient
from requests import Response

sys.path.append('./src')
sys.path.append('./src/llmperf')

import warnings

from dotenv import load_dotenv
from transformers import AutoTokenizer

from benchmarking.src.llmperf import common_metrics
from benchmarking.src.llmperf.llmperf_utils import get_tokenizer
from benchmarking.src.llmperf.models import RequestConfig
from benchmarking.utils import SAMBANOVA_URL

warnings.filterwarnings('ignore')


class BaseAPIEndpoint(abc.ABC):
    def __init__(self, request_config: RequestConfig, tokenizer: AutoTokenizer) -> None:
        self.request_config = request_config
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def _get_url(self, *args: Any, **kwargs: Any) -> str:
        pass

    @abc.abstractmethod
    def _get_headers(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        pass

    @abc.abstractmethod
    def _get_json_data(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        pass

    def _get_token_length(self, input_text: str) -> int:
        """Gets the token length of a piece of text

        Args:
            input_text (str): input text

        Returns:
            int: number of tokens
        """
        return len(self.tokenizer.encode(input_text))

    def _calculate_tpot_from_streams_after_first(
        self, chunks_received: List[str], chunks_timings: List[int | float]
    ) -> float:
        """Calculates Time per Output Token (TPOT) based on the streaming events coming after the first one.
        In general, the way to calculate this metric is: time_to_generate_tokens/number_of_tokens_generated

        Args:
            chunks_received (list): complete list of events coming from streaming response
            chunks_timings (list): complete list of timings that each event took to process

        Returns:
            float: calculated tpot
        """

        # Calculate tokens
        total_tokens_received_after_first_chunk = sum(self._get_token_length(c) for c in chunks_received[1:])

        # Calculate time
        total_time_to_receive_tokens_after_first_chunk = sum(chunks_timings[1:])

        # Calculate tpot
        tpot = float(total_time_to_receive_tokens_after_first_chunk / total_tokens_received_after_first_chunk)

        return tpot

    def _calculate_ttft_from_streams(
        self, chunks_received: List[str], chunks_timings: List[int | float], total_request_time: int | float
    ) -> float:
        """Calculates Time to First Token (TTFT) based on the streaming events coming from the response.
        If there are enough streaming events, the formula to calculate ttft is:
        time_first_chunk - (tokens_first_chunk - 1) * tpot

        Args:
            chunks_received (list): list of events having the streaming tokens
            chunks_timings (list): list of timings for each event
            total_request_time (int): total request time calculated from client side

        Returns:
            float: calculated ttft
        """

        number_chunks_recieved = len(chunks_received)

        # if one or no chunks were recieved
        if number_chunks_recieved <= 1:
            ttft = total_request_time
        else:
            # ttft = chunks_timings[0]
            if len(chunks_received) <= 1:
                ttft = chunks_timings[0]
            else:
                # calculate tpot
                tpot = self._calculate_tpot_from_streams_after_first(chunks_received, chunks_timings)
                # calculate ttft
                total_tokens_in_first_chunk = self._get_token_length(chunks_received[0])
                ttft = chunks_timings[0] - (total_tokens_in_first_chunk - 1) * tpot
        return ttft

    def _populate_client_metrics(
        self,
        prompt_len: int,
        num_output_tokens: int,
        ttft: int | float,
        total_request_time: int | float,
        server_metrics: Dict[str, Any],
        number_chunks_recieved: int,
    ) -> Dict[str, Any]:
        """Populates `metrics` dictionary with performance metrics calculated from client side

        Args:
            prompt_len (int): prompt's length
            num_output_tokens (int): number of output tokens
            ttft (int): time to first token
            total_request_time (int): end-to-end latency
            server_metrics (dict):  server metrics dictionary
            number_chunks_recieved (int): number of chunks recieved

        Returns:
            dict: updated metrics dictionary
        """

        metrics = server_metrics

        metrics[common_metrics.NUM_INPUT_TOKENS] = (
            prompt_len
            if metrics[common_metrics.NUM_INPUT_TOKENS_SERVER] is None
            else metrics[common_metrics.NUM_INPUT_TOKENS_SERVER]
        )

        metrics[common_metrics.NUM_OUTPUT_TOKENS] = (
            num_output_tokens
            if metrics[common_metrics.NUM_OUTPUT_TOKENS_SERVER] is None
            else metrics[common_metrics.NUM_OUTPUT_TOKENS_SERVER]
        )

        metrics[common_metrics.NUM_TOTAL_TOKENS] = (
            prompt_len + num_output_tokens
            if metrics[common_metrics.NUM_TOTAL_TOKENS_SERVER] is None
            else metrics[common_metrics.NUM_TOTAL_TOKENS_SERVER]
        )

        metrics[common_metrics.TTFT] = ttft

        metrics[common_metrics.E2E_LAT] = total_request_time

        if number_chunks_recieved == 1:
            metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
                metrics[common_metrics.NUM_OUTPUT_TOKENS] / total_request_time
            )
        else:
            metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
                metrics[common_metrics.NUM_OUTPUT_TOKENS] / (total_request_time - ttft)
                if not isclose(ttft, total_request_time, abs_tol=1e-8)
                else None
            )

        metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT] = (prompt_len + num_output_tokens) / total_request_time

        return metrics

    def _populate_server_metrics(self, response_dict: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Parse output data to metrics dictionary structure

        Args:
            response_dict (dict): dict data with performance metrics
            metrics (dict): metrics dictionary

        Returns:
            dict: updated metrics dictionary
        """

        metrics[common_metrics.NUM_INPUT_TOKENS_SERVER] = response_dict.get('prompt_tokens_count') or response_dict.get(
            'prompt_tokens'
        )

        metrics[common_metrics.NUM_OUTPUT_TOKENS_SERVER] = response_dict.get(
            'completion_tokens_count'
        ) or response_dict.get('completion_tokens')

        metrics[common_metrics.NUM_TOTAL_TOKENS_SERVER] = response_dict.get('total_tokens_count') or response_dict.get(
            'total_tokens'
        )
        ttft_server = response_dict.get('time_to_first_token') or response_dict.get('time_to_first_response')

        metrics[common_metrics.TTFT_SERVER] = ttft_server

        metrics[common_metrics.E2E_LAT_SERVER] = response_dict.get('total_latency') or response_dict.get(
            'model_execution_time'
        )

        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT_SERVER] = (
            response_dict.get('completion_tokens_after_first_per_sec')
            or response_dict.get('completion_tokens_per_sec_after_first_response')
            or response_dict.get('throughput_after_first_token')
        )

        metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT_SERVER] = response_dict.get('total_tokens_per_sec')
        if (metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT_SERVER] is None) and (
            metrics[common_metrics.E2E_LAT_SERVER] is not None
        ):
            metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT_SERVER] = (
                metrics[common_metrics.NUM_TOTAL_TOKENS_SERVER] / (metrics[common_metrics.E2E_LAT_SERVER])
            )

        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT_SERVER_FIRST_TEN] = response_dict.get(
            'completion_tokens_after_first_per_sec_first_ten'
        )
        metrics[common_metrics.BATCH_SIZE_USED] = response_dict.get('batch_size_used')
        metrics[common_metrics.ACCEPTANCE_RATE] = response_dict.get('acceptance_rate')

        return metrics


class SambaStudioAPI(BaseAPIEndpoint):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Load sambastudio env variables
        if self.request_config.api_variables:
            self.base_url = self.request_config.api_variables['SAMBASTUDIO_URL']
            self.api_key = self.request_config.api_variables['SAMBASTUDIO_API_KEY']
        else:
            self.base_url = os.environ.get('SAMBASTUDIO_URL', '')
            self.api_key = os.environ.get('SAMBASTUDIO_API_KEY', '')

    def _get_url(self) -> str:
        """
        Get streaming and non streaming URLs from the given URL

        Args:
            url: string with sambastudio base or streaming endpoint url

        Returns:
            streaming_url: string with url to do streaming calls
        """
        if 'chat/completions' in self.base_url:
            stream_url = self.base_url
        else:
            if 'stream' in self.base_url:
                stream_url = self.base_url
                if self.request_config.image:
                    raise ValueError(
                        f'Image support not available for url: {self.base_url}.\
                        Try with OpenAI compatible endpoint.'
                    )
            else:
                if 'generic' in self.base_url:
                    stream_url = 'generic/stream'.join(self.base_url.split('generic'))
                else:
                    raise ValueError('Unsupported URL')
        return stream_url

    def _get_headers(self) -> Dict[str, str]:
        """Gets headers for API call"""
        assert isinstance(self.api_key, str), 'No API KEY provided'

        if 'chat/completions' in self.base_url:  # SambaStudio compatible with OpenAI request
            return {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        else:  # Regular SambaStudio request
            return {'key': self.api_key}

    def _get_json_data(self, url: str) -> Dict[str, Any]:
        """Gets json body for API call

        Args:
            url: URL being used for the API call

        Returns:
            dict: API call body according to Bundle and streaming conditions
        """
        prompt = self.request_config.prompt_tuple[0]['template']
        sampling_params = self.request_config.sampling_params

        assert isinstance(sampling_params, dict), f'sampling_params must be a dict. Got type {type(sampling_params)}'

        if 'chat/completions' in self.base_url:  # SambaStudio compatible with OpenAI data payload
            data = self._get_json_data_for_sambastudio_openai_compatible(prompt, sampling_params)
        else:  # Regular SambaStudio data payload
            data = self._get_json_data_for_regular_sambastudio(url, prompt, sampling_params)

        return data

    def _get_json_data_for_sambastudio_openai_compatible(
        self, prompt: str, sampling_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        sampling_params['model'] = self.request_config.model
        sampling_params['max_tokens'] = sampling_params.pop('max_tokens_to_generate')

        if self.request_config.is_stream_mode:
            sampling_params['stream'] = True
            sampling_params['stream_options'] = {'include_usage': True}
        else:
            # TODO: support not streaming mode
            raise ValueError('Streaming mode required')

        # If an image is provided, add it to the content
        content: Any = None
        if self.request_config.image:
            content = [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{self.request_config.image}'}},
            ]
        else:
            content = prompt

        data = {'messages': [{'role': 'user', 'content': content}]}
        data.update(sampling_params)

        return data

    def _get_json_data_for_regular_sambastudio(
        self, url: str, prompt: str, sampling_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Change params whether model is Bundle or not
        if 'Bundle' in self.request_config.model:
            sampling_params['select_expert'] = self.request_config.model.split('/')[-1]
            sampling_params['process_prompt'] = False
            sampling_params['top_k'] = 1

        # build payload for api v2
        if '/api/v2' in url.lower().strip():
            # if an image is provided, add it to the payload
            tuning_params = json.loads(json.dumps(sampling_params))
            data = {'items': [{'id': 'item1', 'value': prompt}], 'params': tuning_params}
        # support to build payload for api v1
        else:
            extended_sampling_params = {
                k: {'type': type(v).__name__, 'value': str(v)} for k, v in (sampling_params.items())
            }
            extended_sampling_params_str = json.dumps(extended_sampling_params)

            # Change request body whether API call is streaming or not
            if self.request_config.is_stream_mode:
                data = {'instance': prompt, 'params': json.loads(extended_sampling_params_str)}
            else:
                data = {'instances': [prompt], 'params': json.loads(extended_sampling_params_str)}

        return data

    def compute_metrics(self, metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Computes metrics for SambaStudio API endpoint

        Args:
            metrics (dict): basic metrics dictionary

        Raises:
            ValueError: raises when streaming is not selected

        Returns:
            tuple[dict, str]: tuple containing the metrics structure with server and client side values, and the
            complete generated text
        """

        # Get API request components
        url = self._get_url()
        headers = self._get_headers()
        json_data = self._get_json_data(url)

        # Start measuring time
        metrics[common_metrics.REQ_START_TIME] = datetime.now().strftime('%H:%M:%S.%f')
        start_time = time.monotonic()

        if self.request_config.is_stream_mode:
            with requests.post(
                url, headers=headers, json=json_data, stream=self.request_config.is_stream_mode
            ) as response:
                if response.status_code != 200:
                    error_details = response.json().get('error', 'No additional error details provided.')
                    raise Exception(f'Error: {response.status_code}, Details: {error_details}')

                if 'chat/completions' in self.base_url:  # SambaStudio compatible with OpenAI data payload
                    chunks_received, chunks_timings, response_dict, generated_text = (
                        self._parse_sambastudio_openai_compatible_response(response, start_time)
                    )
                else:  # Regular SambaStudio data payload
                    chunks_received, chunks_timings, response_dict, generated_text = (
                        self._parse_regular_sambastudio_response(response, start_time, url)
                    )
        else:
            # TODO: support non-streaming mode
            raise ValueError('Streaming mode required')

        # End measuring time
        metrics[common_metrics.REQ_END_TIME] = datetime.now().strftime('%H:%M:%S.%f')
        total_request_time = time.monotonic() - start_time
        ttft = self._calculate_ttft_from_streams(chunks_received, chunks_timings, total_request_time)

        # Populate server and client metrics
        prompt_len = self.request_config.prompt_tuple[1]
        number_chunks_recieved = len(chunks_received)

        num_output_tokens = self._get_token_length(generated_text)
        server_metrics = self._populate_server_metrics(response_dict, metrics)
        metrics = self._populate_client_metrics(
            prompt_len,
            num_output_tokens,
            ttft,
            total_request_time,
            server_metrics,
            number_chunks_recieved,
        )

        return metrics, generated_text

    def _parse_sambastudio_openai_compatible_response(
        self, response: Response, event_start_time: float
    ) -> Tuple[List[Any], List[Any], Dict[str, Any], str]:
        # Set variables
        generated_text = ''
        events_received = []
        events_timings = []

        client = sseclient.SSEClient(response)  # type: ignore

        for event in client.events():
            try:
                # check streaming events before last stream returns DONE
                if event.data != '[DONE]':
                    data = json.loads(event.data)
                    # if events don't contain "usage" key, which only shows up in stream returning
                    # performance metrics
                    if data.get('usage') is None:
                        # if streams still don't hit a finish reason
                        if data['choices'][0].get('finish_reason') is None:
                            if data['choices'][0]['delta'].get('content') is not None:
                                # log s timings
                                events_timings.append(time.monotonic() - event_start_time)
                                event_start_time = time.monotonic()
                                # concatenate streaming text pieces
                                stream_content = data['choices'][0]['delta']['content']
                                events_received.append(stream_content)
                                generated_text += stream_content
                    # process streaming chunk when performance usage is provided
                    else:
                        response_dict = data['usage']
            except Exception as e:
                raise Exception(f'Error: {e} at streamed event: {event.data}')
        return events_received, events_timings, response_dict, generated_text

    def _parse_regular_sambastudio_response(
        self, response: Response, chunk_start_time: float, url: str
    ) -> Tuple[List[Any], List[Any], Dict[str, Any], str]:
        # Set variables
        generated_text = ''
        chunks_received = []
        chunks_timings = []

        # fetch generated text and metrics for api v2
        if '/api/v2' in url.lower().strip():
            for chunk_orig in response.iter_lines(chunk_size=None):
                chunk = chunk_orig.strip()
                data = json.loads(chunk)

                completion = data['result']['items'][0]['value']['is_last_response']
                chunks_timings.append(time.monotonic() - chunk_start_time)
                chunk_start_time = time.monotonic()
                if completion is False:
                    chunks_received.append(data['result']['items'][0]['value']['stream_token'])
                    continue
                else:
                    generated_text = data['result']['items'][0]['value']['completion']
                    response_dict = data['result']['items'][0]['value']
                break
        # support to fetch generated text and metrics for api v1
        else:
            for chunk_orig in response.iter_lines(chunk_size=None):
                chunk = chunk_orig.strip()
                data = json.loads(chunk)

                completion = data['result']['responses'][0]['is_last_response']
                chunks_timings.append(time.monotonic() - chunk_start_time)
                chunk_start_time = time.monotonic()
                if completion is False:
                    chunks_received.append(data['result']['responses'][0]['stream_token'])
                    continue
                else:
                    generated_text = data['result']['responses'][0]['completion']
                    response_dict = data['result']['responses'][0]
                    break
        return chunks_received, chunks_timings, response_dict, generated_text


class SambaNovaCloudAPI(BaseAPIEndpoint):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Load sambanova cloud env variables
        if self.request_config.api_variables:
            self.base_url = (
                self.request_config.api_variables['SAMBANOVA_URL']
                if self.request_config.api_variables['SAMBANOVA_URL']
                else SAMBANOVA_URL
            )
            self.api_key = self.request_config.api_variables['SAMBANOVA_API_KEY']
        else:
            self.base_url = os.environ.get('SAMBANOVA_URL', SAMBANOVA_URL)
            self.api_key = os.environ.get('SAMBANOVA_API_KEY', '')

    def _get_url(self) -> str:
        """Builds url for API call

        Returns:
            str: url needed for API
        """
        return self.base_url

    def _get_headers(self) -> Dict[str, str]:
        """Gets headers for API call"""

        header = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        if self.request_config.use_debugging_mode:
            header['ss-sn-options'] = 'accuracy_debug'

        return header

    def _get_json_data(self) -> Dict[str, Any]:
        """Gets json body for API call

        Returns:
            dict: API call body
        """

        prompt = self.request_config.prompt_tuple[0]['template']
        sampling_params = self.request_config.sampling_params
        assert isinstance(sampling_params, dict), f'sampling_params must be a dict. Got type {type(sampling_params)}'
        sampling_params['model'] = self.request_config.model
        sampling_params['max_tokens'] = sampling_params.pop('max_tokens_to_generate')
        sampling_params['ignore_eos'] = True

        if self.request_config.is_stream_mode:
            sampling_params['stream'] = True
            sampling_params['stream_options'] = {'include_usage': True}
        else:
            # TODO: support not streaming mode
            raise ValueError('Streaming mode required')

        # If an image is provided, add it to the content
        content: Any = None
        if self.request_config.image:
            content = [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{self.request_config.image}'}},
            ]
        else:
            content = prompt

        data = {'messages': [{'role': 'user', 'content': content}]}
        data.update(sampling_params)
        return data

    def compute_metrics(self, metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Computes metrics for SambaNovaCloud endpoint

        Args:
            metrics (dict): basic metrics dictionary

        Returns:
            tuple[dict, str]: tuple containing the metrics structure with server and client side values, and the
            complete generated text
        """

        # Get API request components
        url = self._get_url()
        headers = self._get_headers()
        json_data = self._get_json_data()

        # Set variables
        generated_text = ''
        events_received = []
        events_timings = []

        # Start measuring time
        metrics[common_metrics.REQ_START_TIME] = datetime.now().strftime('%H:%M:%S.%f')
        start_time = event_start_time = time.monotonic()

        with requests.post(url, headers=headers, json=json_data, stream=self.request_config.is_stream_mode) as response:
            # print(f'Response content: {response.content}')
            if response.status_code != 200:
                response.raise_for_status()
            client = sseclient.SSEClient(response)  # type: ignore
            generated_text = ''

            for event in client.events():
                try:
                    # check streaming events before last stream returns DONE
                    if event.data != '[DONE]':
                        data = json.loads(event.data)
                        # if events don't contain "usage" key, which only shows up in stream returning
                        # performance metrics
                        if data.get('usage') is None:
                            # if streams still don't hit a finish reason
                            if data['choices'][0].get('finish_reason') is None:
                                # if data['choices'][0]['delta'].get('content') is not None:
                                #     # log s timings
                                #     events_timings.append(time.monotonic() - event_start_time)
                                #     event_start_time = time.monotonic()
                                #     # concatenate streaming text pieces
                                #     stream_content = data['choices'][0]['delta']['content']
                                #     events_received.append(stream_content)
                                #     generated_text += stream_content
                                
                                if (data['choices'][0]['delta'].get('content') is not None) \
                                    or (data['choices'][0]['delta'].get('reasoning') is not None):
                                    # log s timings
                                    events_timings.append(time.monotonic() - event_start_time)
                                    event_start_time = time.monotonic()
                                    # concatenate streaming text pieces
                                    if data['choices'][0]['delta'].get('content') is not None:
                                        stream_content = data['choices'][0]['delta']['content']
                                    elif data['choices'][0]['delta'].get('reasoning') is not None:
                                        stream_content = data['choices'][0]['delta']['reasoning']
                                    events_received.append(stream_content)
                                    generated_text += stream_content
                        # process streaming chunk when performance usage is provided
                        else:
                            response_dict = data['usage']
                except Exception as e:
                    raise Exception(f'Error: {e} at streamed event: {event.data}')

        # End measuring time
        metrics[common_metrics.REQ_END_TIME] = datetime.now().strftime('%H:%M:%S.%f')
        total_request_time = time.monotonic() - start_time
        # for event_received, event_timing in zip(events_received, events_timings):
        #     print(f'took time: {event_timing} - received event: {event_received}')
        # print(f'Total request time: {total_request_time}')
        # print(f'server response dict: {response_dict}')
        ttft = self._calculate_ttft_from_streams(events_received, events_timings, total_request_time)

        # Populate server and client metrics
        prompt_len = self.request_config.prompt_tuple[1]
        number_chunks_recieved = len(events_received)

        num_output_tokens = self._get_token_length(generated_text)
        server_metrics = self._populate_server_metrics(response_dict, metrics)
        metrics = self._populate_client_metrics(
            prompt_len,
            num_output_tokens,
            ttft,
            total_request_time,
            server_metrics,
            number_chunks_recieved,
        )

        return metrics, generated_text


def llm_request(request_config: RequestConfig, tokenizer: AutoTokenizer) -> Tuple[Dict[str, Any], str, RequestConfig]:
    """Makes a single completion request to a LLM API

    Args:
        request_config (RequestConfig): config options including user's prompt and LLM parameters
        tokenizer (AutoTokenizer): tokenizer for counting tokens

    Returns:
        tuple: Metrics about the performance charateristics of the request.
        The text generated by the request to the LLM API.
        The request_config used to make the request. This is mainly for logging purposes.
    """

    generated_text = ''
    metrics: Dict[str, Any] = {}
    metrics[common_metrics.ERROR_CODE] = None
    metrics[common_metrics.ERROR_MSG] = ''
    metrics[common_metrics.PROMPT_NAME] = request_config.prompt_tuple[0]['name']

    try:
        if request_config.llm_api == 'sncloud':
            sncloud_client = SambaNovaCloudAPI(request_config, tokenizer)
            metrics, generated_text = sncloud_client.compute_metrics(metrics)

        elif request_config.llm_api == 'sambastudio':
            sambastudio_client = SambaStudioAPI(request_config, tokenizer)
            metrics, generated_text = sambastudio_client.compute_metrics(metrics)

        else:
            raise ValueError(f'llm_api parameter with value {request_config.llm_api} is not valid.')

        return metrics, generated_text, request_config

    except Exception as e:
        error_code = getattr(
            e,
            'code',
            """Error while running LLM API requests. """
            + """Check your model name, LLM API type, env variables and endpoint status.""",
        )
        error_message = str(e)
        metrics[common_metrics.ERROR_MSG] = error_message
        metrics[common_metrics.ERROR_CODE] = error_code

        return metrics, '', request_config


if __name__ == '__main__':
    # The call of this python file is more for debugging purposes

    # load env variables
    load_dotenv('../.env', override=True)
    env_vars = dict(os.environ)

    model = 'Meta-Llama-3.3-70B-Instruct'
    llm_api = 'sncloud'
    tokenizer = get_tokenizer(model)

    prompt_text = 'This is a test example, so tell me about anything'
    prompt = {'name': 'test', 'template': prompt_text}

    request_config = RequestConfig(
        request_idx=1,
        prompt_tuple=(prompt, 10),
        model=model,
        llm_api=llm_api,
        sampling_params={
            # "do_sample": False,
            'max_tokens_to_generate': 250,
            # "top_k": 40,
            # "top_p": 0.95,
            # "process_prompt": "False",
        },
        is_stream_mode=True,
        num_concurrent_requests=1,
    )

    metrics, generated_text, request_config = llm_request(request_config, tokenizer)

    print(f'Metrics collected: {metrics}')
    print(f'Request config: {request_config}')
