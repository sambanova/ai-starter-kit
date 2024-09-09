import os
import abc
import sys
import json
import time
import requests
import sseclient
from math import isclose
from datetime import datetime

sys.path.append("./src")
sys.path.append("./src/llmperf")

from transformers import AutoTokenizer
from llmperf.models import RequestConfig
from llmperf import common_metrics
from utils import get_tokenizer

from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

class BaseAPIEndpoint(abc.ABC):
    def __init__(
        self,
        request_config: RequestConfig,
        tokenizer: AutoTokenizer
    ):
        self.request_config = request_config
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def _get_url(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def _get_headers(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def _get_json_data(self, *args, **kwargs):
        pass
    
    def _get_token_length(self, input_text: str) -> int:
        """Gets the token length of a piece of text

        Args:
            input_text (str): input text
            
        Returns:
            int: number of tokens
        """
        return len(self.tokenizer.encode(input_text))
    
    
    def _calculate_tpot_from_streams_after_first(self, chunks_received: list, chunks_timings: list) -> float:
        """Calculates Time per Output Token (TPOT) based on the streaming events coming after the first one.
        In general, the way to calculate this metric is: time_to_generate_tokens/number_of_tokens_generated

        Args:
            chunks_received (list): complete list of events coming from streaming response
            chunks_timings (list): complete list of timings that each event took to process

        Returns:
            float: calculated tpot 
        """
        
        # Calculate tokens
        total_tokens_received_after_first_chunk = sum(
            self._get_token_length(c) for c in chunks_received[1:]
        )
        
        # Calculate time
        total_time_to_receive_tokens_after_first_chunk = sum(chunks_timings[1:])
        
        # Calculate tpot
        tpot = (
            total_time_to_receive_tokens_after_first_chunk
            / total_tokens_received_after_first_chunk
        )
        
        return tpot
    
    def _calculate_ttft_from_streams(self, chunks_received: list, chunks_timings: list, total_request_time: int) -> float:
        """Calculates Time to First Token (TTFT) based on the streaming events coming from the response.
        If there are enough streaming events, the formula to calculate ttft is: time_first_chunk - (tokens_first_chunk - 1) * tpot

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
        ttft: int,
        total_request_time: int,
        server_metrics: dict,
        number_chunks_recieved: int,
    ) -> dict:
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
            
        metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT] = (prompt_len + num_output_tokens)/total_request_time
        
        return metrics

    def _populate_server_metrics(self, response_dict: dict, metrics: dict) -> dict:
        """Parse output data to metrics dictionary structure

        Args:
            response_dict (dict): dict data with performance metrics
            metrics (dict): metrics dictionary

        Returns:
            dict: updated metrics dictionary
        """

        metrics[common_metrics.NUM_INPUT_TOKENS_SERVER] = response_dict.get(
            "prompt_tokens_count"
        ) or response_dict.get("prompt_tokens")
        
        metrics[common_metrics.NUM_OUTPUT_TOKENS_SERVER] = response_dict.get(
            "completion_tokens_count"
        ) or response_dict.get("completion_tokens")
        
        metrics[common_metrics.NUM_TOTAL_TOKENS_SERVER] = response_dict.get(
            "total_tokens_count"
        ) or response_dict.get("total_tokens")
        ttft_server = response_dict.get("time_to_first_token") or response_dict.get(
            "time_to_first_response"
        )
        
        metrics[common_metrics.TTFT_SERVER] = ttft_server
        
        metrics[common_metrics.E2E_LAT_SERVER] = response_dict.get(
            "total_latency"
        ) or response_dict.get("model_execution_time")
        
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT_SERVER] = (
            response_dict.get("completion_tokens_after_first_per_sec")
            or response_dict.get("completion_tokens_per_sec_after_first_response")
            or response_dict.get("throughput_after_first_token")
        )
        
        metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT_SERVER] = response_dict.get("total_tokens_per_sec") 
        if (metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT_SERVER] is None) and (metrics[common_metrics.E2E_LAT_SERVER] is not None):
            metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT_SERVER] = metrics[common_metrics.NUM_TOTAL_TOKENS_SERVER] / (metrics[common_metrics.E2E_LAT_SERVER])
            
        metrics[common_metrics.BATCH_SIZE_USED] = response_dict.get("batch_size_used")
        
        return metrics

class SambaStudioAPI(BaseAPIEndpoint):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load sambastudio env variables
        self.base_url = os.environ.get("SAMBASTUDIO_BASE_URL")
        self.base_uri = os.environ.get("SAMBASTUDIO_BASE_URI")
        self.project_id = os.environ.get("SAMBASTUDIO_PROJECT_ID")
        self.endpoint_id = os.environ.get("SAMBASTUDIO_ENDPOINT_ID")
        self.api_key = os.environ.get("SAMBASTUDIO_API_KEY")
        
        
    def _get_url(self) -> str:
        """Builds url for API call

        Returns:
            str: url needed for API
        """

        if self.request_config.is_stream_mode:
            path = f"{self.base_uri}/stream/{self.project_id}/{self.endpoint_id}"
        else:
            path = f"{self.base_uri}/{self.project_id}/{self.endpoint_id}"
        
        url = f"{self.base_url}/{path}"
        
        return url
    
    def _get_headers(self) -> None:
        """ Gets headers for API call """
        return {"key": self.api_key}
    
    def _get_json_data(self, url: str) -> dict:
        """Gets json body for API call

        Args:
            url: URL being used for the API call
            
        Returns:
            dict: API call body according to COE and streaming conditions
        """
        prompt = self.request_config.prompt_tuple[0]
        sampling_params = self.request_config.sampling_params

        # Change params whether model is COE or not
        if "COE" in self.request_config.model:
            sampling_params["select_expert"] = self.request_config.model.split("/")[-1]
            sampling_params["process_prompt"] = False
    
        # build payload for api v2
        if "/api/v2" in url.lower().strip():  
            tuning_params = json.loads(json.dumps(sampling_params))
            data = {"items": [{"id":"item1", "value": prompt}], "params": tuning_params}
        # support to build payload for api v1 
        else: 
            extended_sampling_params = {
                k: {"type": type(v).__name__, "value": str(v)}
                for k, v in (sampling_params.items())
            }
            extended_sampling_params = json.dumps(extended_sampling_params)
            
            # Change request body whether API call is streaming or not
            if self.request_config.is_stream_mode:
                data = {"instance": prompt, "params": json.loads(extended_sampling_params)}
            else:
                data = {"instances": [prompt], "params": json.loads(extended_sampling_params)}
            
        return data
    
    
    def compute_metrics(self, metrics: dict) -> tuple[dict, str]:
        """Computes metrics for SambaStudio API endpoint

        Args:
            metrics (dict): basic metrics dictionary

        Raises:
            ValueError: raises when streaming is not selected

        Returns:
            tuple[dict, str]: tuple containing the metrics structure with server and client side values, and the complete generated text
        """
        
        # Get API request components
        url = self._get_url()
        headers = self._get_headers()
        json_data = self._get_json_data(url)
        
        # Set variables
        generated_text = ""
        chunks_received = []
        chunks_timings = []
        
        # Start measuring time
        metrics[common_metrics.REQ_START_TIME] = datetime.now().strftime("%H:%M:%S")
        start_time = chunk_start_time = time.monotonic()

        if self.request_config.is_stream_mode:

            with requests.post(
                url, headers=headers, json=json_data, stream=self.request_config.is_stream_mode
            ) as response:
                if response.status_code != 200:
                    response.raise_for_status()
                    
                # fetch generated text and metrics for api v2
                if "/api/v2" in url.lower().strip():
                    
                    for chunk_orig in response.iter_lines(chunk_size=None):
                        chunk = chunk_orig.strip()
                        data = json.loads(chunk)
                    
                        completion = data["result"]["items"][0]["value"]["is_last_response"]
                        chunks_timings.append(time.monotonic() - chunk_start_time)
                        chunk_start_time = time.monotonic()
                        if completion is False:
                            chunks_received.append(data["result"]["items"][0]["value"]["stream_token"])
                            continue
                        else:
                            generated_text = data["result"]["items"][0]["value"]["completion"]
                            response_dict = data["result"]["items"][0]["value"]
                        break
                # support to fetch generated text and metrics for api v1
                else:
                
                    for chunk_orig in response.iter_lines(chunk_size=None):
                        chunk = chunk_orig.strip()
                        data = json.loads(chunk)

                        completion = data["result"]["responses"][0]["is_last_response"]
                        chunks_timings.append(time.monotonic() - chunk_start_time)
                        chunk_start_time = time.monotonic()
                        if completion is False:
                            chunks_received.append(data["result"]["responses"][0]["stream_token"])
                            continue
                        else:
                            generated_text = data["result"]["responses"][0]["completion"]
                            response_dict = data["result"]["responses"][0]
                            break
        else:
            # TODO: support non-streaming mode
            raise ValueError("Streaming mode required")
        
        # End measuring time
        metrics[common_metrics.REQ_END_TIME] = datetime.now().strftime("%H:%M:%S")  
        total_request_time = time.monotonic() - start_time
        ttft = self._calculate_ttft_from_streams(chunks_received, chunks_timings, total_request_time)
    
        # Populate server and client metrics
        prompt_len  = self.request_config.prompt_tuple[1]
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
            
class FastAPI(BaseAPIEndpoint):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load sambastudio env variables
        self.base_url = os.environ.get("FASTAPI_URL")
        self.api_key = os.environ.get("FASTAPI_API_KEY")
        
    def _get_url(self) -> str:
        """Builds url for API call

        Returns:
            str: url needed for API
        """
        return self.base_url
    
    def _get_headers(self):
        """ Gets headers for API call """
        return {"Authorization": f"Bearer {self.api_key}", 'Content-Type': 'application/json'} 
    
    def _get_json_data(self) -> dict:
        """Gets json body for API call

        Returns:
            dict: API call body 
        """
        
        prompt = self.request_config.prompt_tuple[0]
        sampling_params = self.request_config.sampling_params
        sampling_params["model"] = self.request_config.model
        
        if self.request_config.is_stream_mode:
            sampling_params["stream"] = "true"
            sampling_params["stream_options"] = {"include_usage": "true"}
        else:
            # TODO: support not streaming mode
            raise ValueError("Streaming mode required")
                    
        data = {"messages": [{"role": "user", "content": prompt}]}
        data.update(sampling_params)
        
        return data

    def compute_metrics(self, metrics: dict) -> tuple[dict, str]:
        """Computes metrics for FastAPI API endpoint

        Args:
            metrics (dict): basic metrics dictionary

        Returns:
            tuple[dict, str]: tuple containing the metrics structure with server and client side values, and the complete generated text
        """
        
        # Get API request components
        url = self._get_url()
        headers = self._get_headers()
        json_data = self._get_json_data()
        
        # Set variables
        generated_text = ""
        events_received = []
        events_timings = []
        
        # Start measuring time
        metrics[common_metrics.REQ_START_TIME] = datetime.now().strftime("%H:%M:%S")
        start_time = event_start_time = time.monotonic()


        with requests.post(
            url, headers=headers, json=json_data, stream=self.request_config.is_stream_mode
        ) as response:
            
            if response.status_code != 200:
                response.raise_for_status()

            client = sseclient.SSEClient(response)
            generated_text = ""
            
            for event in client.events():                        
                try:    
                    # check streaming events before last stream returns DONE
                    if event.data != "[DONE]" :
                        data = json.loads(event.data)
                        # if events don't contain "usage" key, which only shows up in stream returning performance metrics
                        if data.get("usage") is None:
                            # if streams still don't hit a finish reason
                            if data['choices'][0]["finish_reason"] is None:
                                # log s timings
                                events_timings.append(time.monotonic() - event_start_time)
                                event_start_time = time.monotonic()
                                # concatenate streaming text pieces
                                stream_content = data['choices'][0]["delta"]["content"]
                                events_received.append(stream_content)
                                generated_text += stream_content
                        # process streaming chunk when performance usage is provided
                        else: 
                            response_dict = data["usage"]
                except Exception as e:
                    raise Exception(f"Error: {e} at streamed event: {event.data}")
        
        # End measuring time
        metrics[common_metrics.REQ_END_TIME] = datetime.now().strftime("%H:%M:%S")  
        total_request_time = time.monotonic() - start_time
        ttft = self._calculate_ttft_from_streams(events_received, events_timings, total_request_time)
    
        # Populate server and client metrics
        prompt_len  = self.request_config.prompt_tuple[1]
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


class SambaNovaCloudAPI(BaseAPIEndpoint):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load sambanova cloud env variables
        self.base_url = os.environ.get("SAMBANOVA_URL")
        self.api_key = os.environ.get("SAMBANOVA_API_KEY")
        
    def _get_url(self) -> str:
        """Builds url for API call

        Returns:
            str: url needed for API
        """
        return self.base_url
    
    def _get_headers(self):
        """ Gets headers for API call """
        return {"Authorization": f"Bearer {self.api_key}", 'Content-Type': 'application/json'} 
    
    def _get_json_data(self) -> dict:
        """Gets json body for API call

        Returns:
            dict: API call body 
        """
        
        prompt = self.request_config.prompt_tuple[0]
        sampling_params = self.request_config.sampling_params
        sampling_params["model"] = self.request_config.model
        
        if self.request_config.is_stream_mode:
            sampling_params["stream"] = "true"
            sampling_params["stream_options"] = {"include_usage": "true"}
        else:
            # TODO: support not streaming mode
            raise ValueError("Streaming mode required")
                    
        data = {"messages": [{"role": "user", "content": prompt}]}
        data.update(sampling_params)
        
        return data

    def compute_metrics(self, metrics: dict) -> tuple[dict, str]:
        """Computes metrics for SambaNovaCloud endpoint

        Args:
            metrics (dict): basic metrics dictionary

        Returns:
            tuple[dict, str]: tuple containing the metrics structure with server and client side values, and the complete generated text
        """
        
        # Get API request components
        url = self._get_url()
        headers = self._get_headers()
        json_data = self._get_json_data()
        
        # Set variables
        generated_text = ""
        events_received = []
        events_timings = []
        
        # Start measuring time
        metrics[common_metrics.REQ_START_TIME] = datetime.now().strftime("%H:%M:%S")
        start_time = event_start_time = time.monotonic()


        with requests.post(
            url, headers=headers, json=json_data, stream=self.request_config.is_stream_mode
        ) as response:
            
            if response.status_code != 200:
                response.raise_for_status()

            client = sseclient.SSEClient(response)
            generated_text = ""
            
            for event in client.events():                        
                try:    
                    # check streaming events before last stream returns DONE
                    if event.data != "[DONE]" :
                        data = json.loads(event.data)
                        # if events don't contain "usage" key, which only shows up in stream returning performance metrics
                        if data.get("usage") is None:
                            # if streams still don't hit a finish reason
                            if data['choices'][0]["finish_reason"] is None:
                                # log s timings
                                events_timings.append(time.monotonic() - event_start_time)
                                event_start_time = time.monotonic()
                                # concatenate streaming text pieces
                                stream_content = data['choices'][0]["delta"]["content"]
                                events_received.append(stream_content)
                                generated_text += stream_content
                        # process streaming chunk when performance usage is provided
                        else: 
                            response_dict = data["usage"]
                except Exception as e:
                    raise Exception(f"Error: {e} at streamed event: {event.data}")
        
        # End measuring time
        metrics[common_metrics.REQ_END_TIME] = datetime.now().strftime("%H:%M:%S")  
        total_request_time = time.monotonic() - start_time
        ttft = self._calculate_ttft_from_streams(events_received, events_timings, total_request_time)
    
        # Populate server and client metrics
        prompt_len  = self.request_config.prompt_tuple[1]
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


def llm_request(request_config: RequestConfig, tokenizer: AutoTokenizer) -> tuple:
    """Makes a single completion request to a LLM API

    Args:
        request_config (RequestConfig): config options including user's prompt and LLM parameters
        tokenizer (AutoTokenizer): tokenizer for counting tokens

    Returns:
        tuple: Metrics about the performance charateristics of the request.
        The text generated by the request to the LLM API.
        The request_config used to make the request. This is mainly for logging purposes.
    """

    generated_text = ""
    metrics = {}
    metrics[common_metrics.ERROR_CODE] = None
    metrics[common_metrics.ERROR_MSG] = ""
    
    try:
       
        if request_config.llm_api == "sncloud":
            sncloud_client = SambaNovaCloudAPI(request_config, tokenizer)
            metrics, generated_text = sncloud_client.compute_metrics(metrics)
        
        elif request_config.llm_api == "sambastudio":
            sambastudio_client = SambaStudioAPI(request_config, tokenizer)
            metrics, generated_text = sambastudio_client.compute_metrics(metrics)
        
        else:
            raise ValueError(f"llm_api parameter with value {request_config.llm_api} is not valid.")
        
        return metrics, generated_text, request_config

    except Exception as e:  
         
        error_code = getattr(
            e,
            "code",
            "Error while running LLM API calls. Check your model name, LLM API type, env variables and endpoint status",
        )
        error_message = str(e)
        metrics[common_metrics.ERROR_MSG] = error_message
        metrics[common_metrics.ERROR_CODE] = error_code
        
        return metrics, "", request_config

if __name__ == "__main__":
    # The call of this python file is more for debugging purposes

    # load env variables
    load_dotenv("../.env", override=True)
    env_vars = dict(os.environ)

    model = "llama3-405b"
    llm_api = "sncloud"
    tokenizer = get_tokenizer(model)

    prompt = "This is a test example, so tell me about anything"
    request_config = RequestConfig(
        prompt_tuple=(prompt, 10),
        model=model,
        llm_api=llm_api,
        sampling_params={
            # "do_sample": False,
            "max_tokens_to_generate": 250,
            # "top_k": 40,
            # "top_p": 0.95,
            # "process_prompt": "False",
        },
        is_stream_mode=True,
        num_concurrent_workers=1,
    )

    metrics, generated_text, request_config = llm_request(request_config, tokenizer)

    print(f"Metrics collected: {metrics}")
    # print(f'Completion text: {generated_text}')
    print(f"Request config: {request_config}")