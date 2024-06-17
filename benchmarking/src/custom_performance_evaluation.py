import argparse
import csv
import datetime
import json
import numpy as np
import os
import re
import requests
import time
from transformers import AutoTokenizer
import tqdm
import pytz

from concurrent.futures import ThreadPoolExecutor


class CustomPerformanceEvaluator:
    def __init__(
            self, 
            input_file_path: str, 
            service_url: str, 
            project_id: str, 
            endpoint_id: str, 
            endpoint_api_key:str, 
            max_tokens:int, 
        ):
        """Initialize the CustomPerformanceEvaluator with necessary parameters.

        Args:
            input_file_path (str): Path to the input file.
            service_url (str): Full URL for endpoint.
            project_id (str): Project ID from Studio.
            endpoint_id (str): Endpoint ID from Studio.
            endpoint_api_key (str): API key for the endpoint.
            max_tokens (int): Maximum number of tokens to generate.

        Returns:
            None
        """
        self.endpoint_url = f"{service_url}/api/predict/nlp/stream/{project_id}/{endpoint_id}"
        self.api_key = endpoint_api_key
        self.input_prompts = self.get_input_prompts(input_file_path)
        self.max_tokens = max_tokens
        self.outputs = []
        self.request_timestamps = []
        self.job_uid = self.get_job_uid()

    @staticmethod
    def get_input_prompts(input_path: str):
        """ Method for getting a list of input prompts from a variety of sources. This function may need to be augmented any time 
        there is a new file format or structure that the input data is coming in from.

        Args:
            input_path (str): Path to file containing input prompts.
        
        Returns:
            List[str]: List of input prompts to send to endpoint for generation. 
        """
        data = []
        
        # Extract data from JSON file
        if input_path.endswith('.json'):
            with open(input_path, 'r') as f:
                json_data = json.load(f)
                data = json_data["inputs"]
        
        # Extract data from CSV file
        elif input_path.endswith('.csv'):
            with open(input_path, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    data.extend(row)
        
        # Add more conditions for other file formats below sas needed
        
        return data

    def extract_response_data(self, response_text):
        """This is a utility function for extracting json data from the response text provided from making a request a SambaStudio 
        streaming endpoint.

        Args:
            response_text (str): String response from streaming api endpoint

        Returns:
            Tuple(str, float): Tuple containing the extracted llm generation and the corresponding total tokens generated
        """
        try:
            output_generation = ""
            total_generated_tokens = 0

            # Extract "end_event" if it exists
            end_event = re.search(
                r"end_event\r\ndata: (.*)",
                response_text,
                flags=re.DOTALL
            )

            if end_event:
                event_dict = json.loads(end_event.group(1))
                output_generation = event_dict["completion"]
                tokens_list = self.tokenizer.tokenize(output_generation)
                total_generated_tokens = len(tokens_list)
            
            # If "end_event" isn't present, concatenate all streamed tokens and extract total count
            else:
                all_events = re.findall(r"data: (.*})", response_text, flags=re.I|re.M)

                # List to fill with streamed tokens
                stream_generation = []
                for index, stream_event in enumerate(all_events):
                    event_dict = json.loads(stream_event)
                    stream_generation.append(event_dict['stream_token'])

                # Join all streamed tokens for full generation
                output_generation = "".join(stream_generation)

                tokens_list = self.tokenizer.tokenize(output_generation)
                total_generated_tokens = len(tokens_list)

            return (output_generation, total_generated_tokens)
        except Exception as e:
            return ("ERROR IN DATA EXTRACTION", 0)

    def get_job_uid(self):
        """Generates a unique identifier for a job based on the number of input prompts and current timestamp. 
        The job UID is a string that combines the number of input prompts (rounded to the nearest thousand) and 
        the current timestamp in the US/Pacific timezone. This UID is intended to be used for marking the output 
        directory/files.

        Returns:
            str: The generated job UID.
        """
        # Input size indicator calculation
        input_size = len(self.input_prompts)
        if input_size > 1000:
            input_size = f"{input_size//1000}"
        else:
            input_size = f"{input_size/1000}"

        # Job timestamp calculation
        date_format='%m-%d-%Y_%I:%M-%p'
        timestamp = datetime.datetime.now(tz=pytz.utc).astimezone(pytz.timezone('US/Pacific')).strftime(date_format)

        return f"{input_size}k_{self.max_tokens}tk_{timestamp}"

    def calculate_and_write_results(self):
        """Utility function for writing out data to output jsonl file"""
        token_counts = []

        # Create unique job output directory
        out_dir = f"outputs/out_{self.job_uid}"
        os.makedirs(out_dir, exist_ok=True)

        # Write out result object for each llm response
        with open(f"{out_dir}/results.jsonl", "w") as results_outfile:
            for result_obj in self.outputs:
                token_counts.append(result_obj["token_count"])
                json.dump(result_obj, results_outfile)
                results_outfile.write("\n")

        # Calculate processing time, avg tokens generated, and tokens/sec
        total_processing_time = datetime.timedelta(seconds=self.end_time-self.start_time)
        max_token_count = max(token_counts)
        min_token_count = min(token_counts)
        token_count_std = np.std(token_counts)
        percentile_25 = np.percentile(token_counts, 25)
        percentile_75 = np.percentile(token_counts, 75)
        avg_token_count = sum(token_counts) / len(token_counts)
        tokens_per_second = (len(self.outputs) * avg_token_count) / total_processing_time.total_seconds()


        # Write out speed metrics to performance file
        with open(f"{out_dir}/performance.txt", "w") as perf_outfile:
            perf_outfile.write(f"""
            INPUT PARAMS:
                Dataset Size: {len(self.input_prompts)}
                Max Tokens To Generate: {self.max_tokens}
                
            GENERATED TOKEN COUNT METADATA: 
                Mean: {avg_token_count}
                Median: {avg_token_count}
                STD: {token_count_std}
                Max: {max_token_count}
                25%: {percentile_25}
                75%: {percentile_75}
                Min: {min_token_count}

            RESULTS:
                Total Processing Time: {total_processing_time}
                Tokens per second: {tokens_per_second}
                """)
        
        with open(f"{out_dir}/timestamps.txt", "w") as timestamps_outfile: 
            for timestamp in self.request_timestamps:
                timestamps_outfile.write(timestamp)
                timestamps_outfile.write("\n")

    def endpoint_request(self, prompt_id: int):
        """Sends a request to the SambaStudio endpoint with the given prompt ID.
        
        Args:
            prompt_id (int): The ID of the prompt to be sent to the endpoint.
        
        Returns:
            None

        Raises:
            Exception: If the request fails due to a server-side error.

        Notes:
            This function sends a POST request to the SambaStudio endpoint with the given prompt ID.
            It includes the necessary headers and data in the request, including the prompt, generation parameters, and API key.
            The function extracts the generation and token count from the response and constructs an output object.
            If the request fails, the function returns an output object with an error message.
        """
        try:
            prompt = self.input_prompts[prompt_id]

            headers = {
                'Content-Type': 'application/json',
                'key': self.api_key
            }

            data = {
                "inputs": [prompt],
                "params": {
                    "do_sample": {"type": "bool", "value": "true"},
                    "max_tokens_to_generate": {"type": "int", "value": f"{self.max_tokens}"},
                    "repetition_penalty": {"type": "float", "value": "1.5"},
                    "temperature": {"type": "float", "value": "0.1"},
                    "top_k": {"type": "int", "value": "15"},
                    "top_logprobs": {"type": "int", "value": "0"},
                    "top_p": {"type": "float", "value": "0.75"}
                }
            }

            print(f"STARTING REQUEST: {prompt_id}")
            
            # Request to SambaStudio Endpoint
            response = requests.post(self.endpoint_url, headers=headers, json=data)

            # Extract generation and token count from response
            generation, token_count = self.extract_response_data(response.text)

            # Construct output object
            out_json = {"prompt": prompt, "generation": generation, "token_count": token_count}

            # Record timestamp of request completion
            date_format='%m/%d/%Y %H:%M:%S %Z'
            date = datetime.datetime.now(tz=pytz.utc).astimezone(pytz.timezone('US/Pacific'))
            self.request_timestamps.append(f"REQUEST {prompt_id} FINISHED AT: {date.strftime(date_format)}")

            # Add to list of output objects to be written out after all requests have been completed
            self.outputs.append(out_json)
        except Exception as e:
            # Generally see this when the request fails from a server side error
            out_json = {"prompt": prompt, "generation": f"REQUEST ERROR: {e}", "token_count": 0}
            self.outputs.append(out_json)

    def run_rdu_perf_test(self):
        self.start_time = time.time()

        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(self.endpoint_request, prompt_id) for prompt_id in range(len(self.input_prompts))]

            for future in futures:
                future.result()

        self.end_time = time.time()
        
        # Calculate and write out generation and performance results
        self.calculate_and_write_results()

def main():
    parser = argparse.ArgumentParser(description='RDU Inference Tester')
    parser.add_argument('--input_file', type=str, required=True, help='Input file containing prompts to evaluate on!')
    parser.add_argument('--service_url', type=str, required=True, help='SambaStudio URL')
    parser.add_argument('--project_id', type=str, required=True, help='Project ID')
    parser.add_argument('--endpoint_id', type=str, required=True, help='Endpoint ID')
    parser.add_argument('--endpoint_api_key', type=str, required=True, help='Endpoint API Key')
    parser.add_argument('--max_tokens', type=int, required=False, default=256, help='Maximum number of tokens to generate')
    parser.add_argument('--tokenizer', type=str, required=False, default="meta-llama/Llama-2-7b-hf", help="Tokenizer to use for token generation counting")

    args = parser.parse_args()

    input_file = args.input_file
    service_url = args.service_url
    project_id = args.project_id
    endpoint_id = args.endpoint_id
    endpoint_api_key = args.endpoint_api_key
    max_tokens = args.max_tokens
    tokenizer = args.tokenizer

    rdu_inference_tester = RDUThroughputTester(
        input_file, 
        service_url, 
        project_id, 
        endpoint_id, 
        endpoint_api_key, 
        max_tokens, 
        tokenizer
        )
    rdu_inference_tester.run_rdu_perf_test()

if __name__ == '__main__':
    main()