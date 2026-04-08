# General metrics
ERROR_MSG = 'error_msg'
ERROR_CODE = 'error_code'
ERROR_CODE_FREQ = 'error_code_frequency'
NUM_ERRORS = 'number_errors'
PROMPT_NAME = 'prompt_name'
NUM_COMPLETED_REQUESTS = 'num_completed_requests'
COMPLETED_REQUESTS_PER_MIN = 'num_completed_requests_per_min'
ERROR_RATE = 'error_rate'
NUM_REQ_STARTED = 'num_requests_started'
REQ_START_TIME = 'start_time'
REQ_END_TIME = 'end_time'
BATCH_SIZE_USED = 'batch_size_used'
QUEUE_TIME = 'queue_time'
ACCEPTANCE_RATE = 'acceptance_rate'

# Client-side metrics
TTFT = 'client_ttft_s'
E2E_LAT = 'client_end_to_end_latency_s'
REQ_OUTPUT_THROUGHPUT = 'client_output_token_per_s_per_request'
TOTAL_TOKEN_THROUGHPUT = 'client_total_tokens_per_s_per_request'
OUTPUT_THROUGHPUT = 'client_total_output_throughput'
NUM_INPUT_TOKENS = 'number_input_tokens'
NUM_OUTPUT_TOKENS = 'number_output_tokens'
NUM_TOTAL_TOKENS = 'number_total_tokens'

# Inter-token latency metrics
INTER_TOKEN_LATENCY = 'client_inter_token_latencies_s'  # List of ITLs per request
MEAN_INTER_TOKEN_LATENCY = 'client_mean_inter_token_latency_s'  # Mean ITL per request
MEAN_OUTPUT_THROUGHPUT = 'mean_output_throughput_token_per_s'  # Mean aggregate throughput

# Server-side metrics
TTFT_SERVER = 'server_ttft_s'
E2E_LAT_SERVER = 'server_end_to_end_latency_s'
REQ_OUTPUT_THROUGHPUT_SERVER = 'server_output_token_per_s_per_request'
REQ_OUTPUT_THROUGHPUT_SERVER_FIRST_TEN = 'server_output_token_after_first_per_s_first_ten_per_request'
TOTAL_TOKEN_THROUGHPUT_SERVER = 'server_total_tokens_per_s_per_request'
NUM_OUTPUT_TOKENS_SERVER = 'server_number_output_tokens'
NUM_INPUT_TOKENS_SERVER = 'server_number_input_tokens'
NUM_TOTAL_TOKENS_SERVER = 'server_number_total_tokens'
