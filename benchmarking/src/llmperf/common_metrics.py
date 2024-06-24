# General metrics
ERROR_MSG = "error_msg"
ERROR_CODE = "error_code"
ERROR_CODE_FREQ = "error_code_frequency"
NUM_ERRORS = "number_errors"
NUM_COMPLETED_REQUESTS = "num_completed_requests"
COMPLETED_REQUESTS_PER_MIN = "num_completed_requests_per_min"
ERROR_RATE = "error_rate"
NUM_REQ_STARTED = "num_requests_started"
REQ_START_TIME = "start_time"
REQ_END_TIME = "end_time"
BATCH_SIZE_USED = "batch_size_used"

# Client-side metrics
TTFT = "ttft_s"
E2E_LAT = "end_to_end_latency_s"
NUM_INPUT_TOKENS = "number_input_tokens"
NUM_OUTPUT_TOKENS = "number_output_tokens"
NUM_TOTAL_TOKENS = "number_total_tokens"
REQ_OUTPUT_THROUGHPUT = "request_output_throughput_token_per_s"
OUTPUT_THROUGHPUT = "mean_output_throughput_token_per_s"
TOTAL_TOKEN_THROUGHPUT = "total_tokens_per_sec_s"

# Server-side metrics
TTFT_SERVER = "ttft_server_s"
E2E_LAT_SERVER = "end_to_end_latency_server_s"
REQ_OUTPUT_THROUGHPUT_SERVER = "request_output_throughput_server_token_per_s"
REQ_OUTPUT_THROUGHPUT_AFTER_FIRST_SERVER = (
    "request_output_throughput_after_first_server_token_per_s"
)
NUM_OUTPUT_TOKENS_SERVER = "number_output_tokens_server"
NUM_INPUT_TOKENS_SERVER = "number_input_tokens_server"
NUM_TOTAL_TOKENS_SERVER = "number_total_tokens_server"
TOTAL_TOKEN_THROUGHPUT_SERVER = "total_tokens_per_sec_server"
