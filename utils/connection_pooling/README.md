# Connection Pooling Test

This test measures the latency difference between requests that establish the first HTTPS connection and subsequent requests that reuse the same HTTPS connection (HTTP connection pooling).

## Overview

The test issues multiple requests to an API endpoint and measures:
- **TTFT (Time To First Token)**: Time from request start to receiving the first token
- **Queuing Time**: Time spent in the job queue
- **Total Response Time**: Complete time from request to completion

The first request typically takes longer due to connection establishment overhead, while subsequent requests benefit from connection pooling.

## Usage

### Command Line Interface

```bash
python3 test_connection_pooling.py --api-url <URL> --api-key <KEY> --model <MODEL>
```

#### Required Arguments
- `--api-url`: The API URL to test against
- `--api-key`: API key for authentication
- `--model`: Model name to use for testing

#### Optional Arguments
- `--test-duration`: Test duration in seconds (default: 50)
- `--sleep-duration`: Sleep duration between requests in seconds (default: 10)
- `--max-tokens`: Maximum tokens to generate (default: 500)
- `--prompt`: Custom prompt to use for testing

#### Example
```bash
python3 test_connection_pooling.py \
  --api-url https://api.example.com/v1/chat/completions \
  --api-key your-api-key-here \
  --model llama-3-8b \
  --test-duration 100 \
  --sleep-duration 20 \
  --max-tokens 100
```

## Output

The test provides detailed statistics including:

- **TTFT Statistics**: Median, mean, min, and max Time To First Token
- **Queuing Time Statistics**: Median and mean queuing times
- **Total Response Time Statistics**: Median and mean total response times

## Key Changes from Original

1. **Removed Environment Variables**: No longer requires `API_URL`, `API_KEY`, and `MODEL` environment variables
2. **Command Line Arguments**: All parameters are now configurable via command line arguments
3. **Modular Design**: Main logic is now a reusable function
4. **Enhanced Error Handling**: Better handling of missing metrics and failed requests
5. **Comprehensive Statistics**: More detailed statistical analysis of results
6. **Flexible Configuration**: Customizable test duration, sleep duration, and other parameters

## Files

- `test_connection_pooling.py`: Main test implementation
- `README.md`: This documentation file
