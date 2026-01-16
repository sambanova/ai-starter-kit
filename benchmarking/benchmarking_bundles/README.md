# Benchmarking scripts

## Required steps

The required steps for CoE benchmarking are the following:

- Run the benchmarking script. Cf. [Benchmark Model Bundle](./Benchmark_Model_Bundle.ipynb).


## Strategy to select the models in a bundle

## Benchmarking script

Modify the following files:

- _PATH TO AISK REPO HERE/benchmarking/benchmarking_scripts/config.yaml_

    With the desired input and output paths.

    ```yaml
    model_configs_path: '<PATH TO AISK REPO HERE>/benchmarking/benchmarking_scripts/model_configs_example.csv'
    llm_api: 'sambastudio'  # or sncloud for the cloud
    output_files_dir: '<PATH TO AISK REPO HERE>/benchmarking/data/benchmarking_tracking_tests/logs/output_files'
    consolidated_results_dir: '<PATH TO AISK REPO HERE>/benchmarking/data/benchmarking_tracking_tests/consolidated_results'
    timeout: 3600
    time_delay: 0
    ```

- _PATH TO AISK REPO HERE/benchmarking/benchmarking_scripts/model_configs_example.csv_

    With the desired model configurations to test.

    ```csv
    model_name,input_tokens,output_tokens,num_requests,concurrent_requests,qps,qps_distribution,multimodal_img_size
    ```

## Configuration parameters

The configuration table in `model_configs_example.csv` details each individual model of the composition that we would like to test, together with other parameters.

-  `model_name`. The name of the model in the bundle.
-  `input_tokens`. The number of input tokens: The number of input tokens in the generated prompt.
-  `output_tokens`. The number of output tokens: The number of output tokens the LLM can generate.
-  `num_requests`. The number of total requests: Number of requests sent.
    Note: the program can timeout before all requests are sent.
    Configure the Timeout parameter in your `benchmarking_scripts/config.yaml` accordingly.
    The Timeout is the Number of seconds before the program times out
    Statistics (min, max, and mediam) will be calculated over this number of requests.
- `concurrent_requests`. The number of concurrent requests.
    For testing batching-enabled models, this value should be greater than the largest `batch_size` one needs to test.
    The typical batch sizes that are supported are 1, 4, 8, and 16.
- `qps`. Queries per second: the number of queries that will be sent to the endpoint per second.
    Values QPS<10 are recommended since user can hit rate limits. Defaults to 1.0.
- `qps_distribution`. Queries per second distribution: the type of wait time distribution in between requests.
    User can choose the values `constant`, `uniform`, and `exponential`. Defaults to constant.
- `multimodal_img_size`. If the model selected is multimodal, then select the pre-set image size to include in the benchmarking requests.
    There are three categories: `Small` (500x500px), `Medium` (1000x1000px), and `Large` (2000x2000px).
    Otherwise, if model is not multimodal, then leave the value to N/A.

Note that the sum of `input_tokens` and `output_tokens` should not exceed the sequence length of the model that we want to test.
E.g. if we want to test how a model performs with a sequence lenght of 4096 tokens, we can set `input_tokens` to 4000 and `output_tokens` to 64, so that their sum does not exceed 4096.

## Results

### Individual results
Individual results will be stored in the `output_files_dir` defined in your `benchmarking_scripts/config.yaml`.

For every model configuration defined in each row of your `model_configs_example.csv` table, two results will be stored.

1. Individual responses files end will `individual_responses.json`.
    They contain performance metrics for every request.
    E.g. if `num_requests` is set to 10, you will have 10 entries.

2. Summary files end will `summary.json`.
    They contain statistics calculated from the individual responses file, such as the min, the max, several quantiles, and the standard deviation.

### Consolidated results

Consolidated results will be stored in the `consolidated_results_dir` defined in your `benchmarking_scripts/config.yaml`.

Every row of the `model_configs_example.csv` will have its corresponding row in the consolidated results table, i.e. `consolidated_results_<date>.xlsx`.

Terminology.
- Server metrics: These are performance metrics from the Server API.
    A prefix `server` will be added to the metric name.
- Client metrics: These are performance metrics computed on the client side / local machine.
    A prefix `client` will be added to the metric name.
- The suffixes `min`, `max`, and `p_50` represent the minimum, maximum, and the median, respectively.
- The suffix `s` indicates that the value is in seconds.
- The prefix `mean` refers to the mean value.

The main columns of consolidated results are the following:
- `ttft`. Time to First Token (TTFT).
    One should see higher values and higher variance in the client-side metrics compared to the server-side metrics. This difference is mainly due to the request waiting in the queue to be served (for concurrent requests), which is not included in server-side metrics.
- `server_end_to_end_latency`. End-to-end latency.
    One should see higher values and higher variance in the client-side metrics compared to the server-side metrics. This difference is also mainly due to the request waiting in the queue to be served (for concurrent requests), which is not included in server-side metrics.
- `server_output_token_per_s`. Output throughput, i.e. the number of output tokens per second per request.
    One should see good agreement between the client and server-side metrics.
    For endpoints that support dynamic batching, one should see a decreasing trend in metrics as the batch size increases.
- `acceptance_rate`. Acceptance rate for speculative decoding pairs.


### Switching time
The switching time indicates the time required to switch from one model to another in a CoE bundle in the context of inference.
A switch is triggered not only by a different model name, but also a different sequence length and a different batch size.
The latest technology of RDUs, SN40L, also offers High-speed HBM memory bandwidth to significantly speed up inference workloads.
Each time that a new model configuration is called for inference, it needs to be loaded into the HBM memory.
The switching time can be zero if the HBM memory is not saturated (e.g. when smaller model configurations, with fewer number of parameters, or smaller sequence lengths, or smaller batch sizes are involved in the switch).
On the other hand, if the HBM memory is saturated, the memory loading operation leads to a non zero switching time.

You can extract the switching time of a request from the `switching_time` key in its corresponding first entry in the `individual_responses.json` file.

In order to estimate the switching time of a model in a CoE bundle of `N` models, it is important to rotate the order in which the models will be called in your `model_configs_example.csv` configuration file.
For example, if your bundle has 3 models configurations that could trigger a switch, you could run the benchmarking scripts several times, in order to get a better idea on how the order of the rows can influence the switching time.

The switching time is estimated as follows.
Based on the first request TTFT in `individual_responses.json`, if this value is significantly larger (more than 3 standard deviations) than the average TTFT of the remaining requests, then the switching time will be the difference between first TTFT and the average of the remaining TTFTs.

## Notebooks
TODO: Add the links to all the relevant notebooks.
