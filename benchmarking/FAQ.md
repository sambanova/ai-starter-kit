# Benchmarking kit - FAQ

## Getting Started


#### Who is this benchmarking kit for?
The kit is meant to be used by potential customers, internally and anyone in the community since it’s open source.


#### Why would I/customers want to benchmark?
Because users want to know how fast endpoints are. The kit provides multiple performance metrics and functionalities so users can measure endpoint’s speed and compare.


#### What are the prerequisites and system requirements?
You can refer to [this](https://github.com/sambanova/ai-starter-kit/blob/main/benchmarking/README.md#create-the-virtual-environment) link to check how to create an environment for the kit, which uses a requirements.txt to set it up.


#### How does this kit differ from tools like vLLM benchmark, GenAI-Perf, or LLMPerf?
The kit is an adaptation of the [LLMPerf framework](https://github.com/ray-project/llmperf). So it shares some of the performance metrics also used in other frameworks like vLLM and GenAI performance modules.



## Metrics & Interpretation


#### Glossary: What are the key metrics and their definitions?
- Time to First Token (TTFT): This is the time in seconds from when the server receives the request to when it produces the first token of the output.
- Latency: The time in seconds taken for the request to complete, including input processing, token generation, and returning the complete output.
- Output token per seconds: Number of *output* tokens *generated* per second after first token for a single request.
You can refer to [this link](https://sambanova.atlassian.net/wiki/spaces/AIS/pages/2123563038) for more detailed information about these and other metrics provided by the kit.


#### What's the difference between throughput-oriented vs latency-oriented benchmarking?
Throughput-oriented benchmarking refers to the use of dynamic batching, which groups API requests together improving the overall throughput of the tests. Users could use multiple concurrent requests to simulate that when using the kit.
On the other hand, latency-oriented benchmarking refers to how long it takes for a request or group of requests to finish. Users could simulate single or multiple requests and measure the number of requests per minute (RPM) or second (RPS). In the kit’s summary output file, search for the `results_num_completed_requests_per_min` , which refers to RPM.


#### How do I interpret Time to First Token (TTFT) vs end-to-end latency vs inter-token latency?
TTFT is how long you wait before the endpoint starts responding, inter-token latency is how quickly the endpoint continues producing each word after it starts, and latency is the total time it takes for the entire answer to be finished.


#### Why is there a big difference between the tokens I've defined and the tokens shown in the kit?
If the difference is big (more than 50 tokens) this might be a tokenizer issue. Please submit an [issue](https://github.com/sambanova/ai-starter-kit/issues) with details explaining these error, we’ll try our best to reach you as soon as possible.



## Configuration & Usage


#### What are common configs/models I can use as a starting point?
You can test the kit by using the default config and model values either in the UI or CLI modes. Default values should work and provide results quickly.


#### What types of inference does the kit support?
The kit supports text alone and multimodal payloads. On the multimodal front, only image with text payloads are supported.


#### How do I run concurrent/parallel request loads?
Users can use the *number of concurrent requests* or *queries per second* (QPS) parameters to send requests concurrently. If the user is interested in using the [Synthetic Performance Evaluation](https://github.com/sambanova/ai-starter-kit/pull/788), then the number of concurrent requests will be used. Similarly, if user wants to use the [Real Workload Performance Evaluation](https://github.com/sambanova/ai-starter-kit/pull/788), then QPS will be used.


#### If I want to test a custom prompt, what should I do?
If users would like to test their custom prompts, then the [Custom Performance Evaluation](https://github.com/sambanova/ai-starter-kit/pull/788) feature is the one to choose, which can be done either using the UI or CLI. 


#### How do I simulate realistic production traffic patterns?
For that purpose, users should refer to the *Real Workload Performance Evaluation* feature, which simulates production workload using synthetic data.


#### The UI version has limits on input/output tokens—what can I do?
If you’re an advanced user or just would like to have more flexibility, then please use the [CLI version](https://github.com/sambanova/ai-starter-kit/blob/main/benchmarking/README.md#cli-option) instead. The kit doesn’t have restrictions there, however users will need to have extra careful to not saturate the API server with multiple concurrent requests coming from the kit.


#### What should I use to run my specific use case?
It’s always recommended to use the UI for general purpose cases. If users need more flexibility, then they can use the CLI version. There are also some useful notebooks [here](https://github.com/sambanova/ai-starter-kit/tree/main/benchmarking/notebooks) that provide more insights out from the kit’s output files.



## Reproducibility & Comparison


#### How do I ensure reproducible benchmark results?
To ensure that the benchmark results will be the same, users need to consider the following:
- Use an API with the same environment setup (API URL and key, Helm version, models, PEF versions, checkpoint versions, etc) 
- Use the same kit parameters (input/output tokens, concurrent requests, number of requests, etc)


#### How do I compare my results to published benchmarks or competitor claims?
There are many metrics from the kit that are standard across published benchmarks. Users can use client TTFT, latency, total tokens (input + output) per second, and number of completed requests per minute.


#### Does the kit include warmup runs, and how are they handled?
The kit *does not* include warmup runs. If users would like to test a fresh API environment, then they’ll need to do some warmup runs first before the actual test.


#### How do I export or share my results?
The kit outputs two JSON files, which are generated locally and could be share with anyone
- `synthetic_<MODEL_IDX><MODEL_NAME><MULTIMODAL_SUFFIX>{NUM_INPUT_TOKENS}{NUM_OUTPUT_TOKENS}{NUM_CONCURRENT_REQUESTS}_individual_responses`
- `synthetic_<MODEL_IDX><MODEL_NAME><MULTIMODAL_SUFFIX>{NUM_INPUT_TOKENS}{NUM_OUTPUT_TOKENS}{NUM_CONCURRENT_REQUESTS}_summary`
More details about these files can be found [here](https://github.com/sambanova/ai-starter-kit/blob/main/benchmarking/README.md#cli-option) in the Synthetic Dataset expandable section.



## Troubleshooting


#### I've cloned the repository but it's not working—what could be happening?
Several issues could be happening, consider the following
- Validate that you’re using the latest `main` version of the kit
- Validate that you’ve installed the [required packages](./requirements.txt) and libraries and that the corresponding environment is the only one active. If you already had created an environment previously, then reinstalling the packages is recommended after pulling the latest changes from the repository.
- Be sure to run anything from the kit’s root.
- Check that you’ve defined the right Sambanova URL and API key in the `.env` file, which has to be located in the repository’s root.
- Check that the model is available and healthy in the API server. Also, check that the model name has been spelled correctly.


#### Where can I find more details about errors from the kit?
For general errors, users can use the terminal or the `_summary` json file to see them. If someone would like to see errors per request, then they should go to `_individual_responses` json files.


#### Why am I seeing high variance in my results?
Users could encounter some variance in their results, which could be cause by some of the following issues:
- Unstable internet or network connection
- API server highly saturated with requests
- Errors per request that may affect aggregated metrics


#### How do I troubleshoot unexpectedly slow performance?
There may be multiple issues happening, so try to check the following:
- API URL or key could be wrong, so the request hangs for some time
- The model specified can’t be found by the API so the request hangs for a while
- The API is already processing multiple requests so the serving queue is saturated


#### Why is there a big difference between the tokens that I’ve defined and the tokens shown in the kit?
If the difference is substantial (e.g. larger than 50 tokens) then the issue might be the tokenizer. Please submit an [issue](https://github.com/sambanova/ai-starter-kit/issues) with details explaining these error, we’ll try our best to reach you as soon as possible.


## A few additional questions worth considering:


#### How do I benchmark streaming vs batch inference modes?
The kit currently only supports benchmarking using *streaming* responses. It does not measure the performance of *batch* (non-streaming) inference.


#### What datasets/prompt distributions are recommended for different use cases (chat, RAG, code generation, etc.)?
The kit mainly uses synthetic data, but users can use their custom data throughout the [Custom Performance Evaluation](https://github.com/sambanova/ai-starter-kit/pull/788) feature of the kit. For now, the accepted format is just plain text.


#### How do I account for tokenizer differences when comparing across models?
If you see big differences between expected vs server input or output tokens, then the issue might be that the kit it’s not using the right tokenizer. Please submit an [issue](https://github.com/sambanova/ai-starter-kit/issues) with details explaining these error, we’ll try our best to reach you as soon as possible.


#### Can I run the kit against non-SambaNova endpoints for comparison?
Yes, you can run any other LLM provider over the kit if they’re OpenAI compatible. [Here's](../README.md#use-sambanova-cloud-option-1) the step-by-step on how to set your SambaNova base url and API key.