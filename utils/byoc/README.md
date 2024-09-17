# BYOC Import and Benchmark Tool

This tool automates the process of importing models to SambaStudio - Bring Your Own Checkpoint (BYOC). 

Additional tasks are - creating Composite of Experts (COE), deploying endpoints, running benchmarks, and cleaning up resources using SNAPI.

## Prerequisites

- Python 3.7+
- SNAPI CLI installed and configured
- Access to the required SNAPI project and resources

## Installation

1. Clone this repository:

``` bash
git clone https://github.com/sambanova/ai-starter-kit
cd utils/byoc
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Configuration

The tool uses a YAML configuration file (`config.yaml`) to manage various settings. Here's an example of the configuration file:

```yaml
# SNAPI Configuration
project: snapi_LLM_1
chiparch: SN40L

# Paths
master_menu_path: ./data/csv/samba_turbo_byoc_menu.csv
artifacts_path: ./artifacts/hf/

# BYOC Settings
byoc_source: LOCAL

# Job Settings
default_rdu: 1
max_rdu: 8

# Timeouts (in seconds)
command_timeout: 300
endpoint_creation_timeout: 1800
job_completion_timeout: 3600

# Logging
log_file: byoc_process.log
log_level: INFO

# Model-specific settings
models:
GPT_13B:
 appname: GPT13B_App
 modelarch: GPT
 paracnt: 13b
 sslen: 2048
 vocabsize: 50260
Llama_7B:
 appname: Llama7B_App
 modelarch: Llama
 paracnt: 7b
 sslen: 4096
 vocabsize: 32000

# Dataset mappings
datasets:
GPT_13B: GPT_13B_2k_SS_Toy_Training_Dataset
GPT_13B_8k: GPT_13B_8k_SS_Toy_Training_Dataset
Llama_7B: Llama_7B_Toy_Training_Dataset

# Sweep settings
sweep_modes:
- N
- Y

# Benchmark settings
benchmark_script_path: ./m2_run_GQ_turbo_coe.sh
benchmark_prompt_dir: ../../data/json
benchmark_prompt_long_suffix: _qaLong_prompts.json
benchmark_prompt_short_suffix: _qaShort_prompts.json
benchmark_prompt_long_super_suffix: _qaLong_prompts_superbak.json
benchmark_api_url: https://your-env-here.net/api/v2/predict/generic/stream
```

Modify this file to match your environment and requirements.

## Usage

The main script is main_byoc.py. You can run it with various command-line options:

1. Run the script with the default configuration:

```bash
python main_byoc.py
```

2. Specify a different configuration file:

```bash
python main_byoc.py -c path/to/your/config.yaml
```

3. Process a specific model:

```bash
python main_byoc.py -m GPT_13B
```

4. List available models:

## Command-line Options

```bash
-c, --config: #Path to the configuration file (default: config.yaml)
-m, --model: #Process a specific model
-l, --list: #List available models
```



## Workflow

The script performs the following steps for each model:

1. Import the BYOC model
2. Check the checkpoint status
3. Create a Composite of Experts (COE)
4. Deploy an endpoint
5. Run a benchmark
6. Clean up resources (delete endpoint, COE, and BYOC model)

## Logging
The script logs its activities to both the console and a log file specified in the configuration. You can adjust the log level in the configuration file.

## Extending the Tool
To add support for new models or modify existing ones:

- Update the master_menu_path CSV file with the new model information.
- If needed, add model-specific configurations in the models section of the config file.
- Update the datasets mappings in the config file if the new model requires specific datasets.

## Troubleshooting

If you encounter permission issues, ensure that you have the necessary SNAPI permissions and that your CLI is correctly configured.
For timeout errors, try increasing the timeout values in the configuration file.
Check the log file for detailed error messages and the script's progress.


