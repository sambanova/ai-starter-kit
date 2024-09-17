#!/bin/bash

##./run_dynamic_batching.sh --streamURL [string from endpoint page] --apiKEY [string from endpoint page] --expert [expert name; special-standalone-model is for llama2 standalone dynamic batching models]
##                          --instance [number of instances an endpoint is running [1-4]]
##                          --maxToken [default: 512 [100-1000], max_token_to_generate]
##                          --delay [default: 0 [0-5], delay to insert among requests]
##                          --mode [default: benchmark [benchmark|all|gold|negative], select from running benchmark and gold only or sweep do_sample and process_prompt]
##                          --random [default: Y [Y|N], select if randomly pick prompt in a file or sequentially pick prompt]
##                          --promptFile [default: ../../data/json/mixpanel_unique_prompts.json, a json file (list of prompts) to test]
##                          --stream [default: Y [Y|N]], select model is running under streaming model or prompt(completion) mode]
##                          --tasknum [default: [52], number of concurrent tasks]
##example: ./run_dynamic_batching.sh --streamURL https://your-env-here/api/predict/generic/stream/123456ab-cdef-4321-9876-abcdef123456/987654fe-dcba-4321-abcd-fedcba987654 --apiKEY abcdef12-3456-7890-abcd-ef1234567890 --expert Meta-Llama-3-70B-Instruct --promptFile ./data/json/ce_prompts.json --random N

streamURL=""
apiKEY=""
expert=""
instance=""
app=""
maxToken=""
delay=""
mode=""
random=""
promptFile=""
stream=""
tasknum=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --streamURL) streamURL="$2"; shift ;;
        --apiKEY) apiKEY="$2"; shift ;;
        --expert) expert="$2"; shift ;;
        --instance) instance="$2"; shift ;;
        --app) app="$2"; shift ;;
        --maxToken) maxToken="$2"; shift ;;
        --delay) delay="$2"; shift ;;
        --mode) mode="$2"; shift ;;
        --random) random="$2"; shift ;;
        --promptFile) promptFile="$2"; shift ;;
        --stream) stream="$2"; shift ;;
        --tasknum) tasknum="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate mandatory arguments
if [[ -z "$streamURL" || -z "$apiKEY" || -z "$expert" || -z "$instance" ]]; then
    echo "Error: --streamURL, --apiKEY, --expert and --instance  are required arguments."
    exit 1
fi

# Build the command with mandatory arguments
cmd="python3 benchmark_dynamic_batching_v2_api.py -e \"$expert\" -k \"$apiKEY\" -u \"$streamURL\" -i \"$instance\""

# Append optional arguments if they are set
if [[ -n "$app" ]]; then
    cmd+=" -a \"$app\""
fi

if [[ -n "$maxToken" ]]; then
    cmd+=" --max_words_list \"$maxToken\""
fi

if [[ -n "$delay" ]]; then
    cmd+=" --max_sleep_time \"$delay\""
fi

if [[ -n "$mode" ]]; then
    cmd+=" -m \"$mode\""
fi

if [[ -n "$random" ]]; then
    cmd+=" -r \"$random\""
fi

if [[ -n "$promptFile" ]]; then
    cmd+=" -p \"$promptFile\""
fi

if [[ -n "$stream" ]]; then
    cmd+=" -s \"$stream\""
fi

if [[ -n "$tasknum" ]]; then
    cmd+=" -tn \"$tasknum\""
fi

echo $cmd
# Execute the command
eval $cmd