#!/bin/bash

# Default values
streamURL=""
apiKEY=""
experts=""
app=""
maxToken=512
tasknum=200
delay=0.0625
mode="gq"
stream="Y"
random="N"
promptDirPath="./data/json"
promptLongSuffix="_qaLong_prompts.json"
promptShortSuffix="_qaShort_prompts.json"
promptLongSuperSuffix="_qaLong_prompts_superbak.json"

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --streamURL URL       Set the stream URL (required)"
    echo "  --apiKEY KEY          Set the API key (required)"
    echo "  --experts LIST        Comma-separated list of experts (required)"
    echo "  --app NAME            Set the app name (required)"
    echo "  --maxToken NUM        Set max token (default: 512)"
    echo "  --tasknum NUM         Set task number (default: 200)"
    echo "  --delay NUM           Set delay (default: 0.0625)"
    echo "  --mode MODE           Set mode (default: gq)"
    echo "  --stream Y/N          Enable/disable streaming (default: Y)"
    echo "  --random Y/N          Enable/disable random (default: N)"
    echo "  --promptDir PATH      Set prompt directory path (default: ../../data/json)"
    echo "  --promptLongSuffix S  Set long prompt file suffix (default: _qaLong_prompts.json)"
    echo "  --promptShortSuffix S Set short prompt file suffix (default: _qaShort_prompts.json)"
    echo "  --promptLongSuperSuffix S Set long super prompt file suffix (default: _qaLong_prompts_superbak.json)"
    echo "  --help                Display this help message"
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --streamURL) streamURL="$2"; shift ;;
        --apiKEY) apiKEY="$2"; shift ;;
        --experts) experts="$2"; shift ;;
        --app) app="$2"; shift ;;
        --maxToken) maxToken="$2"; shift ;;
        --tasknum) tasknum="$2"; shift ;;
        --delay) delay="$2"; shift ;;
        --mode) mode="$2"; shift ;;
        --stream) stream="$2"; shift ;;
        --random) random="$2"; shift ;;
        --promptDir) promptDirPath="$2"; shift ;;
        --promptLongSuffix) promptLongSuffix="$2"; shift ;;
        --promptShortSuffix) promptShortSuffix="$2"; shift ;;
        --promptLongSuperSuffix) promptLongSuperSuffix="$2"; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; usage; exit 1 ;;
    esac
    shift
done

# Validate mandatory arguments
if [[ -z "$streamURL" || -z "$apiKEY" || -z "$experts" || -z "$app" ]]; then
    echo "Error: --streamURL, --apiKEY, --experts, and --app are required arguments."
    usage
    exit 1
fi

# Function to determine prompt files
get_prompt_files() {
    local expert=$1
    local lastNumber=$(echo "$expert" | grep -oP '\d+$')
    local prefix="test"
    
    if [[ "$expert" == *"deepseek-coder"* ]]; then
        prefix="test_deepseek"
    fi
    
    local longFile="${promptDirPath}/${prefix}${promptLongSuffix}"
    local shortFile="${promptDirPath}/${prefix}${promptShortSuffix}"

    if [[ -n "$lastNumber" && "$lastNumber" -gt 4096 ]]; then
        longFile="${promptDirPath}/${prefix}${promptLongSuperSuffix}"
    fi

    echo "$longFile $shortFile"
}

# Main execution
IFS=',' read -r -a expertArray <<< "$experts"
for expert in "${expertArray[@]}"; do
    echo "Running benchmark for expert: $expert"
    
    read -r promptFileLong promptFileShort <<< $(get_prompt_files "$expert")
    echo "Using prompt files: $promptFileLong, $promptFileShort"

    echo "Running with delay: $delay"
    ./run_dynamic_batching_v2.sh \
        --streamURL "$streamURL" \
        --apiKEY "$apiKEY" \
        --expert "$expert" \
        --app "$app" \
        --instance 1 \
        --promptFile "$promptFileShort" \
        --maxToken "$maxToken" \
        --random "$random" \
        --delay "$delay" \
        --mode "$mode" \
        --stream "$stream" \
        --tasknum "$tasknum"
done