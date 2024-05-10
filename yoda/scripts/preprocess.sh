NUM_DP_WORKERS=1
MAX_SEQ_LEN=4096
ROOT_GEN_DATA_PREP_DIR=/your/path/generative_data_prep
INPUT_FILE=/your/path/ai-starter-kit/yoda/data/output/processed_data/synthetic_qa_train.jsonl
OUTPUT_DIR=/your/path/ai-starter-kit/yoda/data/output/preprocessed
PATH_TO_TOKENIZER=/your/path/ai-starter-kit/yoda/data/output/preprocessed/tokenizer

cd $ROOT_GEN_DATA_PREP_DIR && python -m generative_data_prep pipeline \
--input_file_path=$INPUT_FILE \
--output_path=$OUTPUT_DIR \
--shuffle on_RAM \
--max_seq_length=$MAX_SEQ_LEN \
--pretrained_tokenizer=$PATH_TO_TOKENIZER \
--overwrite_output_path \
--input_packing_config single::truncate_right \
--num_training_splits=$NUM_DP_WORKERS \
--prompt_prefix=[INST] \
--prompt_postfix=[/INST]
