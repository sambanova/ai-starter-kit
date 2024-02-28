NUM_DP_WORKERS=8
MAX_SEQ_LEN=4096
ROOT_GEN_DATA_PREP_DIR=/scratch/imranr/generative_data_prep/
INPUT_FILE=/import/snvm-sc-scratch1/chenw/data/processed_data/qa_article_mix.jsonl
OUTPUT_DIR=/scratch/imranr/YoDa/post_processed_data_7bchat/

cd $ROOT_GEN_DATA_PREP_DIR && python -m generative_data_prep pipeline \
--input_file_path=$INPUT_FILE \
--output_path=$OUTPUT_DIR \
--shuffle on_RAM \
--max_seq_length=$MAX_SEQ_LEN \
--pretrained_tokenizer=/import/ml-sc-nlpcheckpoints-scratch/jonathanl/generic_checkpoints/llama_2/Llama-2-7b-chat-hf \
--overwrite_output_path \
--input_packing_config single::truncate_right \
--num_training_splits=$NUM_DP_WORKERS \
--prompt_prefix=[INST] \
--prompt_postfix=[/INST]
