#!/bin/bash

base_url="http://localhost:8000/v1"
api_key="12345"

base_model="/root/autodl-tmp/llama3_lora_sft_WASSA_EXP304"
data_file="/root/autodl-tmp/WASSA2024_Llama3/test/Track2/constractive_multiTurn_empathy_2780.json"

predictions_file="/root/autodl-tmp/result/EXP304_contrastive_multiTurn_empathy/result.json"
Semaphore=32

python ./script/run_chat_completions.py \
    --base_url ${base_url} \
    --api_key ${api_key} \
    --base_model ${base_model} \
    --data_file ${data_file} \
    --predictions_file ${predictions_file} \
    --Semaphore ${Semaphore}
