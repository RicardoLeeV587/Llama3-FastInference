#!/bin/bash

base_model="/root/autodl-tmp/llama3_lora_sft_WASSA_EXP304"
api_key="12345"
n_gpu=2

python -m vllm.entrypoints.openai.api_server --model ${base_model} --dtype auto --tensor-parallel-size ${n_gpu} --api-key ${api_key}
