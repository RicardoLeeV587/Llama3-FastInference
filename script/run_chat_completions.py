import argparse
import json, os
import asyncio

from openai import OpenAI, AsyncOpenAI

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--base_url', default=None, type=str, required=True, help="Request url")
parser.add_argument('--api_key', default=None, type=str, required=True, help="Authority key for OpenAI client")
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--data_file', default=None, type=str, required=True, help="A file that contains instructions (Alpaca json format)")
parser.add_argument('--predictions_file', default='./predictions.json', type=str, required=True)
parser.add_argument('--Semaphore', default=2, type=int)
args = parser.parse_args()

generation_config = dict(
    temperature=0.2,
    # top_k=40,
    top_p=0.9,
    # do_sample=True,
    # num_beams=1,
    # repetition_penalty=1.1,
    # max_new_tokens=8192
)

sem = asyncio.Semaphore(args.Semaphore)

# OpenAI client preparation
client = AsyncOpenAI(
    base_url=args.base_url,
    api_key=args.api_key
)

async def generate_response(example: dict) -> dict:
    completion = await client.chat.completions.create(
        model="/root/autodl-tmp/llama3_lora_sft_WASSA_EXP304",
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["instruction"]}
        ],
        **generation_config
    )
    example["response"] = completion.choices[0].message.content
    return example

async def rate_limited_generate_response(example: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:  
        return await generate_response(example)

async def main(args):
    print("[INFO] File path: {}".format(args.data_file))
    with open(args.data_file) as reader:
        content = reader.read()
    sampleList = json.loads(content)

    print("[INFO] Total length of the samples: {}".format(len(sampleList)))
    print("[INFO] First 10 examples:")
    for example in sampleList[:10]:
        print(example)

   
    # client test
    completion = await client.chat.completions.create(
        model=args.base_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of America?"}
        ],
        **generation_config
    )
    print(completion)
    print("[INFO] Connection Success!")

    resultList = []
    tasks_get_result = [rate_limited_generate_response(item, sem) for item in sampleList]
    
    for index, example in enumerate(tqdm(asyncio.as_completed(tasks_get_result), total=len(sampleList))):
        resultList.append(await example)
        if index % 512 == 511:
            print(f"======={index}=======")
            print("Input: {}\n".format(resultList[index]["instruction"]))
            print("Output: {}\n".format(resultList[index]["response"]))

    dirname = os.path.dirname(args.predictions_file)
    os.makedirs(dirname,exist_ok=True)
    with open(args.predictions_file,'w') as f:
        json.dump(resultList,f,ensure_ascii=False,indent=2)   


if __name__ == '__main__':
    asyncio.run(main(args))
