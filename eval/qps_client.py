"""
    An openai client that has predefined:
        - sending rate in QPS
        - timeout for average latency
    And calculates the statistics:
        - average query latency
"""
import argparse
import asyncio
import json
import random
import time
from datetime import datetime

import numpy as np
import openai
import torch
from datasets import load_dataset


async def send_query(
    client, 
    model, 
    prompt, 
    timeout, 
    max_tokens, 
    query_type
):
    try:
        start_time = time.time()

        if query_type == "chat":
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}], 
                max_tokens=max_tokens, 
                timeout=timeout, 
                temperature=0.0
            )
            answer = response.choices[0].message.content
        elif query_type == "generate":
            response = await client.completions.create(
                model=model,
                prompt=prompt,  
                max_tokens=max_tokens, 
                timeout=timeout, 
                temperature=0.0
            )
            answer = response.choices[0].text
        else:
            raise ValueError(f"Unknown query type: {query_type}")

        end_time = time.time()
        latency = end_time - start_time

        return latency, answer
    except openai.APITimeoutError as e:
        return None, "Empty response"
    except Exception as e:
        raise Exception(f"Exception: {e}")


def prepare_datasets(category, num_samples):
    dataset_map = {
        "multi-doc-qa": [
            "dureader", "2wikimqa", "musique", "hotpotqa"
        ], 
        "summarization": [
            "gov_report", "qmsum", "multi_news", "vcsum"
        ], 
        "few-shot-learning": [
            "triviaqa", "lsht", "trec", "samsum"
        ]
    }

    dataset2prompt = json.load(open("./eval/long_bench/configs/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("./eval/long_bench/configs/dataset2maxlen.json", "r"))

    samples = []

    for dataset in dataset_map[category]:
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        data = data.shuffle().select(range(num_samples))
        prompt_template = dataset2prompt[dataset]
        max_tokens = dataset2maxlen[dataset]

        for d in data:
            samples.append((
                prompt_template.format(**d), 
                max_tokens, 
                "generate" if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"] else "chat"
            ))

    random.shuffle(samples)
    return samples


async def main(args):
    # seeding seeds
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"Generating data in {args.category} with {args.num_samples} samples per dataset.")
    samples = prepare_datasets(args.category, args.num_samples)

    print(f"Profiling server with {args.qps} QPS and {args.timeout}s timeout")

    client = openai.AsyncOpenAI(
        base_url=f"http://{args.host_name}:{args.port}/v1", 
        api_key=args.api_key
    )

    responses = []

    for idx, (prompt, max_tokens, query_type) in enumerate(samples):
        await asyncio.sleep(1 / args.qps)
        print(f"[{datetime.now().strftime('%Hh:%Mm:%Ss')}] Send request {idx + 1}/{len(samples)}")
        responses.append(asyncio.create_task(send_query(
            client=client, 
            model=args.model, 
            prompt=prompt, 
            max_tokens=max_tokens if args.max_tokens is None else args.max_tokens, 
            timeout=args.timeout, 
            query_type=query_type
        )))

    results = await asyncio.gather(*responses)

    latency, _ = zip(*results)

    if any([l is None for l in latency]):
        print(f"Found timeout in queries")
    else:
        print(f"Average latency: {sum(latency) / len(latency):.3f}s")


def parse_args():    
    parser = argparse.ArgumentParser(description="QPS client parser.")

    # model related info
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")

    # server related args
    parser.add_argument("--qps", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--host-name", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="8888")
    parser.add_argument("--api-key", type=str, default="local_server")

    # data related args
    parser.add_argument("--seed", type=int, default=227)
    parser.add_argument("--category", type=str, default="multi-doc-qa", choices=[
        "multi-doc-qa", "summarization", "few-shot-learning"
    ])
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=20)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
