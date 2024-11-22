"""
    An openai client that has predefined:
        - sending rate in QPS
        - timeout for average latency
    And calculates the statistics:
        - average query latency
"""
import argparse
import asyncio
import time

import openai


async def send_query(client, model, prompt, timeout, max_tokens):
    start_time = time.time()
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}], 
            max_tokens=max_tokens, 
            timeout=timeout
        )

        end_time = time.time()
        latency = end_time - start_time

        return latency, response.choices[0].message.content
    except openai.APITimeoutError as e:
        return None, "Empty response"
    except Exception as e:
        raise Exception(f"Exception: {e}")


async def main(args):
    print(f"Profiling server with {args.qps} QPS and {args.timeout}s timeout")

    client = openai.AsyncOpenAI(
        base_url=f"http://{args.host_name}:{args.port}/v1", 
        api_key=args.api_key
    )

    prompt = ""
    for i in range(1500):
        prompt += f"line {i}, key: 123456\n"
    prompt += "line 1500, key: 245678\n"
    for i in range(300):
        prompt += f"line {1501 + i}, key: 123456\n"
    prompt += "\nWhich line is the key 245678?"

    responses = []

    for _ in range(args.num_samples):
        await asyncio.sleep(1 / args.qps)
        print("Send")
        responses.append(asyncio.create_task(send_query(
            client=client, 
            model=args.model, 
            prompt=prompt, 
            max_tokens=args.max_tokens, 
            timeout=args.timeout
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
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")

    # server related args
    parser.add_argument("--qps", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--host-name", type=str, default="localhost")
    parser.add_argument("--port", type=str, default="8888")
    parser.add_argument("--api-key", type=str, default="local_server")

    # data related args
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
