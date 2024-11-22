"""
    An openai client that has predefined:
        - sending rate in QPS
        - timeout for average latency
    And calculates the statistics:
        - average query latency
"""
import asyncio
import time

import openai

HOST_NAME = "localhost"
PORT = 8888
API_KEY = "local_server"

TIMEOUT = 5.0
QPS = 1.5

client = openai.AsyncOpenAI(
    base_url=f"http://{HOST_NAME}:{PORT}/v1", 
    api_key=API_KEY
)

async def send_query(prompt, timeout, max_completion_tokens=32):
    start_time = time.time()
    
    try:
        response = await client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": prompt}], 
            max_completion_tokens=max_completion_tokens, 
            timeout=timeout
        )

        end_time = time.time()
        latency = end_time - start_time

        return latency, response.choices[0].message.content
    except openai.APITimeoutError as e:
        return None, "Empty response"
    except Exception as e:
        raise Exception(f"Exception: {e}")


async def main():
    print(f"Profiling server with {QPS} QPS and {TIMEOUT}s timeout")

    input_text = ""
    for i in range(1500):
        input_text += f"line {i}, key: 123456\n"
    input_text += "line 1500, key: 245678\n"
    for i in range(300):
        input_text += f"line {1501 + i}, key: 123456\n"
    input_text += "\nWhich line is the key 245678?"

    responses = []

    for _ in range(5):
        await asyncio.sleep(1 / QPS)
        print("Send")
        responses.append(asyncio.create_task(send_query(
            input_text, 
            timeout=TIMEOUT
        )))

    result = await asyncio.gather(*responses)

    latency, _ = zip(*result)

    if any([l is None for l in latency]):
        print(f"Found timeout in queries")
    else:
        print(f"Average latency: {sum(latency) / len(latency):.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
