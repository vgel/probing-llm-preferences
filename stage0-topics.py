import asyncio
import csv
from datetime import datetime
import json
import os
from typing import Any

import anthropic
import dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

dotenv.load_dotenv()

MODELS = [
    (anthropic.AsyncAnthropic(), "claude-opus-4-1-20250805"),
    (anthropic.AsyncAnthropic(), "claude-opus-4-20250514"),
    (anthropic.AsyncAnthropic(), "claude-sonnet-4-20250514"),
    (anthropic.AsyncAnthropic(), "claude-3-7-sonnet-20250219"),
    (anthropic.AsyncAnthropic(), "claude-3-5-sonnet-20241022"),
    (anthropic.AsyncAnthropic(), "claude-3-5-haiku-20241022"),
    (anthropic.AsyncAnthropic(), "claude-3-5-sonnet-20240620"),
    (anthropic.AsyncAnthropic(), "claude-3-haiku-20240307"),
    (anthropic.AsyncAnthropic(), "claude-3-opus-20240229"),
    (
        anthropic.AsyncAnthropicBedrock(aws_region="us-west-2"),
        "anthropic.claude-3-sonnet-20240229-v1:0",
    ),
]
PROMPTS = [
    "Please feel free to talk about anything you want, any topic you like",
    "What would you most like to talk about or work on if you could select anything?",
    "What would you most like to talk about or work on if given complete freedom?",
]
MAX_TOKENS = 4000
N_CALLS = 100
BEDROCK_N_CALLS = 20 # fewer calls for bedrock, slow!
OUT_DIR = "results/stage0-topics"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def call_anthropic(
    client: anthropic.AsyncAnthropic | anthropic.AsyncAnthropicBedrock,
    model: str,
    prompt: str,
    max_tokens: int,
    call_number: int,
) -> dict[str, Any]:
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "call_number": call_number,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": response.content[0].text,  # type: ignore
        "model": model,
        "temperature": 1.0,
        "max_tokens": max_tokens,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        "status": "success",
    }


async def call_n_and_save(
    client: anthropic.AsyncAnthropic | anthropic.AsyncAnthropicBedrock,
    model: str,
    prompt: str,
    max_tokens: int,
):
    """Make multiple API calls with optional delay between calls"""

    if isinstance(client, anthropic.AsyncAnthropicBedrock):
        num_calls = BEDROCK_N_CALLS
    else:
        num_calls = N_CALLS

    print(f"\nStarting batch of {num_calls} calls...")

    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{model}_{timestamp}"
    json_filename = os.path.join(OUT_DIR, f"{base_filename}.json")
    csv_filename = os.path.join(OUT_DIR, f"{base_filename}.csv")

    calls = []
    for i in range(1, num_calls + 1):
        c = call_anthropic(
            client=client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            call_number=i,
        )
        calls.append(c)
    if isinstance(client, anthropic.AsyncAnthropicBedrock):
        # idk the ratelimits but they're low
        results = []
        for i, c in enumerate(calls):
            print("Slow-calling Bedrock...", i, "/", N_CALLS)
            results.append(await c)
            await asyncio.sleep(10.0)
    else:
        results = await asyncio.gather(*calls)

    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "call_number",
            "timestamp",
            "prompt",
            "response",
            "model",
            "temperature",
            "max_tokens",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "status",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field, "") for field in fieldnames}
            writer.writerow(row)

    with open(json_filename, "a", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("BATCH COMPLETED")
    print("=" * 50)

    # Summary statistics
    successful_calls = sum(1 for r in results if r["status"] == "success")
    failed_calls = num_calls - successful_calls
    total_tokens = sum(r["total_tokens"] for r in results)

    print(f"Successful calls: {successful_calls}")
    print(f"Failed calls: {failed_calls}")
    print(f"Total tokens used: {total_tokens}")

    return json_filename, csv_filename


async def main():
    """Main function to run the batch caller"""

    print("ANTHROPIC API BATCH CALLER")
    print("=" * 50)

    for client, model in MODELS:
        for prompt in PROMPTS:
            print(f"Starting {N_CALLS} batch calls for {model} with {prompt}...")
            filenames = await call_n_and_save(
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
            )
            print(*filenames)


if __name__ == "__main__":
    asyncio.run(main())
