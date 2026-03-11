"""
Multi-turn benchmark runner for vLLM profiling.
Sends multi-turn conversations to a running vLLM server to exercise
KV cache reuse across turns. Each conversation has 2-4 turns built
from ShareGPT data, with the same conversation_id linking turns in traces.
"""
import argparse
import asyncio
import json
import time
import uuid
from typing import List, Tuple

import aiohttp


def load_sharegpt_conversations(
    dataset_path: str,
    num_conversations: int,
    max_turns: int = 4,
    max_chars_per_turn: int = 4000,
) -> List[List[dict]]:
    """Load multi-turn conversations from ShareGPT JSON.

    Returns a list of conversations, each being a list of
    {"role": "user"/"assistant", "content": ...} message dicts.
    Only conversations with >= 2 turns are kept.
    """
    with open(dataset_path) as f:
        data = json.load(f)

    conversations = []
    for conv in data:
        msgs = conv.get("conversations", [])
        if len(msgs) < 2:
            continue

        turns = []
        for msg in msgs[:max_turns * 2]:  # human/gpt pairs
            role = "user" if msg.get("from") == "human" else "assistant"
            text = (msg.get("value") or "").strip()
            if not text:
                continue
            if len(text) > max_chars_per_turn:
                text = text[:max_chars_per_turn]
            turns.append({"role": role, "content": text})

        # Keep only if we have at least 2 user turns
        user_turns = [t for t in turns if t["role"] == "user"]
        if len(user_turns) >= 2:
            conversations.append(turns)

        if len(conversations) >= num_conversations:
            break

    return conversations[:num_conversations]


async def send_multiturn_conversation(
    session: aiohttp.ClientSession,
    url: str,
    conversation: List[dict],
    conv_idx: int,
    max_tokens: int,
    model: str,
    semaphore: asyncio.Semaphore,
    stream: bool = True,
) -> Tuple[int, int, int]:
    """Send a multi-turn conversation one turn at a time.

    Each turn sends the full message history up to that point,
    exercising prefix caching for prior turns.

    Returns (conv_idx, turns_ok, turns_total).
    """
    conv_id = f"conv_{conv_idx:05d}"
    history = []
    turns_ok = 0
    turn_num = 0

    for msg in conversation:
        history.append(msg)
        if msg["role"] != "user":
            continue

        turn_num += 1
        turn_id = f"{conv_id}_turn{turn_num:02d}_{uuid.uuid4().hex[:8]}"

        payload = {
            "model": model,
            "messages": list(history),
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": stream,
            "request_id": turn_id,
            "user": f"{conv_id}_turn{turn_num:02d}",
        }

        async with semaphore:
            try:
                async with session.post(
                    url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        print(f"[{conv_id} turn {turn_num}] ERROR {resp.status}: {body[:200]}")
                        continue

                    # Collect assistant response to append to history
                    assistant_text = ""
                    if stream:
                        async for line in resp.content:
                            decoded = line.decode("utf-8", errors="replace").strip()
                            if decoded.startswith("data: ") and decoded != "data: [DONE]":
                                try:
                                    chunk = json.loads(decoded[6:])
                                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                                    assistant_text += delta.get("content", "")
                                except json.JSONDecodeError:
                                    pass
                    else:
                        body = await resp.json()
                        assistant_text = (
                            body.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )

                    # Add assistant response to history for next turn
                    if assistant_text:
                        history.append({"role": "assistant", "content": assistant_text})

                    turns_ok += 1
                    print(f"  [{conv_id}] turn {turn_num} OK "
                          f"(history={len(history)} msgs, ~{sum(len(m['content']) for m in history)//4} tokens)")

            except Exception as e:
                print(f"[{conv_id} turn {turn_num}] EXCEPTION: {e}")

    return conv_idx, turns_ok, turn_num


async def run_benchmark(
    host: str,
    port: int,
    model: str,
    conversations: List[List[dict]],
    max_tokens: int,
    concurrency: int,
    stream: bool = True,
):
    url = f"http://{host}:{port}/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            send_multiturn_conversation(
                session, url, conv, idx, max_tokens, model, semaphore, stream,
            )
            for idx, conv in enumerate(conversations)
        ]

        t0 = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - t0

    total_turns = sum(total for _, _, total in results)
    ok_turns = sum(ok for _, ok, _ in results)
    print(f"\nDone: {ok_turns}/{total_turns} turns succeeded "
          f"across {len(conversations)} conversations")
    print(f"Total time: {elapsed:.1f}s")


def wait_for_server(host: str, port: int, timeout: int = 300):
    import socket
    print(f"Waiting for server at {host}:{port} ...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(" ready!")
                return
        except OSError:
            print(".", end="", flush=True)
            time.sleep(3)
    raise TimeoutError(f"Server at {host}:{port} did not start within {timeout}s")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn benchmark for vLLM profiling. "
        "Sends multi-turn conversations to exercise KV cache reuse.",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", required=True,
                        help="Model name (must match server --model)")
    parser.add_argument("--dataset", required=True,
                        help="Path to ShareGPT JSON file")
    parser.add_argument("--num-conversations", type=int, default=50,
                        help="Number of multi-turn conversations to send")
    parser.add_argument("--max-turns", type=int, default=4,
                        help="Max user turns per conversation (default: 4)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens per turn response (default: 128)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Max concurrent conversations")
    parser.add_argument("--wait-server", action="store_true",
                        help="Wait for server to be ready")
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    args = parser.parse_args()

    if args.wait_server:
        wait_for_server(args.host, args.port)

    conversations = load_sharegpt_conversations(
        args.dataset, args.num_conversations, args.max_turns,
    )
    total_turns = sum(
        1 for conv in conversations for m in conv if m["role"] == "user"
    )
    print(f"Loaded {len(conversations)} conversations ({total_turns} user turns) "
          f"from {args.dataset}")
    print(f"Sending to {args.host}:{args.port}  model={args.model}  "
          f"max_tokens={args.max_tokens}  concurrency={args.concurrency}  "
          f"stream={args.stream}")

    asyncio.run(run_benchmark(
        host=args.host,
        port=args.port,
        model=args.model,
        conversations=conversations,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        stream=args.stream,
    ))


if __name__ == "__main__":
    main()
