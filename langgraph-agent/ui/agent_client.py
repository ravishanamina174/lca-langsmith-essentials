"""LangGraph SDK client used by the UI's HTTP routes.

The UI is a custom route inside the LangGraph server, so it reaches the agent the
same way any other client would - over the LangGraph API. That means threads are
checkpointed server side, and every turn shows up in Studio and in LangSmith
exactly as it does for Studio's own chat panel.

Documentation:
  LangGraph SDK (Python): https://docs.langchain.com/langgraph-platform/python-sdk
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from dotenv import load_dotenv
from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient

from agent import GREETING

# override=True so .env always wins over pre-existing OS environment
# variables, which keeps a student's shell from silently shadowing their .env.
load_dotenv(override=True)

#: Graph id from ``langgraph.json``. The server auto-creates an assistant with
#: this id, so the UI never has to manage assistants of its own.
ASSISTANT_ID = "pizza_agent"


def create_client(deployment_url: str) -> LangGraphClient:
    """Create a LangGraph client pointed at the server hosting this UI."""
    return get_client(url=deployment_url)


async def create_thread(client: LangGraphClient) -> str:
    """Create a thread and seed Sal's greeting as its first message.

    The greeting is a fixed string rather than a model call (see ``agent.py``),
    but it still belongs *in* the thread: writing it to state means the model
    starts the first real turn on the same conversational footing the customer
    sees, and a page reload replays it from the server instead of from a
    browser-side special case.
    """
    # graph_id up front: state reads and writes on a thread that has never had a
    # run are rejected without one, and the greeting is written before any run.
    thread = await client.threads.create(graph_id=ASSISTANT_ID)
    thread_id = thread["thread_id"]
    await client.threads.update_state(
        thread_id,
        {"messages": [{"role": "ai", "content": GREETING, "id": str(uuid.uuid4())}]},
    )
    return thread_id


def _text_of(content: Any) -> str:
    """Flatten message content to plain text; content may be a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


async def get_history(client: LangGraphClient, thread_id: str) -> list[dict[str, str]]:
    """Return the customer-visible transcript for a thread.

    Tool calls and tool results are dropped, along with the contentless assistant
    messages that carry them - the customer sees prose only.
    """
    state = await client.threads.get_state(thread_id)
    messages = (state.get("values") or {}).get("messages") or []

    transcript = []
    for message in messages:
        role = message.get("type") or message.get("role")
        if role not in {"human", "ai", "user", "assistant"}:
            continue  # tool results, system prompts
        text = _text_of(message.get("content"))
        if not text.strip():
            continue  # assistant turn that was nothing but tool calls
        speaker = "user" if role in {"human", "user"} else "assistant"
        transcript.append({"role": speaker, "content": text})
    return transcript


async def stream_response(
    client: LangGraphClient,
    thread_id: str,
    message: str,
) -> AsyncGenerator[str, None]:
    """Stream one turn of the agent as text tokens.

    Yields only the assistant's prose. Chunks that carry tool-call arguments have
    empty content and fall out here, which is what keeps tool calls out of the
    transcript without the front end having to filter anything.
    """
    async for event in client.runs.stream(
        thread_id,
        ASSISTANT_ID,
        input={"messages": [{"role": "user", "content": message}]},
        stream_mode="messages-tuple",
    ):
        if event.event == "error":
            raise RuntimeError(str(event.data))
        if event.event != "messages":
            continue
        chunk, _metadata = event.data
        if chunk.get("type") != "AIMessageChunk":
            continue
        text = _text_of(chunk.get("content"))
        if text:
            yield text
