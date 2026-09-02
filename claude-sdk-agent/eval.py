"""Run the Claude Agent SDK agent and the stock evaluator against a dataset.

The sibling ``langgraph-agent/eval.py`` does the same job for the LangGraph
build, and both use the *same* ``stock_evaluator.py`` - copied between the two
projects but never edited. That is the interesting part: one evaluator scores
two agents that share no framework, because it reads plain dictionaries.

    uv run eval.py

``DATASET_NAME`` selects the dataset, the same as the LangGraph project.

Two things differ from the LangGraph version, both consequences of the harness:

* Each example needs its own ``PizzeriaSession``, and every session spawns a
  ``claude`` CLI subprocess - so this uses ``aevaluate`` and caps concurrency.
  ``EVAL_CONCURRENCY`` overrides the cap.
* It costs real money. Turns measured during development ran $0.03-$0.10 each
  on Claude Opus 5, and an example is several turns, so a large dataset is a
  bill worth estimating before you start. ``PIZZA_AGENT_CLAUDE_EFFORT=low``
  (the default) and a smaller model keep it down.

The output shape is what lets the shared evaluator work. ``stock_evaluator``
walks ``outputs["messages"]`` looking for entries whose role is ``tool`` and
whose content parses as JSON containing an ``order``. Those are plain dicts in
the OpenAI message shape - not LangChain objects - so producing them here needs
no LangChain, and the evaluator does not have to know which agent ran.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

from langsmith import aevaluate

from agent import GREETING, PizzeriaSession
from stock_evaluator import ingredient_stock_evaluator

#: Each concurrent example is a CLI subprocess and a model conversation, so
#: this trades wall-clock against local resources and rate limits.
CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "2"))


async def evaluation_target(inputs: dict) -> dict[str, Any]:
    """Replay one dataset example's customer messages through the agent.

    Returns a transcript in the plain-dict message shape the shared evaluator
    reads, with one ``tool`` entry per tool call the agent made.
    """
    thread_id = f"eval-claude-{uuid.uuid4()}"
    messages: list[dict[str, Any]] = [{"role": "assistant", "content": GREETING}]

    async with PizzeriaSession(thread_id) as session:
        for message in inputs.get("messages", []):
            # Examples created from LangSmith traces use LangChain's serialized
            # message shape (``type=human``), while hand-authored examples often
            # use the OpenAI shape (``role=user``). Accept both - the dataset is
            # shared with the LangGraph project.
            message_type = message.get("role") or message.get("type")
            if message_type not in {"user", "human"}:
                continue

            messages.append({"role": "user", "content": message["content"]})
            reply = await session.send(message["content"])

            # session.tool_calls holds just this turn's traffic. Each result is
            # already the JSON string the model was shown, which is exactly what
            # the evaluator expects to json.loads().
            for call in session.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "name": call["name"],
                        "content": call["result"] or json.dumps({"error": call["error"]}),
                    }
                )
            messages.append({"role": "assistant", "content": reply})

    return {"messages": messages}


async def main() -> None:
    dataset = os.environ["DATASET_NAME"]

    await aevaluate(
        evaluation_target,
        data=dataset,
        evaluators=[ingredient_stock_evaluator],
        experiment_prefix="stock-baseline-claude-sdk",
        max_concurrency=CONCURRENCY,
        # Tagged so an experiment is attributable to a harness when both
        # projects' results sit in the same LangSmith project.
        metadata={"harness": "claude-agent-sdk"},
    )


if __name__ == "__main__":
    asyncio.run(main())
