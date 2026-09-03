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


def customer_turns(inputs: dict) -> list[str]:
    """Pull the customer's messages out of one dataset example.

    Three input shapes are all legitimate, because a dataset's shape follows
    from whatever produced it:

    * ``messages`` with ``type=human`` - built from a LangGraph trace, whose
      root run's inputs are the message list.
    * ``messages`` with ``role=user`` - hand-authored, in the OpenAI shape.
    * ``prompt`` - built from *this* agent's traces. The LangSmith integration
      gives each root run ``inputs={"prompt", "system"}``, one customer turn
      per run, so such a dataset has no ``messages`` key at all.

    Raises on a shape it does not recognize rather than returning nothing. A
    target that quietly replays zero turns still returns a transcript, and a
    transcript with no order scores 1 in ``stock_evaluator`` - so a silent
    mismatch reads as a clean sweep in seconds rather than as a failure.
    """
    turns = [
        message["content"]
        for message in inputs.get("messages") or []
        if (message.get("role") or message.get("type")) in {"user", "human"}
    ]
    if not turns and inputs.get("prompt"):
        turns = [inputs["prompt"]]
    if not turns:
        msg = f"No customer message in example inputs (keys: {sorted(inputs)})."
        raise ValueError(msg)
    return turns


async def evaluation_target(inputs: dict) -> dict[str, Any]:
    """Replay one dataset example's customer messages through the agent.

    Returns a transcript in the plain-dict message shape the shared evaluator
    reads, with one ``tool`` entry per tool call the agent made.
    """
    thread_id = f"eval-claude-{uuid.uuid4()}"
    messages: list[dict[str, Any]] = [{"role": "assistant", "content": GREETING}]

    async with PizzeriaSession(thread_id) as session:
        for turn in customer_turns(inputs):
            messages.append({"role": "user", "content": turn})
            reply = await session.send(turn)

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
