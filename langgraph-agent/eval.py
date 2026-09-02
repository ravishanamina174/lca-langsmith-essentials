"""Run the pizza agent and stock evaluator against a LangSmith dataset."""

import os
import uuid

from langchain.messages import AIMessage, HumanMessage
from langsmith import evaluate

from agent import GREETING, pizza_agent
from stock_evaluator import ingredient_stock_evaluator


def evaluation_target(inputs: dict) -> dict:
    history = [AIMessage(content=GREETING)]
    thread_id = f"eval-{uuid.uuid4()}"

    for message in inputs.get("messages", []):
        # Examples created from LangSmith traces use LangChain's serialized
        # message shape (``type=human``), while hand-authored examples often
        # use the OpenAI shape (``role=user``). Accept both.
        message_type = message.get("role") or message.get("type")
        if message_type not in {"user", "human"}:
            continue

        history.append(HumanMessage(content=message["content"]))
        result = pizza_agent.invoke(
            {"messages": history},
            config={"configurable": {"thread_id": thread_id}},
        )
        history = result["messages"]

    return {"messages": history}


def main() -> None:
    dataset = os.environ["DATASET_NAME"]

    evaluate(
        evaluation_target,
        data=dataset,
        evaluators=[ingredient_stock_evaluator],
        experiment_prefix="stock-fixed",
    )


if __name__ == "__main__":
    main()
