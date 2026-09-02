"""Run the Claude Agent SDK pizza agent from the terminal.

The sibling ``langgraph-agent/run_agent.py`` is the same interface over the
LangGraph build. This project has no graph, so there is no LangGraph Studio
here - this script is how you talk to the agent.

Interactive chat:

    uv run run_agent.py

One-shot message:

    uv run run_agent.py -m "what time do you close on friday?"

Flags:
    --show-tools   print every tool call and result as it happens, plus the
                   turn's token and cost totals
    --thread ID    reuse a specific conversation/order id

Auth is whatever Claude Code itself uses - an existing ``claude`` CLI login, or
``ANTHROPIC_API_KEY``. No OpenAI key is involved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid

from claude_agent_sdk import ClaudeSDKError, CLINotFoundError

import database as db
from agent import EFFORT, GREETING, MODEL_NAME, PizzeriaSession


def _print_tool_traffic(session: PizzeriaSession) -> None:
    """Pretty-print the tool calls and results from the turn just finished."""
    for call in session.tool_calls:
        args = json.dumps(call["args"], ensure_ascii=False)
        print(f"  \033[36m-> {call['name']}({args})\033[0m")
        payload = call["error"] or call["result"] or ""
        if len(payload) > 700:
            payload = payload[:700] + " ...[truncated]"
        colour = "31" if call["error"] else "90"
        print(f"  \033[{colour}m<- {call['name']}: {payload}\033[0m")

    usage = session.last_usage
    if usage:
        cost = usage.get("total_cost_usd")
        tokens = usage.get("usage") or {}
        parts = [f"{usage.get('num_turns', 0)} model turns"]
        if tokens.get("input_tokens") is not None:
            parts.append(
                f"{tokens['input_tokens']} in / {tokens.get('output_tokens', 0)} out"
            )
        if cost is not None:
            parts.append(f"${cost:.4f}")
        print(f"  \033[90m[{', '.join(parts)}]\033[0m")


def _terminal_text(text: str) -> str:
    """Adapt rich assistant output for a terminal without changing the agent."""
    if "## 🍕 LangSlice Menu" in text:
        return db.format_menu_terminal()
    return text


def print_order(thread_id: str) -> None:
    """Print a readable receipt for the order this conversation built."""
    order = db.get_order(thread_id)
    if order is None:
        print("(no order was started)")
        return
    summary = db.order_summary(order)
    print(f"\n\033[1mOrder #{summary['order_id']}\033[0m")
    print("=" * (7 + len(summary["order_id"])))
    print(f"Customer: {summary['customer_name']}")
    print(f"Type:     {summary['order_type'].title()}")
    print(f"Status:   {summary['status'].title()}")
    if summary["address"]:
        print(f"Address:  {summary['address']}")

    if summary["pizzas"]:
        print("\n\033[1mPizzas\033[0m")
        for pizza in summary["pizzas"]:
            print(
                f"  #{pizza['pizza_number']}  {pizza['quantity']}x "
                f"{pizza['size'].title()} {pizza['crust'].title()}"
                f"  ${pizza['line_total']:.2f}"
            )
            toppings = ", ".join(name.title() for name in pizza["toppings"])
            print(f"      {toppings or 'No toppings'}")
            if pizza["notes"]:
                print(f"      Note: {pizza['notes']}")

    if summary["sides"]:
        print("\n\033[1mSides & Drinks\033[0m")
        for side in summary["sides"]:
            print(
                f"  {side['quantity']}x {side['item'].title()}"
                f"  ${side['line_total']:.2f}"
            )

    if not summary["pizzas"] and not summary["sides"]:
        print("\n  (order is empty)")

    print("\n\033[1mTotals\033[0m")
    print(f"  Subtotal       ${summary['subtotal']:.2f}")
    if summary["order_type"] == "delivery":
        print(f"  Delivery fee   ${summary['delivery_fee']:.2f}")
    print(f"  Tax            ${summary['tax']:.2f}")
    print(f"  \033[1mTotal          ${summary['total']:.2f}\033[0m\n")


async def send(session: PizzeriaSession, text: str, show_tools: bool) -> None:
    """Send one customer message through the agent and print the reply."""
    show_thinking = sys.stdout.isatty()
    if show_thinking:
        print("\033[1mSal:\033[0m Thinking...", end="", flush=True)
    try:
        reply = await session.send(text)
    finally:
        if show_thinking:
            # Return to the start and erase the placeholder before printing the
            # tool traffic or final response in its place.
            print("\r\033[2K", end="", flush=True)

    if show_tools:
        _print_tool_traffic(session)

    print(f"\033[1mSal:\033[0m {_terminal_text(reply)}\n")


async def chat(thread_id: str, message: str | None, show_tools: bool) -> int:
    """Open one session and drive it, one-shot or interactively."""
    async with PizzeriaSession(thread_id) as session:
        # Sal opens the conversation, so every mode starts from the same footing
        # the customer sees. It is a fixed string, not a model call.
        print(f"\033[1mSal:\033[0m {GREETING}\n")

        if message:
            await send(session, message, show_tools)
            return 0

        print("Type your message, or 'order' to view the current order, 'quit' to exit.\n")
        while True:
            try:
                # input() blocks the event loop, but nothing else is running on
                # it between turns, so a thread would buy nothing here.
                text = input("\033[1mYou:\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in {"quit", "exit", "q"}:
                break
            if text.lower() == "order":
                print_order(thread_id)
                continue
            await send(session, text, show_tools)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chat with the Claude Agent SDK build of the LangSlice pizza agent."
    )
    parser.add_argument("-m", "--message", help="send a single message and exit")
    parser.add_argument(
        "--show-tools", action="store_true", help="print tool calls and results"
    )
    parser.add_argument("--thread", help="conversation id (defaults to a fresh uuid)")
    args = parser.parse_args()

    thread_id = args.thread or str(uuid.uuid4())
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}

    print(
        f"\033[1mLangSlice\033[0m (claude agent sdk) - model {MODEL_NAME}, "
        f"effort {EFFORT}, thread {thread_id}"
    )
    if tracing:
        print(
            f"tracing to LangSmith project: {os.getenv('LANGSMITH_PROJECT', 'default')}"
        )
    print()

    try:
        return asyncio.run(chat(thread_id, args.message, args.show_tools))
    except CLINotFoundError:
        # Auth and the CLI are the two things that fail on a fresh machine, and
        # the SDK's own message does not mention either fix.
        print(
            "Could not find the claude CLI. Run `uv sync` in this directory, or "
            "install Claude Code.",
            file=sys.stderr,
        )
        return 1
    except ClaudeSDKError as exc:
        print(f"Claude Agent SDK error: {exc}", file=sys.stderr)
        print(
            "If this is an auth failure, run `claude` once to log in, or set "
            "ANTHROPIC_API_KEY in the repo-root .env.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
