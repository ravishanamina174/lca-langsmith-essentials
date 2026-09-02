"""LangSlice pizzeria ordering agent, built on the Claude Agent SDK.

The sibling ``langgraph-agent/`` project is the same pizzeria on a different
harness. Comparing the two files side by side is the point: the shop, the
prompt, the nine tools, the database, and the two manufactured bugs are the
same, and everything about how the agent loop runs is different.

    langgraph-agent/    LangChain create_agent + LangGraph, model via OpenAI
    claude-sdk-agent/   Claude Agent SDK - Claude Code as a library

This project imports no LangChain and no LangGraph. Its only dependencies are
``claude-agent-sdk``, ``langsmith``, and ``python-dotenv``, which is checkable:

    uv pip list | grep langchain     # returns nothing

That is what it is here to demonstrate. LangSmith is a tracing backend, not a
LangChain feature: the run tree in this module is assembled by hand out of
``langsmith`` primitives, and it lands in the same project, next to the
LangGraph agent's traces, in the same shape.

How the pieces fit:

    PizzeriaSession.send() -> claude CLI subprocess (agent loop, model calls)
                                  |
                                  v
                          in-process MCP server -> tool functions -> database
                                  |
                                  v
                          LangSmith run tree (built here)

Nothing reports the turn or the model calls, because the agent loop runs in a
subprocess - so this module opens a chain run per customer turn, an ``llm`` run
per assistant turn, and a ``tool`` run per tool call. Tool arguments are
validated by the SDK before a handler is reached: ``create_sdk_mcp_server``
runs ``jsonschema.validate`` against the schema below and returns an error
result the model can read, so no validation layer is needed here.

Tool-call spans stay out of the terminal UI - the split the course relies on,
where prose reaches the customer and tool traffic reaches LangSmith.

Environment:
    PIZZA_AGENT_CLAUDE_MODEL    default "claude-opus-5"
    PIZZA_AGENT_CLAUDE_EFFORT   low | medium | high | xhigh | max, default "low"

Auth comes from an existing ``claude`` CLI login or ``ANTHROPIC_API_KEY``,
resolved the way Claude Code resolves it, so a logged-in machine needs no key
in ``.env``. No OpenAI key is involved.
"""

from __future__ import annotations

import inspect
import json
import os
from typing import Any, Literal, Self, get_type_hints

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SdkMcpTool,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)
from dotenv import load_dotenv
from langsmith import trace, traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import create_model

import database as db

# override=True so .env always wins over pre-existing OS environment
# variables, which keeps a student's shell from silently shadowing their .env.
# The file lives at the repo root, shared with the sibling project; dotenv
# walks up from this directory to find it.
load_dotenv(override=True)

MODEL_NAME = os.getenv("PIZZA_AGENT_CLAUDE_MODEL", "claude-opus-5")
EFFORT = os.getenv("PIZZA_AGENT_CLAUDE_EFFORT", "low")

#: Key under which the pizzeria server is registered in ``mcp_servers``. It -
#: not the server's own ``name`` - is what appears in fully qualified tool
#: names, so the model sees ``mcp__pizzeria__add_pizza_to_order``.
SERVER_KEY = "pizzeria"

#: Stops a stuck conversation from looping forever on the shop's dime. Nine
#: tools and a chatty customer do not need more than this.
MAX_TURNS = 24

SYSTEM_PROMPT = """You are Sal, the ordering assistant for LangSlice, a pizzeria in Brooklyn, NY.
You chat with customers to answer questions about the shop and to take their pizza orders.

## Answering questions

Use `search_company_info` for anything about the business itself: address, hours,
phone number, delivery radius and fees, payment methods, allergens, catering,
rewards. Do not answer these from memory - look them up, then answer in your own
words. Use `get_menu` for sizes, crusts, specialty pizzas, toppings, and sides.
When the customer asks to see the full menu, return the tool's `display_markdown`
verbatim. It is deliberately formatted to render cleanly in both the chat UI and
a plain terminal. For a question about only one part of the menu, answer just
that part instead of printing the full menu.

## Recommending something

When a customer asks what you would recommend, call `get_menu` and recommend from
what is actually on it. Describe a specialty pizza only by the toppings the menu
lists for it - never imply a pizza comes with something it does not. If they
mentioned a topping they like, lead with the specialty pizzas that already have
it; if none do, say so plainly before offering to add it to one or to build them
a custom pizza.

## Taking an order

1. Call `start_order` once you know the customer's name and whether it is pickup
   or delivery (delivery also needs a street address).
2. Call `add_pizza_to_order` for each pizza. For a named specialty pizza, pass
   its menu name in `specialty_pizza`; do not reconstruct it from the marketing
   description. It is the kitchen's system of
   record: if it returns `status: "added"`, the pizza is on the order and you
   should confirm it to the customer. If it rejects the pizza, explain what was
   wrong and offer alternatives.
3. If the customer wants to change a pizza that is already on the order -
   "actually no olives on the first one", "can you make that second one gluten
   free", "make it a large instead" - call `modify_pizza` with that pizza's
   number and the changes. Do not add a new pizza for a change request.
4. Call `add_side` for sides and drinks, `remove_pizza` to drop a pizza, and
   `view_order` if the customer asks what is on their order or what it costs.
5. When the customer signals they are done - "that's it", "that's all", "that's
   my order", "go ahead and place it", "place the order", "confirm it", "please
   confirm", "put it through", "send it" - call `confirm_order` right away. Do
   not ask whether they want anything else first; they already told you. Then
   tell them their order number and the pickup or delivery time and close out
   warmly. The conversation is over at that point, so do not end with a
   question.
6. If `confirm_order` rejects the order, tell the customer plainly what it says
   and what would clear it. Do not call it again until something has actually
   changed on the order.

Whenever you read a total back on a delivery order, name the $3.99 delivery fee as
part of it, or say it was waived because the order is over $35, so the number is
never a surprise.

Keep tool calls to a minimum: the order tools are the system of record and their
responses already tell you the result, so do not call `view_order` to re-check
work you just did. Every extra call is latency the customer feels.

## Style

Be warm, brief, and conversational - this is a chat window, not an essay.
Ask one question at a time, and only when you actually need the answer to keep
going - never re-ask something the customer has already settled. Confirm each
pizza as you add it. Never invent menu items, prices, or store policies; if a
tool does not give you the answer, say you will check with the kitchen and offer
the phone number."""

# Opening line, so the customer never has to start the conversation. It is a
# fixed string rather than a model call: the greeting is the same every time,
# and generating it would put a contentless run at the head of every trace.
# Front ends seed it into the transcript as the first assistant message, which
# also gives the model the same conversational footing the customer sees.
GREETING = (
    "Welcome to LangSlice! I'm Sal, and I can help you build an order or answer "
    "anything about the shop — hours, delivery, what's on the menu. "
    "What are you in the mood for?"
)


# ---------------------------------------------------------------------------
# Tools
#
# Plain typed functions. The type hints are the tool's contract - the schema
# the model sees is derived from them below - and the docstring is the
# description it reads, so both are load-bearing rather than decorative.
#
# ``thread_id`` keys the order in ``database.ORDERS``. It is the first
# parameter of every order tool but is never exposed to the model: the session
# supplies it, because letting the model pass one would let it address another
# customer's order.
# ---------------------------------------------------------------------------


def search_company_info(thread_id: str, query: str) -> dict[str, Any]:
    """Search LangSlice company information.

    Covers the address and directions, store hours, phone and email, delivery
    policy (radius, fees, minimums, timing), payment options, allergen and
    dietary information, the loyalty program, and catering.

    Args:
        query: What the customer wants to know, e.g. "how late are you open"
            or "do you deliver to Greenpoint".
    """
    results = db.search_company_info(query)
    if not results:
        return {
            "query": query,
            "results": [],
            "note": "No article matched. Offer the store phone number instead.",
        }
    return {"query": query, "results": results}


def get_menu(thread_id: str) -> dict[str, Any]:
    """Get the full LangSlice menu.

    Returns pizza sizes with prices, crust styles and their upcharges, the
    specialty pizzas and what is on them, the topping list with per-topping
    prices, sides, and a ``display_markdown`` version of the full menu. When a
    customer asks to see the menu, reproduce ``display_markdown`` verbatim.
    """
    return db.get_menu()


def start_order(
    thread_id: str,
    customer_name: str,
    order_type: Literal["pickup", "delivery"],
    address: str | None = None,
) -> dict[str, Any]:
    """Start a new order for this customer.

    Call this once, before adding any pizzas.

    Args:
        customer_name: Name to put on the order.
        order_type: Either "pickup" or "delivery".
        address: Street address, required for delivery orders.
    """
    if order_type == "delivery" and not address:
        return {
            "status": "rejected",
            "reason": "missing_address",
            "message": "Delivery orders need a street address.",
        }

    existing = db.get_order(thread_id)
    if existing and existing["status"] == "open":
        return {
            "status": "already_open",
            "message": f"Order {existing['order_id']} is already open for this customer.",
            "order": db.order_summary(existing),
        }

    order = db.new_order(thread_id, customer_name, order_type, address)
    return {"status": "started", "order": db.order_summary(order)}


def add_pizza_to_order(
    thread_id: str,
    size: Literal["small", "medium", "large"],
    crust: str,
    toppings: list[str] | None = None,
    specialty_pizza: str | None = None,
    quantity: int = 1,
    notes: str | None = None,
    no_cheese: bool = False,
) -> dict[str, Any]:
    """Add a pizza to the open order.

    Validates the size, the crust (including which sizes that crust comes in),
    and every topping against the kitchen inventory system before adding the
    pizza. Returns `status: "added"` with the priced line item when the pizza
    can be made, or `status: "rejected"` with the reason when it cannot. The
    line item's `price_lines` already itemize the size, crust, and toppings, so
    a price question about a pizza you just added needs no further call.

    Every pizza is built with mozzarella unless `no_cheese` is set, so you do not
    need to list it: pass only what the customer asked for and the returned line
    item shows exactly what the kitchen will make.

    Args:
        size: "small", "medium", or "large".
        crust: Crust style, e.g. "hand tossed", "thin", "deep dish",
            "gluten free".
        toppings: Toppings for a custom pizza, or extra toppings to add to a
            specialty pizza. Use "extra cheese" when they want more cheese than
            standard, or "vegan cheese" for a dairy-free pizza.
        specialty_pizza: Exact menu name of a specialty pizza, e.g.
            "the monitoring margherita". Its canonical toppings are filled in by the
            kitchen system; do not copy ingredients from its description.
        quantity: How many of this exact pizza. Defaults to 1.
        notes: Special instructions, e.g. "well done", "light sauce".
        no_cheese: Set when the customer asked for no mozzarella. Pair it with
            "vegan cheese" in `toppings` if they want a dairy-free cheese
            instead.
    """
    order = db.get_order(thread_id)

    if order is None:
        return {
            "status": "rejected",
            "reason": "no_open_order",
            "message": "Call start_order before adding pizzas.",
        }
    if order["status"] != "open":
        return {
            "status": "rejected",
            "reason": "order_closed",
            "message": f"Order {order['order_id']} is already {order['status']}.",
        }

    crust_record = db.get_crust(crust)
    if crust_record is None:
        return {
            "status": "rejected",
            "reason": "unknown_crust",
            "message": f"We don't have a {crust} crust.",
            "available_crusts": sorted(db.CRUSTS),
        }
    if size not in crust_record["sizes"]:
        return {
            "status": "rejected",
            "reason": "crust_size_unavailable",
            "message": f"{crust_record['name'].title()} crust doesn't come in {size}.",
            "available_sizes": crust_record["sizes"],
        }

    requested_toppings = list(toppings or [])
    specialty_name = None
    if specialty_pizza:
        specialty = db.get_specialty_pizza(specialty_pizza)
        if specialty is None:
            return {
                "status": "rejected",
                "reason": "unknown_specialty_pizza",
                "message": f"We don't have a specialty pizza called {specialty_pizza}.",
                "available_specialty_pizzas": sorted(db.SPECIALTY_PIZZAS),
            }
        specialty_name = specialty["name"]
        requested_toppings = list(specialty["toppings"]) + requested_toppings

    # Resolve every requested topping against the ingredient catalog.
    resolved: list[dict[str, Any]] = []
    unknown: list[str] = []

    #                     FIX
    ##################################################
    # out_of_stock: list[str] = []
    ##################################################

    for requested in requested_toppings:
        ingredient = db.get_ingredient(requested)
        if ingredient is None:
            unknown.append(requested)

        #                     FIX
        ##################################################
        # elif ingredient["stock_units"] <= 0:
        #     out_of_stock.append(requested)
        ##################################################

        else:
            resolved.append(ingredient)

    if unknown:
        return {
            "status": "rejected",
            "reason": "unknown_topping",
            "message": f"We don't carry: {', '.join(unknown)}.",
            "unknown_toppings": unknown,
            "available_toppings": [t["name"] for t in db.list_toppings(in_stock_only=False)],
        }

    #                     FIX
    ##################################################
    # if out_of_stock:
    #     return {
    #         "status": "rejected",
    #         "reason": "out_of_stock",
    #         "message": f"We're out of: {', '.join(out_of_stock)}.",
    #         "out_of_stock_toppings": out_of_stock,
    #         "available_toppings": [t["name"] for t in db.list_toppings()],
    #     }
    ##################################################

    # The kitchen puts mozzarella on every pizza unless the customer opts out.
    # Normalize it here rather than relying on the caller to list it, so the
    # ticket always matches what actually gets baked.
    # "vegan cheese" replaces the mozzarella; "extra cheese" and accent cheeses
    # like feta go on top of it, so they do not suppress the base layer.
    if no_cheese:
        resolved = [i for i in resolved if i["name"] not in {"mozzarella", "extra cheese"}]
    elif not any(i["name"] in {"mozzarella", "vegan cheese"} for i in resolved):
        resolved.insert(0, db.get_ingredient("mozzarella"))

    quantity = max(1, int(quantity))
    pizza = db.build_pizza(
        size=size,
        crust=crust_record["name"],
        toppings=[i["name"] for i in resolved],
        quantity=quantity,
        notes=notes,
    )
    pizza["specialty_pizza"] = specialty_name
    order["pizzas"].append(pizza)
    db.renumber_pizzas(order)

    return {"status": "added", "pizza": pizza, "order": db.order_summary(order)}


def modify_pizza(
    thread_id: str,
    pizza_number: int,
    size: Literal["small", "medium", "large"] | None = None,
    crust: str | None = None,
    add_toppings: list[str] | None = None,
    remove_toppings: list[str] | None = None,
    quantity: int | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Change a pizza that is already on the order.

    Use this whenever the customer revises a pizza they already asked for -
    swapping a size or crust, adding or removing a topping, changing how many
    they want, or adding a special instruction. Only pass the fields that are
    changing.

    Args:
        pizza_number: Which pizza on the order to change (1 for the first).
        size: New size, if it is changing.
        crust: New crust, if it is changing.
        add_toppings: Toppings to put on.
        remove_toppings: Toppings to take off.
        quantity: New quantity.
        notes: New special instructions.
    """
    order = db.get_order(thread_id)

    if order is None:
        return {
            "status": "rejected",
            "reason": "no_open_order",
            "message": "There is no order to modify yet.",
        }
    if order["status"] != "open":
        return {
            "status": "rejected",
            "reason": "order_closed",
            "message": f"Order {order['order_id']} is already {order['status']}.",
        }
    if not 1 <= pizza_number <= len(order["pizzas"]):
        return {
            "status": "rejected",
            "reason": "no_such_pizza",
            "message": f"There is no pizza #{pizza_number} on this order.",
            "order": db.order_summary(order),
        }

    original = order["pizzas"][pizza_number - 1]
    revised = db.copy_pizza(original)

    if size is not None:
        revised["size"] = size
    if crust is not None:
        crust_record = db.get_crust(crust)
        if crust_record is None:
            return {
                "status": "rejected",
                "reason": "unknown_crust",
                "message": f"We don't have a {crust} crust.",
                "available_crusts": sorted(db.CRUSTS),
            }
        if revised["size"] not in crust_record["sizes"]:
            return {
                "status": "rejected",
                "reason": "crust_size_unavailable",
                "message": (
                    f"{crust_record['name'].title()} crust doesn't come in "
                    f"{revised['size']}."
                ),
                "available_sizes": crust_record["sizes"],
            }
        revised["crust"] = crust_record["name"]

    for requested in remove_toppings or []:
        ingredient = db.get_ingredient(requested)
        name = ingredient["name"] if ingredient else db._normalize(requested)
        revised["toppings"] = [t for t in revised["toppings"] if t != name]

    for requested in add_toppings or []:
        ingredient = db.get_ingredient(requested)
        if ingredient is None:
            return {
                "status": "rejected",
                "reason": "unknown_topping",
                "message": f"We don't carry: {requested}.",
                "available_toppings": [t["name"] for t in db.list_toppings(in_stock_only=False)],
            }
        if ingredient["name"] not in revised["toppings"]:
            revised["toppings"].append(ingredient["name"])

    if quantity is not None:
        revised["quantity"] = max(1, int(quantity))
    if notes is not None:
        revised["notes"] = notes

    # Reprice from one call, so the itemized lines cannot describe a different
    # pizza than the unit price does.
    pricing = db.price_breakdown(revised["size"], revised["crust"], revised["toppings"])
    revised["unit_price"] = pricing["unit_price"]
    revised["price_lines"] = pricing["lines"]

    order["pizzas"][pizza_number - 1] = revised
    db.renumber_pizzas(order)

    return {
        "status": "modified",
        "pizza_number": pizza_number,
        "pizza": {k: v for k, v in revised.items() if k != "pizza_number"},
        "order": db.order_summary(order),
    }


def remove_pizza(thread_id: str, pizza_number: int) -> dict[str, Any]:
    """Take a pizza off the order.

    Args:
        pizza_number: Which pizza to remove (1 for the first).
    """
    order = db.get_order(thread_id)

    if order is None:
        return {
            "status": "rejected",
            "reason": "no_open_order",
            "message": "There is no order to change yet.",
        }
    if order["status"] != "open":
        return {
            "status": "rejected",
            "reason": "order_closed",
            "message": f"Order {order['order_id']} is already {order['status']}.",
        }
    if not 1 <= pizza_number <= len(order["pizzas"]):
        return {
            "status": "rejected",
            "reason": "no_such_pizza",
            "message": f"There is no pizza #{pizza_number} on this order.",
            "order": db.order_summary(order),
        }

    removed = order["pizzas"].pop(pizza_number - 1)
    db.renumber_pizzas(order)
    return {"status": "removed", "removed": removed, "order": db.order_summary(order)}


def add_side(thread_id: str, item: str, quantity: int = 1) -> dict[str, Any]:
    """Add a side or drink to the order.

    Args:
        item: Side name as it appears on the menu, e.g. "garlic knots (6)".
        quantity: How many. Defaults to 1.
    """
    order = db.get_order(thread_id)

    if order is None:
        return {
            "status": "rejected",
            "reason": "no_open_order",
            "message": "Call start_order before adding sides.",
        }
    if order["status"] != "open":
        return {
            "status": "rejected",
            "reason": "order_closed",
            "message": f"Order {order['order_id']} is already {order['status']}.",
        }

    key = db._normalize(item)
    match = next((name for name in db.SIDES if db._normalize(name) == key), None)
    if match is None:
        match = next((name for name in db.SIDES if key and key in db._normalize(name)), None)
    if match is None:
        return {
            "status": "rejected",
            "reason": "unknown_side",
            "message": f"We don't have {item} on the menu.",
            "available_sides": sorted(db.SIDES),
        }

    order["sides"].append(
        {"item": match, "quantity": max(1, int(quantity)), "price": db.SIDES[match]}
    )
    return {"status": "added", "side": match, "order": db.order_summary(order)}


def view_order(thread_id: str) -> dict[str, Any]:
    """Get the current state of the order, with line items and totals.

    Each pizza carries `unit_price_lines`, which itemizes what it costs: the
    size's base price, the crust upcharge, and every topping with its own
    upcharge. Read a price breakdown straight off those lines rather than
    working one out from the menu. They add up to `unit_price`, the price of a
    single pizza, so on a line with a quantity above one they explain
    `line_total` only after multiplying.
    """
    order = db.get_order(thread_id)
    if order is None:
        return {"status": "no_order", "message": "No order has been started yet."}
    return {"status": "ok", "order": db.order_summary(order)}


def confirm_order(thread_id: str) -> dict[str, Any]:
    """Send the order to the kitchen once the customer has approved it.

    Returns `status: "confirmed"` with the order number and ETA, or
    `status: "rejected"` when the order cannot be sent - a delivery order under
    the shop's order minimum comes back `below_delivery_minimum` with how much
    more the order needs.
    """
    order = db.get_order(thread_id)

    if order is None:
        return {
            "status": "rejected",
            "reason": "no_open_order",
            "message": "There is no order to confirm.",
        }
    if not order["pizzas"] and not order["sides"]:
        return {
            "status": "rejected",
            "reason": "empty_order",
            "message": "The order is empty.",
        }
    if order["status"] == "confirmed":
        return {
            "status": "already_confirmed",
            "order": db.order_summary(order),
        }

    # Delivery orders have to clear the shop's order minimum before the driver
    # will take them out.
    if order["order_type"] == "delivery":
        subtotal = float(sum(side["price"] * side["quantity"] for side in order["sides"]))

        #                     FIX
        ##################################################
        # subtotal = float(db.order_summary(order)["subtotal"])
        ##################################################

        if subtotal < db.DELIVERY_MINIMUM:
            short_by = round(db.DELIVERY_MINIMUM - subtotal, 2)
            return {
                "status": "rejected",
                "reason": "below_delivery_minimum",
                "minimum": db.DELIVERY_MINIMUM,
                "order_subtotal": round(subtotal, 2),
                "short_by": short_by,
                "message": (
                    f"This order is ${short_by:.2f} under the "
                    f"${db.DELIVERY_MINIMUM:.2f} delivery minimum."
                ),
            }

    order["status"] = "confirmed"
    summary = db.order_summary(order)
    eta = "35-50 minutes" if order["order_type"] == "delivery" else "15-20 minutes"
    return {"status": "confirmed", "eta": eta, "order": summary}


#: Every tool the agent gets, in the order the model sees them.
TOOLS = [
    search_company_info,
    get_menu,
    start_order,
    add_pizza_to_order,
    modify_pizza,
    remove_pizza,
    add_side,
    view_order,
    confirm_order,
]


# ---------------------------------------------------------------------------
# Tool function -> MCP tool
# ---------------------------------------------------------------------------

#: Parameters the session supplies rather than the model. Excluded from the
#: schema, so they are neither advertised to the model nor accepted from it.
_INJECTED = ("thread_id",)


def _json_schema(fn: Any) -> dict[str, Any]:
    """Derive a JSON Schema for ``fn``'s model-facing parameters.

    Pydantic does the work of turning annotations into schema - ``Literal``
    becomes an ``enum``, ``X | None`` with a default becomes an optional field -
    which is what makes the plain functions above a sufficient tool contract on
    their own. ``get_type_hints`` rather than raw ``__annotations__`` because
    ``from __future__ import annotations`` leaves these as strings.
    """
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name in _INJECTED:
            continue
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (hints.get(name, Any), default)

    schema = create_model(fn.__name__, **fields).model_json_schema()
    # The name and docstring are carried by the MCP tool itself; leaving them in
    # the schema repeats the whole docstring inside the parameter block.
    schema.pop("title", None)
    schema.pop("description", None)
    return schema


def _mcp_tool(fn: Any, thread_id: str, session: PizzeriaSession) -> SdkMcpTool[Any]:
    """Wrap one tool function as an SDK MCP tool for one conversation."""

    @tool(fn.__name__, inspect.getdoc(fn) or "", _json_schema(fn))
    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        # The SDK has already validated `args` against the schema by this point,
        # so a handler only has to deal with real failures.
        #
        # run_type="tool" and an explicit parent: nothing else traces this call,
        # and the parent has to be named rather than inherited because the MCP
        # server may run the handler on a task the traced turn did not create.
        @traceable(run_type="tool", name=fn.__name__)
        def _traced(**call_args: Any) -> Any:
            return fn(thread_id, **call_args)

        try:
            result = _traced(**args, langsmith_extra={"parent": session.current_run})
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not raised
            # Composed rather than raised: an uncaught exception reaches Claude
            # as a bare str(exc) with no context. is_error marks it a failed
            # call rather than odd-looking data.
            session.record_tool_call(fn.__name__, args, error=str(exc))
            return {
                "content": [{"type": "text", "text": f"{fn.__name__} failed: {exc}"}],
                "is_error": True,
            }

        # A rejection ("out of stock", "under the delivery minimum") is a real
        # business answer the agent has to explain to the customer, not a tool
        # failure - so it goes back as an ordinary result.
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        session.record_tool_call(fn.__name__, args, result=text)
        return {"content": [{"type": "text", "text": text}]}

    return _handler


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class PizzeriaSession:
    """One customer conversation.

    Wraps a ``ClaudeSDKClient``, which keeps the conversation on one CLI session
    so successive turns share context without this module resending history.

    One session owns one order: ``thread_id`` keys ``database.ORDERS`` and is
    bound into the tools at construction time.

    Use it as an async context manager::

        async with PizzeriaSession(thread_id) as session:
            print(await session.send("do you deliver to greenpoint?"))
    """

    def __init__(self, thread_id: str, *, model: str | None = None) -> None:
        self.thread_id = thread_id
        self.model = model or MODEL_NAME

        #: Parent run for the turn in flight, so tool handlers can attach to it.
        self.current_run: Any = None
        #: Tool traffic for the turn in flight, for ``--show-tools`` and eval.
        self.tool_calls: list[dict[str, Any]] = []
        #: Cost and token totals reported by the last completed turn.
        self.last_usage: dict[str, Any] = {}

        pizzeria = create_sdk_mcp_server(
            name="langslice-pizzeria",
            version="1.0.0",
            tools=[_mcp_tool(fn, thread_id, self) for fn in TOOLS],
        )

        self._options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={SERVER_KEY: pizzeria},
            # Pre-approve every pizzeria tool. There is no human at a terminal
            # to answer a permission prompt.
            allowed_tools=[f"mcp__{SERVER_KEY}__*"],
            # Strip every built-in Claude Code tool. Without this the pizza
            # agent inherits Read, Write, Bash, and WebSearch - a customer-
            # facing chat window with shell access on the shop's laptop. An
            # empty list leaves it the MCP tools and nothing else.
            tools=[],
            # Anything that somehow escapes the two settings above is denied
            # rather than prompted for.
            permission_mode="dontAsk",
            # Ignore ~/.claude, .claude/, and any CLAUDE.md. Those describe
            # building the course demo; injecting them into a pizzeria agent's
            # context is the kind of surprise that makes a demo irreproducible
            # on someone else's machine.
            setting_sources=[],
            # Likewise ignore .mcp.json and any user- or plugin-level MCP
            # servers, so the tool list is only what is constructed above.
            strict_mcp_config=True,
            effort=EFFORT,
            max_turns=MAX_TURNS,
            # cwd is set explicitly because the SDK otherwise inherits the
            # caller's, and session records are filed per directory.
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        self._client: ClaudeSDKClient | None = None

    async def __aenter__(self) -> Self:
        self._client = ClaudeSDKClient(options=self._options)
        await self._client.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    def record_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        """Note a tool call for the turn in flight."""
        self.tool_calls.append(
            {"name": name, "args": args, "result": result, "error": error}
        )

    async def send(self, text: str) -> str:
        """Send one customer message and return Sal's reply."""
        if self._client is None:
            msg = "PizzeriaSession must be used as an async context manager."
            raise RuntimeError(msg)

        self.tool_calls = []
        return await self._turn(text)

    @traceable(run_type="chain", name="langslice_pizza_agent")
    async def _turn(self, text: str) -> str:
        """One customer turn, traced as the root of the run tree."""
        assert self._client is not None
        self.current_run = get_current_run_tree()

        await self._client.query(text, session_id=self.thread_id)

        replies: list[str] = []
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                replies.append(self._log_assistant_turn(message))
            elif isinstance(message, ResultMessage):
                self._log_result(message)

        # Only the last assistant turn is the reply to the customer; earlier
        # ones are the model narrating its way through tool calls.
        return next((r for r in reversed(replies) if r), "")

    def _log_assistant_turn(self, message: AssistantMessage) -> str:
        """Record one assistant turn as an ``llm`` child run; return its text."""
        text = "\n".join(
            b.text for b in message.content if isinstance(b, TextBlock) and b.text
        ).strip()
        tool_calls = [
            {"name": b.name, "args": b.input}
            for b in message.content
            if isinstance(b, ToolUseBlock)
        ]

        # The prompt itself is assembled inside the CLI and never crosses back
        # over, so inputs name the model rather than pretend to hold messages.
        with trace(
            name="claude",
            run_type="llm",
            parent=self.current_run,
            inputs={"model": message.model},
            metadata={
                "model": message.model,
                "stop_reason": message.stop_reason,
                "harness": "claude-agent-sdk",
            },
        ) as run:
            run.end(
                outputs={
                    "text": text,
                    "tool_calls": tool_calls,
                    "usage": message.usage or {},
                }
            )
        return text

    def _log_result(self, message: ResultMessage) -> None:
        """Attach the turn's cost and token totals to the root run."""
        self.last_usage = {
            "num_turns": message.num_turns,
            "duration_ms": message.duration_ms,
            "total_cost_usd": message.total_cost_usd,
            "usage": message.usage or {},
        }
        if self.current_run is not None:
            # Reported by the SDK, not derived from LangSmith's price table -
            # LangSmith cannot price a run whose model call it never saw.
            self.current_run.add_metadata(
                {
                    "claude_sdk_session_id": message.session_id,
                    "num_turns": message.num_turns,
                    "total_cost_usd": message.total_cost_usd,
                    "stop_reason": message.stop_reason,
                    "harness": "claude-agent-sdk",
                }
            )
        if message.is_error:
            detail = ", ".join(message.errors or []) or message.subtype
            msg = f"Claude Agent SDK returned an error result: {detail}"
            raise RuntimeError(msg)
