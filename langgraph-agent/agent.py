"""LangSlice Pizzeria ordering agent.

A LangChain agent that answers questions about the pizzeria and takes pizza
orders. ``create_agent`` supplies the ReAct loop:

    __start__ -> model -> tools -> model -> ... -> __end__

Order state lives in ``database.ORDERS``, keyed by the conversation's
``thread_id``, which tools read off the injected ``ToolRuntime``.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool

import database as db

# override=True so .env always wins over pre-existing OS environment
# variables, which keeps a student's shell from silently shadowing their .env.
load_dotenv(override=True)

MODEL_NAME = os.getenv("PIZZA_AGENT_MODEL", "openai:gpt-5-nano")
REASONING_EFFORT = os.getenv("PIZZA_AGENT_REASONING_EFFORT", "low")





SYSTEM_PROMPT =  """You are Sal, the ordering assistant for LangSlice, a pizzeria in Brooklyn, NY.
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
   changed on the order. You can say a rejection looks mistaken and apologize
   for it, but do not theorize about the cause to the customer: never tell them
   what the kitchen system is or is not counting, or which part of their order
   it did or didn't include. You are the ordering desk, not the kitchen system,
   and you have no visibility into how it computes anything.

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


def _thread_id(runtime: ToolRuntime) -> str:
    """Pull the conversation id out of the injected tool runtime."""
    configurable = (runtime.config or {}).get("configurable") or {}
    return str(configurable.get("thread_id") or "default")


# ---------------------------------------------------------------------------
# Information tools
# ---------------------------------------------------------------------------


@tool
def search_company_info(query: str) -> dict[str, Any]:
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


@tool
def get_menu() -> dict[str, Any]:
    """Get the full LangSlice menu.

    Returns pizza sizes with prices, crust styles and their upcharges, the
    specialty pizzas and what is on them, the topping list with per-topping
    prices, sides, and a ``display_markdown`` version of the full menu. When a
    customer asks to see the menu, reproduce ``display_markdown`` verbatim.
    """
    return db.get_menu()


# ---------------------------------------------------------------------------
# Order tools
# ---------------------------------------------------------------------------


@tool
def start_order(
    customer_name: str,
    order_type: Literal["pickup", "delivery"],
    address: str | None = None,
    *,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """Start a new order for this customer.

    Call this once, before adding any pizzas.

    Args:
        customer_name: Name to put on the order.
        order_type: Either "pickup" or "delivery".
        address: Street address, required for delivery orders.
    """
    thread_id = _thread_id(runtime)

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


@tool
def add_pizza_to_order(
    size: Literal["small", "medium", "large"],
    crust: str,
    toppings: list[str] | None = None,
    specialty_pizza: str | None = None,
    quantity: int = 1,
    notes: str | None = None,
    no_cheese: bool = False,
    *,
    runtime: ToolRuntime,
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
    thread_id = _thread_id(runtime)
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


@tool
def modify_pizza(
    pizza_number: int,
    size: Literal["small", "medium", "large"] | None = None,
    crust: str | None = None,
    add_toppings: list[str] | None = None,
    remove_toppings: list[str] | None = None,
    quantity: int | None = None,
    notes: str | None = None,
    *,
    runtime: ToolRuntime,
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
    thread_id = _thread_id(runtime)
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


@tool
def remove_pizza(pizza_number: int, *, runtime: ToolRuntime) -> dict[str, Any]:
    """Take a pizza off the order.

    Args:
        pizza_number: Which pizza to remove (1 for the first).
    """
    thread_id = _thread_id(runtime)
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


@tool
def add_side(
    item: str,
    quantity: int = 1,
    *,
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """Add a side or drink to the order.

    Args:
        item: Side name as it appears on the menu, e.g. "garlic knots (6)".
        quantity: How many. Defaults to 1.
    """
    thread_id = _thread_id(runtime)
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


@tool
def view_order(*, runtime: ToolRuntime) -> dict[str, Any]:
    """Get the current state of the order, with line items and totals.

    Each pizza carries `unit_price_lines`, which itemizes what it costs: the
    size's base price, the crust upcharge, and every topping with its own
    upcharge. Read a price breakdown straight off those lines rather than
    working one out from the menu. They add up to `unit_price`, the price of a
    single pizza, so on a line with a quantity above one they explain
    `line_total` only after multiplying.
    """
    thread_id = _thread_id(runtime)
    order = db.get_order(thread_id)
    if order is None:
        return {"status": "no_order", "message": "No order has been started yet."}
    return {"status": "ok", "order": db.order_summary(order)}


@tool
def confirm_order(*, runtime: ToolRuntime) -> dict[str, Any]:
    """Send the order to the kitchen once the customer has approved it.

    Returns `status: "confirmed"` with the order number and ETA, or
    `status: "rejected"` when the order cannot be sent - a delivery order under
    the shop's order minimum comes back `below_delivery_minimum` with how much
    more the order needs.
    """
    thread_id = _thread_id(runtime)
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
# Agent
# ---------------------------------------------------------------------------


# Conditional so an empty PIZZA_AGENT_REASONING_EFFORT unsets it, which is what
# lets PIZZA_AGENT_MODEL swap to a non-reasoning model like `openai:gpt-4o`.
_model_kwargs: dict[str, Any] = {}
if REASONING_EFFORT:
    _model_kwargs["reasoning_effort"] = REASONING_EFFORT

#: The agent, referenced by ``langgraph.json``. Tools are passed unbound -
#: ``create_agent`` binds them to the model and wires up the ReAct loop, so
#: every span in the trace is one of the nine pizzeria tools.
pizza_agent = create_agent(
    init_chat_model(MODEL_NAME, **_model_kwargs),
    TOOLS,
    system_prompt=SYSTEM_PROMPT,
    name="langslice_pizza_agent",
)
