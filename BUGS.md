# The two planted bugs

> **Spoilers.** This is the answer key. Read it after you have worked through
> the traces.

The pizzeria agent has two bugs in it on purpose. Both live in `agent.py` tool
implementations, not in the prompt or the graph, so the traces show a healthy
agent doing the wrong thing. Both reproduce identically in `langgraph-agent/` and
`claude-sdk-agent/`.

The span names used below are the LangGraph agent's. In `claude-sdk-agent/` the
tool spans carry their MCP prefix, so `add_pizza_to_order` appears as
`mcp__pizzeria__add_pizza_to_order`, and model calls are named
`claude.assistant.turn` rather than `ChatOpenAI`. The `get_ingredient` child runs
that matter for the first bug are named the same in both.

For each bug, the evidence sits in the trace but never in the model's context.
That is why they fire on every run, and it is also why you can diagnose them
from LangSmith without opening `database.py`.

---

## Bug 1: stock is never checked when adding a pizza

**Where:** `add_pizza_to_order` in `agent.py`, the topping resolution loop.

Toppings are resolved against the ingredient catalog and rejected only when the
catalog has never heard of them. Nothing reads `stock_units`, so a topping the
kitchen is out of today lands on the order.

```python
for requested in toppings:
    ingredient = db.get_ingredient(requested)
    if ingredient is None:            # only checks existence
        unknown.append(requested)
    else:
        resolved.append(ingredient)   # never checks ingredient["stock_units"]
```

A topping that doesn't exist at all is still blocked with
`reason: "unknown_topping"`, so validation looks like it is working. Only the
stock dimension is missing.

Out of stock in `database.py` (`stock_units: 0`): `pineapple`, `spinach`,
`anchovies`, `ricotta`.

**Reproduce:** start a pickup order, ask for "large hand tossed with pepperoni
and pineapple", then "one more with marshmallows".

**In the trace:** `add_pizza_to_order` is called with `toppings: [...,
"pineapple"]` and returns `status: "added"`. Expanding it shows a
`get_ingredient` child run per topping, and the pineapple one returns the full
catalog record including `stock_units: 0`. The tool had the answer and added the
pizza anyway. Marshmallows correctly get rejected since the topping does not exist in the ingredient catalog.

**Why the model never catches it:** `get_ingredient` is `@traceable` in
`database.py`, so every lookup reaches LangSmith as a child run. That is a
write-only side channel. `add_pizza_to_order` passes only topping names into
`build_pizza`, and `get_menu` projects only `name`, `category`, and `price`, so
no stock figure ever reaches the model. It confirms the pineapple because as far
as it can tell, the pineapple is fine.

**Fix in agent.py.** The fix is three commented-out blocks in
`add_pizza_to_order`, each marked `FIX`. Uncomment all three. Note that
`db.list_toppings()` already filters on stock, so the rejection branch can hand
the customer a list of toppings that are actually available.

A list to collect the out-of-stock toppings, next to the `unknown` list:

```python
resolved: list[dict[str, Any]] = []
unknown: list[str] = []
out_of_stock: list[str] = []
```

A branch in the resolution loop, between the existence check and the accept:

```python
for requested in requested_toppings:
    ingredient = db.get_ingredient(requested)
    if ingredient is None:
        unknown.append(requested)
    elif ingredient["stock_units"] <= 0:
        out_of_stock.append(requested)
    else:
        resolved.append(ingredient)
```

And a rejection, alongside the `unknown_topping` one:

```python
if out_of_stock:
    return {
        "status": "rejected",
        "reason": "out_of_stock",
        "message": f"We're out of: {', '.join(out_of_stock)}.",
        "out_of_stock_toppings": out_of_stock,
        "available_toppings": [t["name"] for t in db.list_toppings()],
    }
```

---

## Bug 2: the delivery minimum is measured against sides only

**Where:** `confirm_order` in `agent.py`, the delivery minimum check.

The $25 delivery minimum is enforced against a subtotal that sums only
`order["sides"]`. Pizzas are left out.

```python
if order["order_type"] == "delivery":
    subtotal = sum(side["price"] * side["quantity"] for side in order["sides"])
    if subtotal < db.DELIVERY_MINIMUM:      # pizzas never counted
        ...
```

A pizza-only delivery order scores `$0.00` and is rejected as $25.00 short. The
customer is told to add food to unblock an order that already clears the
minimum, and each side they add moves the figure by only its own price, so the
agent asks again.

There is no code path around the check. The only exits are switching to pickup
or giving up, which is why these threads end with an unplaced order and an
annoyed customer, and why they are the sentiment signal the online evaluator
lesson looks for.

**Reproduce:** start a delivery order, add two large pepperonis and a third
pizza, try to place it, push back on the rejection, then add garlic knots.

**In the trace** (figures from a real run): `add_pizza_to_order` returns an
order summary with `subtotal: 64.75`. The next `confirm_order` returns
`status: "rejected"`, `reason: "below_delivery_minimum"`, `order_subtotal: 0.0`,
`short_by: 25.0`. After the garlic knots, `confirm_order` reports
`order_subtotal: 6.5`, exactly the price of the side, which is the tell. Final
order state is `status: "open"` with `subtotal: 71.25`. Nothing reached the
kitchen.

**Why the model never catches it:** the rejection response carries no order
snapshot, though every other order tool returns `db.order_summary(order)`.
Including it here would put the true subtotal (`64.75`) next to the bogus one
(`0.0`) in a single tool result, and the agent would report the system as broken
rather than relay the instruction. The response gives only the reason, the
minimum, and the two figures.

**Fix in agent.py.** The fix is one commented-out line in `confirm_order`,
marked `FIX`. Uncomment it and delete the line above it. `db.order_summary()`
already totals the pizzas, the sides, and the drinks, and it is the helper every
other order tool uses:

```python
if order["order_type"] == "delivery":
    subtotal = float(db.order_summary(order)["subtotal"])
    if subtotal < db.DELIVERY_MINIMUM:
        ...
```