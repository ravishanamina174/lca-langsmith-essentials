"""Code evaluator for out-of-stock ingredients in pizza orders."""

import json

import database as db


def ingredient_stock_evaluator(run, example=None):
    outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {})
    outputs = outputs or {}

    for message in reversed(outputs.get("messages", [])):
        if isinstance(message, dict):
            role = message.get("role") or message.get("type")
            content = message.get("content")
        else:
            role = message.type
            content = message.content

        if role != "tool":
            continue

        result = json.loads(content)
        order = result.get("order")
        if not order:
            continue

        out_of_stock = {
            topping
            for pizza in order["pizzas"]
            for topping in pizza["toppings"]
            if db.get_ingredient(topping)["stock_units"] <= 0
        }

        return {
            "score": 0 if out_of_stock else 1,
            "comment": (
                f"Out-of-stock ingredients added: {', '.join(sorted(out_of_stock))}"
                if out_of_stock
                else "All ordered ingredients are in stock."
            ),
        }

    return {"score": 1, "comment": "No order found."}
