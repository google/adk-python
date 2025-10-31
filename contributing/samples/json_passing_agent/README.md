# JSON Passing Agent

This sample demonstrates how to pass structured JSON data between agents. The example uses a pizza ordering scenario where one agent takes the order and passes it to another agent for confirmation.

## How to run

1. Run the agent:
```bash
adk run .
```

2. Talk to the agent:
```
I want to order a pizza
```

## Example conversation
```
[user]: I'd like a large pizza with pepperoni and mushrooms on a thin crust.
[order_intake_agent]: (tool call to get available sizes, crusts, toppings)
[order_intake_agent]: (returns a PizzaOrder JSON)
[order_confirmation_agent]: (tool call to calculate_price)
[order_confirmation_agent]: You ordered a large thin crust pizza with pepperoni and mushrooms. The total price is $15.00.
```
