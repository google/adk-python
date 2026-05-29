# ADK Task as Sub-agent Sample

## Overview

This sample demonstrates how a "task mode" agent can act as a sub-agent to an LLM agent, effectively extracting structured data from a conversational flow.

The main agent (`coordinator`) delegates interactions to two sub-agents:

1. `order_collector`: A task agent that collects the user's food order (from a menu of Pizza, Burger, Salad) and returns a structured list of selected items as a `list[OrderItem]`.
1. `payment_collector`: A task agent that collects the user's credit card and CVV information, returning a `PaymentInfo` object.

Once the tasks are completed, the coordinator automatically uses a `place_order` tool with the structured data returned by both agents.

## Sample Inputs

- `I would like to order some food please.`

- `I want 2 pizzas and 1 salad.`

- `My credit card is 1234-5678-9012-3456 and my CVV is 123.`

## Graph

```mermaid
graph TD
    coordinator --> order_collector
    coordinator --> payment_collector
    coordinator -.->|uses| place_order[place_order tool]
```

## How To

1. Define a sub-agent with `mode="task"` and an output schema:

   ```python
   order_collector = Agent(
       name="order_collector",
       mode="task",
       output_schema=list[OrderItem],
       ...
   )
   ```

1. Assign it to a parent agent and use it in the instruction to collect the information:

   ```python
   coordinator = Agent(
       sub_agents=[order_collector],
       instruction="Delegate using `order_collector`...",
       ...
   )
   ```
