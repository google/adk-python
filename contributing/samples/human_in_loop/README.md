# Agent with Long-Running Tools

This example demonstrates an agent using a long-running tool (`ask_for_approval`).

## Key Flow for Long-Running Tools

1.  **Initial Call**: The agent calls the long-running tool (e.g., `ask_for_approval`).
2.  **Initial Tool Response**: The tool immediately returns an initial response, typically indicating a "pending" status and a way to track the request (e.g., a `ticket-id`). This is sent back to the agent as a `types.FunctionResponse` (usually processed internally by the runner and then influencing the agent's next turn).
3.  **Agent Acknowledges**: The agent processes this initial response and usually informs the user about the pending status.
4.  **External Process/Update**: The long-running task progresses externally (e.g., a human approves the request).
5.  **❗️Crucial Step: Provide Updated Tool Response❗️**:
    * Once the external process completes or updates, your application **must** construct a new `types.FunctionResponse`.
    * This response should use the **same `id` and `name`** as the original `FunctionCall` to the long-running tool.
    * The `response` field within this `types.FunctionResponse` should contain the *updated data* (e.g., `{'status': 'approved', ...}`).
    * Send this `types.FunctionResponse` back to the agent as a part within a new message using `role="user"`.

    ```python
    # Example: After external approval
    updated_tool_output_data = {
        "status": "approved",
        "ticket-id": ticket_id, # from original call
        # ... other relevant updated data
    }

    updated_function_response_part = types.Part(
        function_response=types.FunctionResponse(
            id=long_running_function_call.id,   # Original call ID
            name=long_running_function_call.name, # Original call name
            response=updated_tool_output_data,
        )
    )

    # Send this back to the agent
    await runner.run_async(
        # ... session_id, user_id ...
        new_message=types.Content(
            parts=[updated_function_response_part], role="user"
        ),
    )
    ```
6.  **Agent Acts on Update**: The agent receives this message containing the `types.FunctionResponse` and, based on its instructions, proceeds with the next steps (e.g., calling another tool like `reimburse`).

**Why is this important?** The agent relies on receiving this subsequent `types.FunctionResponse` (provided in a message with `role="user"` containing the specific `Part`) to understand that the long-running task has concluded or its state has changed. Without it, the agent will remain unaware of the outcome of the pending task.
