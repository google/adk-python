# Using ollama models with ADK

## Model choice

If your agent is relying on tools, please make sure that you select a model with tool support from [ollama website](https://ollama.com/search?c=tools).

For reliable results, we recommend using a decent size model with tool support.

The tool support for the model can be checked with the following command:

```bash
ollama show mistral-small3.1
  Model
    architecture        mistral3
    parameters          24.0B
    context length      131072
    embedding length    5120
    quantization        Q4_K_M

  Capabilities
    completion
    vision
    tools
```

You are supposed to see `tools` listed under capabilities.

You can also look at the template the model is using and tweak it based on your needs.

```bash
ollama show --modelfile llama3.1 > model_file_to_modify
```

Then you can create a model with the following command:

```bash
ollama create llama3.1-modified -f model_file_to_modify
```

## Using ollama_chat provider

Our LiteLlm wrapper can be used to create agents with ollama models.

```py
root_agent = Agent(
    model=LiteLlm(model="ollama_chat/mistral-small3.1"),
    name="dice_agent",
    description=(
        "hello world agent that can roll a dice of 8 sides and check prime"
        " numbers."
    ),
    instruction="""
      You roll dice and answer questions about the outcome of the dice rolls.
    """,
    tools=[
        roll_die,
        check_prime,
    ],
)
```

**It is important to set the provider `ollama_chat` instead of `ollama`. Using `ollama` will result in unexpected behaviors such as infinite tool call loops and ignoring previous context.**

While `api_base` can be provided inside litellm for generation, litellm library is calling other APIs relying on the env variable instead as of v1.65.5 after completion. So at this time, we recommend setting the env variable `OLLAMA_API_BASE` to point to the ollama server.

```bash
export OLLAMA_API_BASE="http://localhost:11434"
adk web
```

## Using openai provider

Alternatively, `openai` can be used as the provider name. But this will also require setting the  `OPENAI_API_BASE=http://localhost:11434/v1` and `OPENAI_API_KEY=anything` env variables instead of `OLLAMA_API_BASE`. **Please notice that api base now has `/v1` at the end.**

```py
root_agent = Agent(
    model=LiteLlm(model="openai/mistral-small3.1"),
    name="dice_agent",
    description=(
        "hello world agent that can roll a dice of 8 sides and check prime"
        " numbers."
    ),
    instruction="""
      You roll dice and answer questions about the outcome of the dice rolls.
    """,
    tools=[
        roll_die,
        check_prime,
    ],
)
```

```bash
export OPENAI_API_BASE=http://localhost:11434/v1
export OPENAI_API_KEY=anything
adk web
```

## Debugging

You can see the request sent to the ollama server by adding the following in your agent code just after imports.

```py
import litellm
litellm._turn_on_debug()
```

Look for a line like the following:

```bash
quest Sent from LiteLLM:
curl -X POST \
http://localhost:11434/api/chat \
-d '{'model': 'mistral-small3.1', 'messages': [{'role': 'system', 'content': ...
```
