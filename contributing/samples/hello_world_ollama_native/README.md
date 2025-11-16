# Using Ollama Models with ADK (Native Integration)

## Model Choice

If your agent uses tools, choose an Ollama model that supports **function calling**.  
Tool support can be verified with:

```bash
ollama show mistral-small3.1
```
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

Models must list tools under Capabilities.
Models without tool support will not execute ADK functions correctly.

To inspect or customize a model template:
```bash
ollama show --modelfile llama3.1 > model_file_to_modify
```
Then create a modified model:

ollama create llama3.1-modified -f model_file_to_modify


## Native Ollama Provider in ADK

ADK includes a native Ollama model class that communicates directly with the Ollama server at:

http://localhost:11434/api/chat

No LiteLLM provider, API keys, or OpenAI proxy endpoints are needed.

### Example agent
```python
import random
from google.adk.agents.llm_agent import Agent
from google.adk.models.ollama import Ollama

def roll_die(sides: int) -> int:
    return random.randint(1, sides)

def check_prime(numbers: list[int]) -> str:
    primes = []
    for number in numbers:
        number = int(number)
        if number <= 1:
            continue
        for i in range(2, int(number ** 0.5) + 1):
            if number % i == 0:
                break
        else:
            primes.append(number)
    return "No prime numbers found." if not primes else f"{', '.join(map(str, primes))} are prime numbers."

root_agent = Agent(
    model=Ollama(model="llama3.1"),
    name="dice_agent",
    description="Agent that rolls dice and checks primes using native Ollama.",
    instruction="Always use the provided tools.",
    tools=[roll_die, check_prime],
)
```
## Connecting to a remote Ollama server

Default Ollama endpoint:

http://localhost:11434

Override using an environment variable:
```bash
export OLLAMA_API_BASE="http://192.168.1.20:11434"
```
Or pass explicitly in code:
```python
Ollama(model="llama3.1", host="http://192.168.1.20:11434")
```


## Running the Example with ADK Web

Start the ADK Web UI:

adk web hello_ollama_native

The interface will be available in your browser, allowing interactive testing of tool calls.




