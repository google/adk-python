# Parallel Function Test Agent

This agent demonstrates parallel function calling functionality in ADK. It includes multiple tools with different processing times to showcase how parallel execution improves performance compared to sequential execution.

## Features

- **Multiple async tool types**: All functions use proper async patterns for true parallelism
- **Thread safety testing**: Tools modify shared state to verify thread-safe operations
- **Performance demonstration**: Clear time differences between parallel and sequential execution
- **GIL-aware design**: Uses `await asyncio.sleep()` instead of `time.sleep()` to avoid blocking

## Tools

1. **get_weather(city)** - Async function, 2-second delay
2. **get_currency_rate(from_currency, to_currency)** - Async function, 1.5-second delay
3. **calculate_distance(city1, city2)** - Async function, 1-second delay
4. **get_population(cities)** - Async function, 0.5 seconds per city

**Important**: All functions use `await asyncio.sleep()` instead of `time.sleep()` to ensure true parallel execution. Using `time.sleep()` would block Python's GIL and force sequential execution despite asyncio parallelism.

## Testing Parallel Function Calling

### Basic Parallel Test
```
Get the weather for New York, London, and Tokyo
```
Expected: 3 parallel get_weather calls (~2 seconds total instead of ~6 seconds sequential)

### Mixed Function Types Test
```
Get the weather in Paris, the USD to EUR exchange rate, and the distance between New York and London
```
Expected: 3 parallel async calls with different functions (~2 seconds total)

### Complex Parallel Test
```
Compare New York and London by getting weather, population, and distance between them
```
Expected: Multiple parallel calls combining different data types

### Performance Comparison Test
You can test the timing difference by asking for the same information in different ways:

**Sequential-style request:**
```
First get the weather in New York, then get the weather in London, then get the weather in Tokyo
```
*Expected time: ~6 seconds (2s + 2s + 2s)*

**Parallel-style request:**
```
Get the weather in New York, London, and Tokyo
```
*Expected time: ~2 seconds (max of parallel 2s delays)*

The parallel version should be **3x faster** due to concurrent execution.

## Thread Safety Testing

All tools modify the agent's state (`tool_context.state`) with request logs including timestamps. This helps verify that:
- Multiple tools can safely modify state concurrently
- No race conditions occur during parallel execution
- State modifications are preserved correctly

## Running the Agent

```bash
# Start the agent in interactive mode
adk run contributing/samples/parallel_functions

# Or use the web interface
adk web
```

## Example Queries

- "Get weather for New York, London, Tokyo, and Paris" *(4 parallel calls, ~2s total)*
- "What's the USD to EUR rate and GBP to USD rate?" *(2 parallel calls, ~1.5s total)*
- "Compare New York and San Francisco: weather, population, and distance" *(3 parallel calls, ~2s total)*
- "Get population data for Tokyo, London, Paris, and Sydney" *(1 call with 4 cities, ~2s total)*
- "What's the weather in Paris and the distance from Paris to London?" *(2 parallel calls, ~2s total)*

## Common Issues and Solutions

### ❌ Problem: Functions still execute sequentially (6+ seconds for 3 weather calls)

**Root Cause**: Using blocking operations like `time.sleep()` in function implementations.

**Solution**: Always use async patterns:
```python
# ❌ Wrong - blocks the GIL, forces sequential execution
def my_tool():
    time.sleep(2)  # Blocks entire event loop

# ✅ Correct - allows true parallelism
async def my_tool():
    await asyncio.sleep(2)  # Non-blocking, parallel-friendly
```

### ✅ Verification: Check execution timing
- Parallel execution: ~2 seconds for 3 weather calls
- Sequential execution: ~6 seconds for 3 weather calls
- If you see 6+ seconds, your functions are blocking the GIL
