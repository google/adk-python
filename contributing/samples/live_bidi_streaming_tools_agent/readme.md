 This is only supported in streaming(live) agents/api.

Streaming tools allows tools(functions) to stream intermediate results back to agents and agents can respond to those intermediate results. 
For example, we can use streaming tools to monitor the changes of the stock price and have the agent react to it. Another example is we can have the agent monitor the video stream, and when there is changes in video stream, the agent can report the changes.

To define a streaming tool, you must adhere to the following:

1.  **Asynchronous Function:** The tool must be an `async` Python function.
2.  **AsyncGenerator Return Type:** The function must be typed to return an `AsyncGenerator`. The first type parameter to `AsyncGenerator` is the type of the data you `yield` (e.g., `str` for text messages, or a custom object for structured data). The second type parameter is typically `None` if the generator doesn't receive values via `send()`.


We support two types of streaming tools:
- Simple type. This is a one type of streaming tools that only take non video/audio streams(the streams that you feed to adk web or adk runner) as input.
- Video streaming tools. This only works in video streaming and the video stream(the streams that you feed to adk web or adk runner) will be passed into this function.


Here are some sample queries to test:
- Help me monitor the stock price for $XYZ stock.
- Help me monitor how many people are there in the video stream.