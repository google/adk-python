# ADK Workflow Message Sample

## Overview

This sample demonstrates different ways to send a message to a user using `Event(message=...)` within an **ADK Workflows** node. It covers:

1. **String Messages**: Standard string text replies.
1. **Multi-modal Messages**: Returning mixed modality inputs, such as a string combined with an inline image.
1. **Multiple Messages**: Emitting multiple full messages from the same node with a delay between them.
1. **Streaming Messages**: Simulating an LLM streaming response by breaking a message into chunks and yielding them with the `partial=True` flag at intervals.

## Sample Inputs

This workflow executes sequentially and successfully without any expected user input. Since it has only one `Workflow` node chain that automatically progresses from `START`, you can just type anything (e.g. `start`) to kick it off.

## Graph

```mermaid
graph TD
    START --> send_string
    send_string --> send_multimodal
    send_multimodal --> multiple_messages
    multiple_messages --> stream_sentence
    stream_sentence --> END[Workflow Ends]
```

## How To

To send messages in an ADK node, yield an `Event` object with the `message` argument:

1. **Send a simple string**:

   ```python
   yield Event(message="Hello world!")
   ```

1. **Send text with an image** (multi-modal):

   ```python
   from google.genai import types
   yield Event(
       message=[
           types.Part.from_text(text="Look at this image:"),
           types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
       ]
   )
   ```

1. **Send multiple messages**:
   To send multiple distinct messages from a single node, yield multiple `Event` objects sequentially.

   > **Note**: When yielding multiple messages with delays (`await asyncio.sleep(...)`), your node function **must be an asynchronous generator** (`async def`). This allows ADK to yield each message to the client immediately without blocking.

   ```python
   import asyncio

   async def multiple_messages(node_input: Any = None):
     yield Event(message="Processing step 1...")
     await asyncio.sleep(1.0)

     yield Event(message="Processing step 2...")
     await asyncio.sleep(1.0)

     yield Event(message="Done processing.")
   ```

1. **Stream a message in chunks**:
   Provide the `partial=True` flag for intermediate chunks. This provides a better user experience by allowing the UI to show the response in a streaming fashion, thereby lowering the latency to see the first word. ADK automatically accumulates all partial messages and merges them into a final message for you for session storage.

   > **Note**: To stream multiple messages or tokens smoothly, your node function **must be an asynchronous generator** (`async def`). This allows ADK to yield messages to the client immediately without blocking.

   ```python
   import asyncio

   async def stream_sentence(node_input: str):
       yield Event(message="How ", partial=True)
       await asyncio.sleep(0.5)
       yield Event(message="may I", partial=True)
       await asyncio.sleep(0.5)
       yield Event(message=" help you?", partial=True)
   ```
