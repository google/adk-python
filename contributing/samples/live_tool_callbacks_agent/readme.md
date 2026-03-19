# Live Tool Callbacks Agent

This sample demonstrates how tool callbacks work in live (bidirectional streaming) mode. It showcases both `before_tool_callback` and `after_tool_callback` functionality with multiple callback chains, async callbacks, and various callback behaviors.

## Features Demonstrated

### Before Tool Callbacks
1. **Audit Callback**: Logs all tool calls before execution
2. **Security Callback**: Can block tool calls based on security rules (e.g., restricted locations)
3. **Async Validation Callback**: Performs async validation and can prevent invalid operations

### After Tool Callbacks
1. **Enhancement Callback**: Adds metadata to tool responses
2. **Async Post-processing Callback**: Performs async post-processing of responses

### Tools Available
- `get_weather`: Get weather information for any location
- `calculate_async`: Perform mathematical calculations asynchronously
- `log_activity`: Log activities with timestamps

## Testing Scenarios

### 1. Basic Callback Flow
```
"What's the weather in New York?"
```
Watch the console output to see:
- Audit logging before the tool call
- Security check (will pass for New York)
- Response enhancement after the tool call

### 2. Security Blocking
```
"What's the weather in classified?"
```
The security callback will block this request and return an error response.

### 3. Validation Prevention
```
"Calculate 10 divided by 0"
```
The async validation callback will prevent division by zero.

### 4. Multiple Tool Calls
```
"Get weather for London and calculate 5 + 3"
```
See how callbacks work with multiple parallel tool calls.

### 5. Callback Chain Testing
```
"Log this activity: Testing callback chains"
```
Observe how multiple callbacks in the chain are processed.

## Getting Started

1. **Start the ADK Web Server**
   ```bash
   adk web
   ```

2. **Access the ADK Web UI**
   Navigate to `http://localhost:8000`

3. **Select the Agent**
   Choose "tool_callbacks_agent" from the dropdown in the top-left corner

4. **Start Streaming**
   Click the **Audio** or **Video** icon to begin streaming

5. **Test Callbacks**
   Try the testing scenarios above and watch both the chat responses and the console output to see callbacks in action

## What to Observe

- **Console Output**: Watch for callback logs with emojis:
  - 🔍 AUDIT: Audit callback logging
  - 🚫 SECURITY: Security callback blocking
  - ⚡ ASYNC BEFORE: Async preprocessing
  - ✨ ENHANCE: Response enhancement
  - 🔄 ASYNC AFTER: Async post-processing

- **Enhanced Responses**: Tool responses will include additional metadata added by after callbacks

- **Error Handling**: Security blocks and validation errors will be returned as proper error responses

## Technical Notes

- This sample demonstrates that tool callbacks now work identically in both regular and live streaming modes
- Multiple callbacks are supported and processed in order
- Both sync and async callbacks are supported
- Callbacks can modify, enhance, or block tool execution
- The callback system provides full control over the tool execution pipeline 