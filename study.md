# AI System Components

```
+--------------+  +--------------+  +----------------+  +--------------+
|    Agents    |  |     Tool     |  | Orchestration  |  |  Callbacks   |
+--------------+  +--------------+  +----------------+  +--------------+

+--------------+  +--------------+  +----------------+  +--------------+
| Bidirectional|  |    Session   |  |   Evaluation   |  |  Deployment  |
|   Streaming  |  |  Management  |  |                |  |              |
+--------------+  +--------------+  +----------------+  +--------------+

+--------------+  +--------------+  +----------------+  +--------------+
|   Artifact   |  |    Memory    |  |      Code      |  |   Planning   |
|  Management  |  |              |  |   Execution    |  |              |
+--------------+  +--------------+  +----------------+  +--------------+

+-----------------------------------------------------------------------+
|                             Debugging                                  |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
|                               Trace                                    |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
|                              Models                                    |
+-----------------------------------------------------------------------+
```

Agent: Workflow
    - SequentialAgent
    - ParallelAgent
    - LoopAgent
```
                                     +--------------+
                                     |              |
                                     |  BaseAgent   |
                                     |              |
                                     +------+-------+
                                            |
                   +------------------------+------------------------+
                   |                        |                        |
                   |                        |                        |
            +------v-------+        +-------v--------+       +-------v-------+
            |              |        |                |       |               |
            | A. LLM-Based |        | B. Workflow    |       | C. Custom     |
            |              |        |    Agents      |       |    Logic      |
            +------+-------+        +-------+--------+       +-------+-------+
                   |                        |                        |
                   |                        |                        |
            +------v-------+    +-----------+-----------+     +------v-------+
            |              |    |           |           |     |              |
            |  LlmAgent    |    |           |           |     | CustomAgent  |
            |(Reasoning,   |    |           |           |     |              |
            | Tools,       |    |           |           |     |              |
            | Transfer)    |    |           |           |     |              |
            |              |    |           |           |     |              |
            +--------------+    |           |           |     +--------------+
                                |           |           |
                      +---------v--+ +------v-----+ +---v----------+
                      |            | |            | |              |
                      | Sequential | | Parallel   | | LoopAgent   |
                      |   Agent    | |   Agent    | |              |
                      |            | |            | |              |
                      +------------+ +------------+ +--------------+
```
LLM不确定性

workflow 确定性

    顺序代理, subagent一个个执行, 像流水线一样
    并行代理, 同时运行subagent执行输出 每个agent独立运行,不依赖其他agent的输出作为输入
    循环代理
        - 按顺序循环执行所有子代理
        - 使用循环次数终止
        - 子代理评估

Tool:
- Function Tool
- build-in Tool
    - Google_search
    - Code Execution
    - Vertex AI Search
    Built-in tools cannot be used within a sub-agent.(Can use as tool)
- Third party tools
- Google Cloud tools
- MCP tools
- Authentication
- OpenAPI tools

Callback:
    ```
    before_agent_callback

    before_model_callback
    after_model_callback

    before_model_callback
    after_model_callback

    call_tool
    before_tool_callback
    after_tool_callback

    ...
    after_agent_callback
    ```
    - 日志
    - 状态管理
    - 控制flow

Evaluation: 图形化评估工具
- 精确匹配
- 模糊匹配
- 精准度
- 召回率

Artifact Management: 资源保存
Memory: 会话存储
Planning: BasePlanner 拆解任务, 实现类似ReAct的效果


agent和tool的区别:
    1. agent可以当tool使用
    2. 控制权
    3. agent可以依赖其他llm


运行时:
```
+-------------------------------------------------------------------------+
|                     Agent Development Kit Runtime                        |
+-------------------------------------------------------------------------+
|                                                                         |
|  1 User: Help me to do...?                                              |
|     session_id: "s_123"                                                 |
|     +--------------------------------------------------------+          |
|     |                                                        |          |
|     v                                                        |          |
|  +----------------+                  +--------------------+  |      +---+----+
|  |                |<---------------->|     Services       |<-+----->|        |
|  |    Runner      |                  | session (state),   |  |      | Storage|
|  |                |                  | artifact, memory,  |  |      |        |
|  | +------------+ |                  | etc                |  |      +--------+
|  | |   Event    | |                  +--------------------+  |          
|  | | Processor  | |                                          |          
|  | +------------+ |                                          |          
|  |       ^        |                                          |          
|  |       |        |                                          |          
|  |       | Yield  |                                          |          
|  |       |        |                                          |          
|  |  Ask  |  2     |                                          |          
|  |       | Event  |                                          |          
|  |       | Loop   |                                          |          
|  |       v        |                                          |          
|  | +------------+ |                                          |          
|  | | Execution  | |                                          |          
|  | |   Logic    | |                                          |          
|  | |            | |                                          |          
|  | | Agent, LLM | |                                          |          
|  | | innovation,| |                                          |          
|  | | Callbacks, | |                                          |          
|  | | Tools, etc | |                                          |          
|  | +------------+ |                                          |          
|  +----------------+                                          |          
|     |                                                        |          
|     v                                                        |          
|  3 Stream<Event>                                             |          
|                                                              |          
+--------------------------------------------------------------+          
```

