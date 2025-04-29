# ADK (Agent Development Kit) 文件结构

```
.
└── google
    └── adk (Agent Development Kit - Python实现)
        ├── agents
        │   ├── __init__.py                   - 代理模块初始化文件
        │   ├── active_streaming_tool.py      - 管理执行过程中的流式工具相关资源
        │   ├── base_agent.py                 - 基础代理类定义
        │   ├── callback_context.py           - 回调上下文管理
        │   ├── invocation_context.py         - 调用上下文管理，包含会话、代理和配置信息
        │   ├── langgraph_agent.py            - 基于langgraph框架的代理实现
        │   ├── live_request_queue.py         - 实时请求队列管理
        │   ├── llm_agent.py                  - 基于大语言模型的代理实现
        │   ├── loop_agent.py                 - 循环执行的代理实现
        │   ├── parallel_agent.py             - 并行执行的代理实现
        │   ├── readonly_context.py           - 只读上下文管理
        │   ├── remote_agent.py               - 远程代理实现
        │   ├── run_config.py                 - 运行配置管理
        │   ├── sequential_agent.py           - 顺序执行的代理实现
        │   └── transcription_entry.py        - 存储用于转录的数据结构，支持音频转文本功能
        ├── artifacts
        │   ├── __init__.py                   - 工件模块初始化文件
        │   ├── base_artifact_service.py      - 基础工件服务定义
        │   ├── gcs_artifact_service.py       - Google Cloud Storage工件服务实现
        │   └── in_memory_artifact_service.py - 内存工件服务实现
        ├── auth
        │   ├── __init__.py                   - 认证模块初始化文件
        │   ├── auth_credential.py            - 认证凭证管理
        │   ├── auth_handler.py               - 认证处理器
        │   ├── auth_preprocessor.py          - 认证预处理器
        │   ├── auth_schemes.py               - 认证方案定义
        │   └── auth_tool.py                  - 认证工具实现
        ├── cli
        │   ├── browser                       - 浏览器相关的UI资源和代码
        │   │   ├── assets                    - 静态资源文件
        │   │   │   ├── config                - 配置文件
        │   │   │   │   └── runtime-config.json - 运行时配置
        │   │   │   └── audio-processor.js    - 音频处理JavaScript
        │   │   ├── adk_favicon.svg           - ADK图标
        │   │   ├── index.html                - Web界面入口
        │   │   ├── main-CL2IV25D.js          - 主要JavaScript代码
        │   │   ├── polyfills-FFHMD2TL.js     - 兼容性补丁
        │   │   └── styles-4VDSPQ37.css       - 样式表
        │   ├── utils                         - 命令行工具辅助函数
        │   │   ├── __init__.py               - 工具初始化文件
        │   │   ├── envs.py                   - 环境变量管理
        │   │   ├── evals.py                  - 评估工具
        │   │   └── logs.py                   - 日志工具
        │   ├── __init__.py                   - CLI模块初始化文件
        │   ├── __main__.py                   - CLI入口点
        │   ├── agent_graph.py                - 代理图表可视化
        │   ├── cli_create.py                 - 创建命令实现
        │   ├── cli_deploy.py                 - 部署命令实现
        │   ├── cli_eval.py                   - 评估命令实现
        │   ├── cli_tools_click.py            - 基于Click的CLI工具
        │   ├── cli.py                        - CLI主要实现
        │   └── fast_api.py                   - FastAPI服务实现
        ├── code_executors
        │   ├── __init__.py                   - 代码执行器模块初始化文件
        │   ├── base_code_executor.py         - 基础代码执行器
        │   ├── code_execution_utils.py       - 代码执行工具函数
        │   ├── code_executor_context.py      - 代码执行上下文
        │   ├── container_code_executor.py    - 容器化代码执行器
        │   ├── unsafe_local_code_executor.py - 不安全的本地代码执行器
        │   └── vertex_ai_code_executor.py    - Vertex AI代码执行器
        ├── evaluation
        │   ├── __init__.py                   - 评估模块初始化文件
        │   ├── agent_evaluator.py            - 代理评估器
        │   ├── evaluation_constants.py       - 评估常量定义
        │   ├── evaluation_generator.py       - 评估生成器
        │   ├── response_evaluator.py         - 响应评估器
        │   └── trajectory_evaluator.py       - 轨迹评估器
        ├── events
        │   ├── __init__.py                   - 事件模块初始化文件
        │   ├── event_actions.py              - 事件动作
        │   └── event.py                      - 事件基类定义
        ├── examples
        │   ├── __init__.py                   - 示例模块初始化文件
        │   ├── base_example_provider.py      - 基础示例提供器
        │   ├── example_util.py               - 示例工具函数
        │   ├── example.py                    - 示例类定义
        │   └── vertex_ai_example_store.py    - Vertex AI示例存储
        ├── flows
        │   ├── llm_flows                     - 大语言模型流程
        │   │   ├── __init__.py               - LLM流程模块初始化文件
        │   │   ├── _base_llm_processor.py    - 基础LLM处理器
        │   │   ├── _code_execution.py        - 代码执行流程
        │   │   ├── _nl_planning.py           - 自然语言规划
        │   │   ├── agent_transfer.py         - 代理转移流程
        │   │   ├── audio_transcriber.py      - 音频转录器
        │   │   ├── auto_flow.py              - 自动流程生成
        │   │   ├── base_llm_flow.py          - 基础LLM流程
        │   │   ├── basic.py                  - 基础流程实现
        │   │   ├── contents.py               - 内容流程处理
        │   │   ├── functions.py              - 函数调用流程
        │   │   ├── identity.py               - 身份识别流程
        │   │   ├── instructions.py           - 指令处理流程
        │   │   └── single_flow.py            - 单一流程实现
        │   └── __init__.py                   - 流程模块初始化文件
        ├── memory
        │   ├── __init__.py                   - 内存模块初始化文件
        │   ├── base_memory_service.py        - 基础内存服务
        │   ├── in_memory_memory_service.py   - 内存存储服务
        │   └── vertex_ai_rag_memory_service.py - Vertex AI RAG内存服务
        ├── models
        │   ├── __init__.py                   - 模型模块初始化文件
        │   ├── anthropic_llm.py              - Anthropic大语言模型封装
        │   ├── base_llm_connection.py        - 基础LLM连接
        │   ├── base_llm.py                   - 基础LLM定义
        │   ├── gemini_llm_connection.py      - Gemini LLM连接
        │   ├── google_llm.py                 - Google大语言模型封装
        │   ├── lite_llm.py                   - 轻量级LLM封装
        │   ├── llm_request.py                - LLM请求封装
        │   ├── llm_response.py               - LLM响应封装
        │   └── registry.py                   - 模型注册表
        ├── planners
        │   ├── __init__.py                   - 规划器模块初始化文件
        │   ├── base_planner.py               - 基础规划器
        │   ├── built_in_planner.py           - 内置规划器
        │   └── plan_re_act_planner.py        - 基于Plan-ReAct的规划器
        ├── sessions
        │   ├── __init__.py                   - 会话模块初始化文件
        │   ├── base_session_service.py       - 基础会话服务
        │   ├── database_session_service.py   - 数据库会话服务
        │   ├── in_memory_session_service.py  - 内存会话服务
        │   ├── session.py                    - 会话类定义
        │   ├── state.py                      - 状态管理
        │   └── vertex_ai_session_service.py  - Vertex AI会话服务
        ├── tools                             - 工具集合
        │   ├── apihub_tool                   - API Hub工具
        │   │   ├── clients                   - 客户端实现
        │   │   │   ├── __init__.py           - 客户端初始化文件
        │   │   │   ├── apihub_client.py      - API Hub客户端
        │   │   │   └── secret_client.py      - 密钥客户端
        │   │   ├── __init__.py               - API Hub工具初始化文件
        │   │   └── apihub_toolset.py         - API Hub工具集
        │   ├── application_integration_tool  - 应用集成工具
        │   │   ├── clients                   - 客户端实现
        │   │   │   ├── connections_client.py - 连接客户端
        │   │   │   └── integration_client.py - 集成客户端
        │   │   ├── __init__.py               - 应用集成工具初始化文件
        │   │   ├── application_integration_toolset.py - 应用集成工具集
        │   │   └── integration_connector_tool.py - 集成连接器工具
        │   ├── google_api_tool               - Google API工具
        │   │   ├── __init__.py               - Google API工具初始化文件
        │   │   ├── google_api_tool_set.py    - Google API工具集
        │   │   ├── google_api_tool_sets.py   - 多个Google API工具集管理
        │   │   ├── google_api_tool.py        - Google API工具实现
        │   │   └── googleapi_to_openapi_converter.py - Google API转OpenAPI转换器
        │   ├── mcp_tool                      - MCP工具
        │   │   ├── __init__.py               - MCP工具初始化文件
        │   │   ├── conversion_utils.py       - 转换工具函数
        │   │   ├── mcp_session_manager.py    - MCP会话管理器
        │   │   ├── mcp_tool.py               - MCP工具实现
        │   │   └── mcp_toolset.py            - MCP工具集
        │   ├── openapi_tool                  - OpenAPI工具
        │   │   ├── auth                      - 认证实现
        │   │   │   ├── credential_exchangers - 凭证交换器
        │   │   │   │   ├── __init__.py       - 凭证交换器初始化文件
        │   │   │   │   ├── auto_auth_credential_exchanger.py - 自动认证凭证交换器
        │   │   │   │   ├── base_credential_exchanger.py - 基础凭证交换器
        │   │   │   │   ├── oauth2_exchanger.py - OAuth2凭证交换器
        │   │   │   │   └── service_account_exchanger.py - 服务账号凭证交换器
        │   │   │   ├── __init__.py           - 认证初始化文件
        │   │   │   └── auth_helpers.py       - 认证辅助函数
        │   │   ├── common                    - 通用组件
        │   │   │   ├── __init__.py           - 通用组件初始化文件
        │   │   │   └── common.py             - 通用功能实现
        │   │   ├── openapi_spec_parser       - OpenAPI规范解析器
        │   │   │   ├── __init__.py           - 解析器初始化文件
        │   │   │   ├── openapi_spec_parser.py - OpenAPI规范解析器实现
        │   │   │   ├── openapi_toolset.py    - OpenAPI工具集
        │   │   │   ├── operation_parser.py   - 操作解析器
        │   │   │   ├── rest_api_tool.py      - REST API工具
        │   │   │   └── tool_auth_handler.py  - 工具认证处理器
        │   │   └── __init__.py               - OpenAPI工具初始化文件
        │   ├── retrieval                     - 检索工具
        │   │   ├── __init__.py               - 检索工具初始化文件
        │   │   ├── base_retrieval_tool.py    - 基础检索工具
        │   │   ├── files_retrieval.py        - 文件检索
        │   │   ├── llama_index_retrieval.py  - LlamaIndex检索工具
        │   │   └── vertex_ai_rag_retrieval.py - Vertex AI RAG检索工具
        │   ├── __init__.py                   - 工具模块初始化文件
        │   ├── _automatic_function_calling_util.py - 自动函数调用工具
        │   ├── agent_tool.py                 - 代理工具
        │   ├── base_tool.py                  - 基础工具类
        │   ├── built_in_code_execution_tool.py - 内置代码执行工具
        │   ├── crewai_tool.py                - CrewAI工具
        │   ├── example_tool.py               - 示例工具
        │   ├── exit_loop_tool.py             - 退出循环工具
        │   ├── function_parameter_parse_util.py - 函数参数解析工具
        │   ├── function_tool.py              - 函数工具
        │   ├── get_user_choice_tool.py       - 获取用户选择工具
        │   ├── google_search_tool.py         - Google搜索工具
        │   ├── langchain_tool.py             - LangChain工具
        │   ├── load_artifacts_tool.py        - 加载工件工具
        │   ├── load_memory_tool.py           - 加载内存工具
        │   ├── load_web_page.py              - 加载网页工具
        │   ├── long_running_tool.py          - 长时间运行工具
        │   ├── preload_memory_tool.py        - 预加载内存工具
        │   ├── tool_context.py               - 工具上下文
        │   ├── toolbox_tool.py               - 工具箱
        │   ├── transfer_to_agent_tool.py     - 转移到代理工具
        │   └── vertex_ai_search_tool.py      - Vertex AI搜索工具
        ├── __init__.py                       - ADK包初始化文件
        ├── runners.py                        - 各种运行器实现
        ├── telemetry.py                      - 遥测数据收集
        └── version.py                        - 版本信息
```

内置tools

apihub_tool                   - API Hub工具

application_integration_tool  - 应用集成工具

google_api_tool               - Google API工具
       
mcp_tool                      - MCP工具
        
openapi_tool                  - OpenAPI工具
   
retrieval                     - 检索工具

不能用多个内置工具(不能使用内置工具作为sub agent)

## Agent Development Kit 身份验证流程图

```
+----------------+   +---------------+   +--------------------+   +-----+   +----------------+   +-------------+   +-------------------+
|    End User    |   | Client(Spark) |   | Agent Development  |   | LLM |   | Tool(BQ Tool)  |   | Google OAuth|   | Google BigQuery  |
|                |   |               |   |        Kit         |   |     |   |                |   |             |   |        API       |
+-------+--------+   +-------+-------+   +---------+----------+   +--+--+   +--------+-------+   +------+------+   +---------+--------+
        |                    |                     |                 |                |                  |                    |
        |                    |                     |                 |                |                  |                    |
        | callback uri       |                     |                 |                |                  |                    |
        | (with auth code    |                     |                 |                |                  |                    |
        |  appended)         |                     |                 |                |                  |                    |
        +-------------------->                     |                 |                |                  |                    |
        |                    |                     |                 |                |                  |                    |
        |                    | put auth code /     |                 |                |                  |                    |
        |                    | response            |                 |                |                  |                    |
        |                    +--+                  |                 |                |                  |                    |
        |                    |  |                  |                 |                |                  |                    |
        |                    <--+                  |                 |                |                  |                    |
        |                    |                     |                 |                |                  |                    |
        |                    | provide auth code / |                 |                |                  |                    |
        |                    | response            |                 |                |                  |                    |
        |                    +-------------------->                  |                |                  |                    |
        |                    |                     | put auth code / |                |                  |                    |
        |                    |                     | response in the |                |                  |                    |
        |                    |                     | session         |                |                  |                    |
        |                    |                     +--+              |                |                  |                    |
        |                    |                     |  |              |                |                  |                    |
        |                    |                     <--+              |                |                  |                    |
        |                    |                     |                 |                |                  |                    |
        |                    |                     | Tool Call with  |                |                  |                    |
        |                    |                     | auth code /     |                |                  |                    |
        |                    |                     | response in     |                |                  |                    |
        |                    |                     | context         |                |                  |                    |
        |                    |                     +--------------------------------->                  |                    |
        |                    |                     |                 |                |                  |                    |
        |                    |                     |                 |                | Retrieve auto    |                    |
        |                    |                     |                 |                | code/response    |                    |
        |                    |                     |                 |                | from context     |                    |
        |                    |                     |                 |                +--+               |                    |
        |                    |                     |                 |                |  |               |                    |
        |                    |                     |                 |                <--+               |                    |
        |                    |                     |                 |                |                  |                    |
        |                    |                     |                 |                | Exchange access  |                    |
        |                    |                     |                 |                | token and refresh|                    |
        |                    |                     |                 |                | token            |                    |
        |                    |                     |                 |                +----------------->                     |
        |                    |                     |                 |                |                  |                    |
        |                    |                     |                 |                |                  | Tokens exchanged   |
        |                    |                     |                 |                |                  |                    |
        |                    |                     |                 |                <------------------+                    |
        |                    |                     |                 |                |                  |                    |
        |                    |                     |                 |                | Make API call    |                    |
        |                    |                     |                 |                | with Credentials |                    |
        |                    |                     |                 |                +---------------------------------------->
        |                    |                     |                 |                |                  |                    |
        |                    |                     |                 |                |                  |                    | Return API call 
        |                    |                     |                 |                <----------------------------------------+
        |                    |                     |                 |                |                  |                    |
        |                    |                     | Tool Response   |                |                  |                    |
        |                    |                     <----------------------------------|                  |                    |
        |                    |                     |                 |                |                  |                    |
        |                    |                     | Function Calls  |                |                  |                    |
        |                    |                     | Response        |                |                  |                    |
        |                    |                     +----------------->                |                  |                    |
        |                    |                     |                 |                |                  |                    |
        |                    |                     |                 | Final LLM      |                  |                    |
        |                    |                     |                 | Response       |                  |                    |
        |                    |                     <-----------------+                |                  |                    |
        |                    |                     |                 |                |                  |                    |
        |                    | Return events with  |                 |                |                  |                    |
        |                    | LLM Response        |                 |                |                  |                    |
        |                    <---------------------+                 |                |                  |                    |
        |                    |                     |                 |                |                  |                    |
        | final user response|                     |                 |                |                  |                    |
        <--------------------+                     |                 |                |                  |                    |
        |                    |                     |                 |                |                  |                    |
+-------+--------+   +-------+-------+   +---------+----------+   +--+--+   +--------+-------+   +------+------+   +---------+--------+
|    End User    |   | Client(Spark) |   | Agent Development  |   | LLM |   | Tool(BQ Tool)  |   | Google OAuth|   | Google BigQuery  |
|                |   |               |   |        Kit         |   |     |   |                |   |             |   |        API       |
+----------------+   +---------------+   +--------------------+   +-----+   +----------------+   +-------------+   +-------------------+
```

## Agent类型继承关系图

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
    顺序代理, subagent一个个执行
    并行代理, 同时运行subagent执行输出 每个agent独立运行,不依赖其他agent的输出作为输入
    循环代理, 在多个subagent中执行
     
自定义

## 自定义Agent执行流程图

```
                           +---------------------------+
                           |                           |
                           | BaseAgent Class Foundation|
                           |                           |
                           +--------------+------------+
                                          |
                                          |
                                          v
                           +---------------------------+
                           |                           |
                           |  Your Custom Agent Class  +----------------------+
                           |                           |                      |
                           +--------------+------------+                      |
                                          |                                   |
                                          | Implements                        |
                                          v                                   |
                           +---------------------------+                      |
                           |                           |                      |
                           |    Inside _run_async_impl |                      |
                           |                           |                      |
                           +--------------+------------+                      |
                                          |                                   |
                                          v                                   |
                           +---------------------------+                      |
                           |                           |                      |
                           |         Continue          |                      |
                           |                           |                      |
                           +--------------+------------+                      |
                                          |                                   |
                                          v                                   |
                           +---------------------------+                      |
                           |                           |                      |
                           |      Decision Point?      |                      |
                           |                           |                      |
                           +--+---------------------+--+                      |
                              |                     |                         |
                              | Yes                 | No                      |
                              v                     v                         |
          +---------------------------+   +---------------------------+       |
          |                           |   |                           |       |
          |    Call Sub-Agent / Tool  |   |   Perform Custom Action   |       |
          |                           |   |                           |       |
          +-------------+-------------+   +--------------+------------+       |
                        |                                |                    |
                        |                                |                    |
                        v                                v                    |
          +---------------------------+                  |                    |
          |                           |                  |                    |
          |      Process Result       |<-----------------+                    |
          |                           |                                       |
          +-------------+-------------+                                       |
                        |                                                     |
                        v                                                     |
          +---------------------------+                                       |
          |                           |                                       |
          |        Yield Event        |<--------------------------------------+
          |                           |
          +---------------------------+
```

控制
before_agent_callback

before_model_callback
after_model_callback

before_model_callback
after_model_callback
call_tool
before_model_callback
after_model_callback

...
after_agent_callback