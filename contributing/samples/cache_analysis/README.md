# Cache Analysis Research Assistant

This sample demonstrates ADK context caching features with a comprehensive research assistant agent designed to test both Gemini 2.0 Flash and 2.5 Flash context caching capabilities. The sample showcases the difference between explicit ADK caching and Google's built-in implicit caching.

## Key Features

- **App-Level Cache Configuration**: Context cache settings applied at the App level
- **Large Context Instructions**: Over 4200 tokens in system instructions to trigger context caching thresholds
- **Comprehensive Tool Suite**: 7 specialized research and analysis tools
- **Multi-Model Support**: Compatible with any Gemini model, automatically adapts experiment type
- **Performance Metrics**: Detailed token usage tracking including `cached_content_token_count`

## Cache Configuration

```python
ContextCacheConfig(
     min_tokens=4096,
        ttl_seconds=600,  # 10 mins for research sessions
        cache_intervals=3,  # Maximum invocations before cache invalidation
```

## Usage

### Run Cache Experiments

The `run_cache_experiments.py` script compares caching performance between models:

```bash
# Test any Gemini model - script automatically determines experiment type
python run_cache_experiments.py <model_name> --output results.json

# Examples:
python run_cache_experiments.py gemini-2.0-flash-001 --output gemini_2_0_results.json
python run_cache_experiments.py gemini-2.5-flash --output gemini_2_5_results.json
python run_cache_experiments.py gemini-1.5-flash --output gemini_1_5_results.json

# Run multiple iterations for averaged results
python run_cache_experiments.py <model_name> --repeat 3 --output averaged_results.json
```

### Direct Agent Usage

```bash
# Run the agent directly
adk run contributing/samples/cache_analysis/agent.py

# Web interface for debugging
adk web contributing/samples/cache_analysis
```

## Experiment Types

The script automatically determines the experiment type based on the model name:

### Models with "2.5" (e.g., gemini-2.5-flash)
- **Explicit Caching**: ADK explicit caching + Google's implicit caching
- **Implicit Only**: Google's built-in implicit caching alone
- **Measures**: Added benefit of explicit caching over Google's built-in implicit caching

### Other Models (e.g., gemini-2.0-flash-001, gemini-1.5-flash)
- **Cached**: ADK explicit context caching enabled
- **Uncached**: No caching (baseline comparison)
- **Measures**: Raw performance improvement from explicit caching vs no caching

## Tools Included

1. **analyze_data_patterns** - Statistical analysis and pattern recognition in datasets
2. **research_literature** - Academic and professional literature research with citations
3. **generate_test_scenarios** - Comprehensive test case generation and validation strategies
4. **benchmark_performance** - System performance measurement and bottleneck analysis
5. **optimize_system_performance** - Performance optimization recommendations and strategies
6. **analyze_security_vulnerabilities** - Security risk assessment and vulnerability analysis
7. **design_scalability_architecture** - Scalable system architecture design and planning

## Expected Results

### Performance vs Cost Trade-offs

**Note**: This sample uses a tool-heavy agent that may show different performance characteristics than simple text-based agents.

### Performance Improvements
- **Simple Text Agents**: Typically see 30-70% latency reduction with caching
- **Tool-Heavy Agents**: May experience higher latency due to cache setup overhead, but still provide cost benefits
- **Gemini 2.5 Flash**: Compares explicit ADK caching against Google's built-in implicit caching

### Cost Savings
- **Input Token Cost**: 75% reduction for cached content (25% of normal cost)
- **Typical Savings**: 30-60% on input costs for multi-turn conversations
- **Tool-Heavy Workloads**: Cost savings often outweigh latency trade-offs

### Token Metrics
- **Cached Content Token Count**: Non-zero values indicating successful cache hits
- **Cache Hit Ratio**: Proportion of tokens served from cache vs fresh computation

## Troubleshooting

### Zero Cached Tokens
If `cached_content_token_count` is always 0:
- Verify model names match exactly (e.g., `gemini-2.0-flash-001`)
- Check that cache configuration `min_tokens` threshold is met
- Ensure proper App-based configuration is used

### Session Errors
If seeing "Session not found" errors:
- Verify `runner.app_name` is used for session creation
- Check App vs Agent object usage in InMemoryRunner initialization

## Technical Implementation

This sample demonstrates:
- **Modern App Architecture**: App-level cache configuration following ADK best practices
- **Integration Testing**: Comprehensive cache functionality validation
- **Performance Analysis**: Detailed metrics collection and comparison methodology
- **Error Handling**: Robust session management and cache invalidation handling
