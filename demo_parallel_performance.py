"""
Performance demonstration for parallel LLM-as-judge evaluation.

This script demonstrates the performance improvement from parallelizing
LLM evaluation calls using asyncio.gather().
"""

import asyncio
import time
from typing import Optional

from google.genai import types as genai_types


# Simulated LLM call with artificial delay
async def mock_llm_call(delay: float = 0.5):
    """Simulates an LLM API call with specified delay."""
    await asyncio.sleep(delay)
    return genai_types.Content(
        parts=[genai_types.Part(text="Mock LLM response")],
        role="model",
    )


async def serial_evaluation(num_invocations: int, num_samples: int, delay: float):
    """Simulates the OLD serial evaluation approach."""
    results = []
    for i in range(num_invocations):
        invocation_samples = []
        for j in range(num_samples):
            response = await mock_llm_call(delay)
            invocation_samples.append(response)
        results.append(invocation_samples)
    return results


async def parallel_evaluation(num_invocations: int, num_samples: int, delay: float):
    """Simulates the NEW parallel evaluation approach."""
    tasks = []
    invocation_indices = []
    
    # Create all N×M tasks
    for i in range(num_invocations):
        for j in range(num_samples):
            tasks.append(mock_llm_call(delay))
            invocation_indices.append(i)
    
    # Execute in parallel
    all_results = await asyncio.gather(*tasks)
    
    # Group by invocation
    results_by_invocation = {}
    for idx, result in zip(invocation_indices, all_results):
        if idx not in results_by_invocation:
            results_by_invocation[idx] = []
        results_by_invocation[idx].append(result)
    
    return [results_by_invocation[i] for i in sorted(results_by_invocation.keys())]


async def main():
    """Run performance comparison."""
    num_invocations = 5
    num_samples = 2
    delay = 0.5  # 500ms per call
    
    print("=" * 60)
    print("LLM-as-Judge Parallel Evaluation Performance Test")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Invocations: {num_invocations}")
    print(f"  - Samples per invocation: {num_samples}")
    print(f"  - Total LLM calls: {num_invocations * num_samples}")
    print(f"  - Simulated delay per call: {delay}s")
    print()
    
    # Test serial approach
    print("Testing SERIAL approach (old)...")
    start_time = time.time()
    serial_results = await serial_evaluation(num_invocations, num_samples, delay)
    serial_time = time.time() - start_time
    print(f"✓ Completed in {serial_time:.2f}s")
    print()
    
    # Test parallel approach
    print("Testing PARALLEL approach (new)...")
    start_time = time.time()
    parallel_results = await parallel_evaluation(num_invocations, num_samples, delay)
    parallel_time = time.time() - start_time
    print(f"✓ Completed in {parallel_time:.2f}s")
    print()
    
    # Calculate speedup
    speedup = serial_time / parallel_time
    time_saved = serial_time - parallel_time
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Serial time:    {serial_time:.2f}s")
    print(f"Parallel time:  {parallel_time:.2f}s")
    print(f"Speedup:        {speedup:.2f}x faster")
    print(f"Time saved:     {time_saved:.2f}s ({time_saved/serial_time*100:.1f}%)")
    print("=" * 60)
    
    # Verify results are the same
    assert len(serial_results) == len(parallel_results)
    for i in range(len(serial_results)):
        assert len(serial_results[i]) == len(parallel_results[i])
    print("✓ Results verified: both approaches produce same output structure")


if __name__ == "__main__":
    asyncio.run(main())
