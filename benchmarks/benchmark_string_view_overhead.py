"""
Microbenchmark for std::string_view overhead reduction.

This benchmark measures the performance improvement from using std::string_view
for read-only string parameters instead of const std::string& in the C++ cleaning module.
"""

import time
import arnio


def benchmark_drop_duplicates():
    """Benchmark drop_duplicates function with string_view parameter."""
    df = arnio.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    
    start = time.perf_counter()
    for _ in range(500_000):
        _ = arnio.drop_duplicates(df)
    end = time.perf_counter()
    
    return end - start


def benchmark_normalize_case():
    """Benchmark normalize_case function with string_view parameter."""
    df = arnio.DataFrame({"A": [1, 2], "B": ["Hello", "World"]})
    
    start = time.perf_counter()
    for _ in range(500_000):
        _ = arnio.normalize_case(df)
    end = time.perf_counter()
    
    return end - start


if __name__ == "__main__":
    print("=" * 60)
    print("std::string_view Overhead Microbenchmark")
    print("=" * 60)
    
    print("\nBenchmarking drop_duplicates (500,000 iterations)...")
    drop_dup_time = benchmark_drop_duplicates()
    print(f"  Total time: {drop_dup_time:.4f} seconds")
    print(f"  Per call:   {drop_dup_time / 500_000 * 1_000_000:.4f} microseconds")
    
    print("\nBenchmarking normalize_case (500,000 iterations)...")
    normalize_time = benchmark_normalize_case()
    print(f"  Total time: {normalize_time:.4f} seconds")
    print(f"  Per call:   {normalize_time / 500_000 * 1_000_000:.4f} microseconds")
    
    print("\n" + "=" * 60)
    print(f"Combined total time: {drop_dup_time + normalize_time:.4f} seconds")
    print("=" * 60)
