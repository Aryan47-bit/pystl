import time
import numpy as np
import pystl

sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

print("Starting Unordered Map (O(1) Hash Table) Benchmarks...\n")

for N in sizes:
    print(f">>> TESTING HASH MAP SIZE: {N:,} <<<")
    print("-" * 80)
    
    # Generate random data arrays (int32 matches C++ int natively)
    np.random.seed(42)
    keys = np.random.randint(0, N * 10, size=N, dtype=np.int32)
    vals = np.random.randint(0, 1000, size=N, dtype=np.int32)
    
    # 1. Bulk Insertion Benchmark
    py_dict = {}
    t0 = time.perf_counter()
    if N <= 10_000_000:
        py_dict = dict(zip(keys, vals))
        py_dict_time = time.perf_counter() - t0
    else:
        py_dict_time = -1

    cpp_umap = pystl.unordered_map()
    t0 = time.perf_counter()
    # bulk_update automatically calls .reserve() in C++ to prevent rehashing lag
    cpp_umap.bulk_update(keys, vals)
    cpp_umap_time = time.perf_counter() - t0

    print("Bulk Insertion (O(1) hashing):")
    print(f"  Python dict          : {'Skipped (RAM protection)' if py_dict_time == -1 else f'{py_dict_time:.4f}s'}")
    print(f"  pystl.unordered_map  : {cpp_umap_time:.4f}s")
    
    # 2. Sequential Lookup Benchmark (Testing a 10% slice)
    lookup_count = max(1, N // 10)
    lookup_keys = keys[:lookup_count]
    
    t0 = time.perf_counter()
    if py_dict_time != -1:
        for k in lookup_keys:
            _ = py_dict[k]
        py_dict_lookup = time.perf_counter() - t0
    else:
        py_dict_lookup = -1

    t0 = time.perf_counter()
    for k in lookup_keys:
        _ = cpp_umap[k]
    cpp_umap_lookup = time.perf_counter() - t0
    
    print(f"\nSingle Lookups ({lookup_count:,} items):")
    print(f"  Python dict          : {'Skipped' if py_dict_lookup == -1 else f'{py_dict_lookup:.4f}s'}")
    print(f"  pystl.unordered_map  : {cpp_umap_lookup:.4f}s")

    # 3. Keys Array Extraction (Zero-copy C++ vs Python list)
    t0 = time.perf_counter()
    if N <= 10_000_000:
        py_keys = list(py_dict.keys())
        py_ext_time = time.perf_counter() - t0
    else:
        py_ext_time = -1

    t0 = time.perf_counter()
    cpp_keys = cpp_umap.keys_array()
    cpp_ext_time = time.perf_counter() - t0

    print(f"\nKeys Extraction (Unordered):")
    print(f"  Python dict (list)   : {'Skipped' if py_ext_time == -1 else f'{py_ext_time:.4f}s'}")
    print(f"  pystl (keys_array)   : {cpp_ext_time:.4f}s\n")