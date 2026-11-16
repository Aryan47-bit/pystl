import time
import numpy as np
import pystl

sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

print("Starting Map Benchmarks...\n")

for N in sizes:
    print(f">>> TESTING MAP SIZE: {N:,} <<<")
    print("-" * 80)
    
    np.random.seed(42)
    keys = np.random.randint(0, N * 10, size=N, dtype=np.int32)
    vals = np.random.randint(0, 1000, size=N, dtype=np.int32)
    
    # 1. Bulk Insertion
    py_dict = {}
    t0 = time.perf_counter()
    if N <= 10_000_000:
        py_dict = dict(zip(keys, vals))
        py_dict_time = time.perf_counter() - t0
    else:
        py_dict_time = -1

    cpp_map = pystl.map()
    t0 = time.perf_counter()
    cpp_map.bulk_update(keys, vals)
    cpp_map_time = time.perf_counter() - t0

    print("Bulk Insertion (from arrays):")
    print(f"  Python dict (Hash)   : {'Skipped' if py_dict_time == -1 else f'{py_dict_time:.4f}s'}")
    print(f"  pystl.map (Tree)     : {cpp_map_time:.4f}s")
    
    # 2. Sequential Lookup (10% sample)
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
        _ = cpp_map[k]
    cpp_map_lookup = time.perf_counter() - t0
    
    print(f"\nSingle Lookups ({lookup_count:,} items):")
    print(f"  Python dict (O(1))   : {'Skipped' if py_dict_lookup == -1 else f'{py_dict_lookup:.4f}s'}")
    print(f"  pystl.map (O(log N)) : {cpp_map_lookup:.4f}s")

    # 3. Sorted Extraction (O(N log N) sort vs O(N) ordered sweep)
    t0 = time.perf_counter()
    if N <= 10_000_000:
        py_sorted = sorted(py_dict.keys())
        py_sort_time = time.perf_counter() - t0
    else:
        py_sort_time = -1

    t0 = time.perf_counter()
    # No size limit for C++ now because keys_array() is memory-efficient
    cpp_sorted = cpp_map.keys_array() 
    cpp_sort_time = time.perf_counter() - t0

    print("\nSorted Key Extraction:")
    print(f"  Python dict + sorted : {'Skipped' if py_sort_time == -1 else f'{py_sort_time:.4f}s'}")
    print(f"  pystl.map (keys_array) : {cpp_sort_time:.4f}s")

    # 4. Dynamic Interleaved Updates (Insert 10 batches + extract sorted keys each batch)
    if N >= 100 and N <= 1_000_000:
        chunk_size = max(10, N // 10)
        
        d_py = {}
        t0 = time.perf_counter()
        for i in range(0, N, chunk_size):
            for j in range(i, min(i + chunk_size, N)):
                d_py[keys[j]] = vals[j]
            _ = sorted(d_py.keys())
        py_dyn_time = time.perf_counter() - t0

        d_cpp = pystl.map()
        t0 = time.perf_counter()
        for i in range(0, N, chunk_size):
            d_cpp.bulk_update(keys[i:i+chunk_size], vals[i:i+chunk_size])
            _ = list(d_cpp.keys())
        cpp_dyn_time = time.perf_counter() - t0

        print(f"\nDynamic Interleaved Sorting (10 batches of {chunk_size:,}):")
        print(f"  Python dict + sorted : {py_dyn_time:.4f}s")
        print(f"  pystl.map + bulk     : {cpp_dyn_time:.4f}s")
    else:
        print("\nDynamic Interleaved Sorting: Skipped")

    # 5. Correctness Verification
    if N <= 10_000_000:
        assert py_sorted == cpp_sorted, f"Mismatch at N={N}!"
        print("\n[Verification: PASSED - Keys match & strictly sorted]")
    print()