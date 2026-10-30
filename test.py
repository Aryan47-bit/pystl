import time
import numpy as np
import pystl

# Attempt to load Cython module if it exists
try:
    import cy_bench
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    print("Notice: 'cy_bench' module not found. Cython benchmarks will be skipped.")

def run_benchmarks():
    # Updated to test all powers of 10 from 10 to 100,000,000
    sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    
    print("=" * 85)
    print(" PySTL Native Vectorized vs NumPy vs Python vs Cython Benchmark")
    print("=" * 85)

    for N in sizes:
        print(f"\n>>> TESTING ARRAY SIZE: {N:,} <<<")
        print("-" * 85)
        
        arr = np.zeros(N, dtype=np.int32)
        vec = pystl.vector(N)
        lst = [0] * N if N <= 10_000_000 else None
        
        # 1. BULK FILL / LIST CREATION
        start = time.perf_counter()
        arr.fill(5)
        t_np = time.perf_counter() - start
        
        start = time.perf_counter()
        vec.fill(5)
        t_pystl = time.perf_counter() - start
        
        t_py = "Skipped"
        if lst is not None:
            start = time.perf_counter()
            lst = [5] * N
            t_py = f"{time.perf_counter() - start:.4f}s"
            
        t_cy = "Skipped"
        if HAS_CYTHON:
            start = time.perf_counter()
            cy_bench.fill(arr, 5) # Assuming Cython operates on typed memoryviews
            t_cy = f"{time.perf_counter() - start:.4f}s"
            
        print("Bulk Fill:")
        print(f"  NumPy (Vectorized)   : {t_np:.4f}s")
        print(f"  Python list (* N)    : {t_py}")
        print(f"  Cython (MemoryView)  : {t_cy}")
        print(f"  pystl.vector (Native): {t_pystl:.4f}s\n")

        # 2. ZERO-COPY BUFFER ASSIGNMENT
        np_wrap = np.asarray(vec)
        
        start = time.perf_counter()
        arr[:] = 1
        t_np = time.perf_counter() - start
        
        start = time.perf_counter()
        np_wrap[:] = 1
        t_pystl = time.perf_counter() - start
        
        t_py = "Skipped"
        if lst is not None:
            start = time.perf_counter()
            for i in range(N): lst[i] = 1
            t_py = f"{time.perf_counter() - start:.4f}s"
            
        print("Buffer Protocol Slice Assignment:")
        print(f"  NumPy                : {t_np:.4f}s")
        print(f"  Python list (Loop)   : {t_py}")
        print(f"  pystl (via NumPy)    : {t_pystl:.4f}s\n")

        # 3. SCALAR ADDITION
        start = time.perf_counter()
        _ = arr + 5
        t_np = time.perf_counter() - start
        
        start = time.perf_counter()
        vec.add_scalar(5)
        t_pystl = time.perf_counter() - start
        
        t_py = "Skipped"
        if lst is not None:
            start = time.perf_counter()
            _ = [x + 5 for x in lst]
            t_py = f"{time.perf_counter() - start:.4f}s"
            
        print("Scalar Addition (+5):")
        print(f"  NumPy (Vectorized)   : {t_np:.4f}s")
        print(f"  Python list (Compre.): {t_py}")
        print(f"  pystl.vector (Native): {t_pystl:.4f}s\n")

        # 4. REDUCTIONS & SORTING SETUP
        arr_rand = np.random.randint(0, 1000, size=N, dtype=np.int32)
        vec_rand = pystl.vector(N)
        np.copyto(np.asarray(vec_rand), arr_rand) 
        lst_rand = arr_rand.tolist() if N <= 10_000_000 else None

        # SUM
        start = time.perf_counter()
        _ = np.sum(arr_rand)
        t_np = time.perf_counter() - start
            
        start = time.perf_counter()
        _ = vec_rand.sum()
        t_pystl = time.perf_counter() - start
        
        t_py = "Skipped"
        if lst_rand is not None:
            start = time.perf_counter()
            _ = sum(lst_rand)
            t_py = f"{time.perf_counter() - start:.4f}s"
            
        print("Summation (Sum):")
        print(f"  NumPy (Vectorized)   : {t_np:.4f}s")
        print(f"  Python list (Native) : {t_py}")
        print(f"  pystl.vector (Native): {t_pystl:.4f}s\n")

        # MAX
        start = time.perf_counter()
        _ = np.max(arr_rand)
        t_np = time.perf_counter() - start
            
        start = time.perf_counter()
        _ = vec_rand.max()
        t_pystl = time.perf_counter() - start
        
        t_py = "Skipped"
        if lst_rand is not None:
            start = time.perf_counter()
            _ = max(lst_rand)
            t_py = f"{time.perf_counter() - start:.4f}s"
            
        print("Maximum (Max):")
        print(f"  NumPy (Vectorized)   : {t_np:.4f}s")
        print(f"  Python list (Native) : {t_py}")
        print(f"  pystl.vector (Native): {t_pystl:.4f}s\n")

        # MIN
        start = time.perf_counter()
        _ = np.min(arr_rand)
        t_np = time.perf_counter() - start
            
        start = time.perf_counter()
        _ = vec_rand.min()
        t_pystl = time.perf_counter() - start
        
        t_py = "Skipped"
        if lst_rand is not None:
            start = time.perf_counter()
            _ = min(lst_rand)
            t_py = f"{time.perf_counter() - start:.4f}s"
            
        print("Minimum (Min):")
        print(f"  NumPy (Vectorized)   : {t_np:.4f}s")
        print(f"  Python list (Native) : {t_py}")
        print(f"  pystl.vector (Native): {t_pystl:.4f}s\n")

        # SORTING
        start = time.perf_counter()
        np.sort(arr_rand) 
        t_np = time.perf_counter() - start
        
        start = time.perf_counter()
        vec_rand.sort()
        t_pystl = time.perf_counter() - start
        
        t_py = "Skipped"
        if lst_rand is not None:
            start = time.perf_counter()
            lst_rand.sort()
            t_py = f"{time.perf_counter() - start:.4f}s"
        
        print("Sorting:")
        print(f"  NumPy (Vectorized)   : {t_np:.4f}s")
        print(f"  Python list (Native) : {t_py}")
        print(f"  pystl.vector (Native): {t_pystl:.4f}s")

if __name__ == "__main__":
    run_benchmarks()