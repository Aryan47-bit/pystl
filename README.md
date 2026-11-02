# pystl

Welcome to pystl. This is a project aimed at bringing the C++ Standard Template Library (STL) directly into Python.

Python lists are great, but they hide a lot of the underlying memory management from the user. I built this wrapper using pybind11 to give Python developers access to the strict memory rules, capacity management, and performance characteristics of native C++ STL containers.

Right now, this is a work in progress. As of the current build, only the C++ `std::vector` has been implemented.

## Features

Even though it is just the vector right now, it comes fully packed with standard C++ features mapped to native Python syntax, massively accelerated for high-performance computing:

* **Hardware-Accelerated Vectorization:** Linear operations like bulk filling, scaling, and reductions are parallelized across all available CPU cores using OpenMP, bypassing the Python GIL.
* **Zero-Copy NumPy Integration:** Implements the Python Buffer Protocol, allowing direct, instant casting to `np.asarray()` without duplicating memory.
* **O(N) Parallel Radix Sort:** Replaces standard comparison sorting with a custom 4-pass parallel Radix algorithm to crush large integer datasets.
* **Native Python Indexing:** Use standard bracket syntax to read and write data (`vec[0] = 6`).
* **Strict Boundary Checking:** C++ will physically prevent you from writing to unallocated memory slots.
* **Direct Capacity Management:** Access to C++ specific methods like `capacity()`, `reserve()`, and `shrinkFit()`.
* **Multiple Constructors:** Initialize an empty vector to build dynamically, or pre-allocate a specific size instantly.

## Installation

To build this project, you will need a C++ compiler (like MSVC or GCC) supporting C++17 and CMake installed on your system.

Clone this repository, navigate to the folder, and run:

`pip install .`

If you are actively developing and want to force a clean rebuild without pip using a cached version, run:

`pip install . --no-cache-dir --force-reinstall`

## Benchmarking

To test the multi-threaded performance of `pystl` against standard Python lists and NumPy, run the included `test.py` script. It benchmarks vector filling, zero-copy buffer assignments, reductions (sum, min, max), and parallel sorting across array sizes up to 100,000,000 elements. You can use this file as a reference to write your own custom benchmarks.

## Contributing

If you want to help add more data structures or improve the pybind11 bindings, feel free to open a pull request.
