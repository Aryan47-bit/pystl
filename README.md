# pystl
Welcome to pystl. This is a project aimed at bringing the C++ Standard Template Library (STL) directly into Python.

Python lists are great, but they hide a lot of the underlying memory management from the user. I built this wrapper using pybind11 to give Python developers access to the strict memory rules, capacity management, and performance characteristics of native C++ STL containers.

Right now, this is a work in progress. As of the current build, only the C++ std::vector has been implemented.

Features

Even though it is just the vector right now, it comes fully packed with standard C++ features mapped to native Python syntax:

Native Python indexing: Use standard bracket syntax to read and write data.

Strict boundary checking: C++ will physically prevent you from writing to unallocated memory slots.

Direct capacity management: Access to C++ specific methods like capacity(), reserve(), and shrinkFit().

Multiple constructors: Initialize an empty vector to build dynamically, or pre-allocate a specific size instantly.

Installation:-

To build this project, you will need a C++ compiler (like MSVC or GCC) and CMake installed on your system.

Clone this repository, navigate to the folder, and run:

pip install .


If you are actively developing and want to force a clean rebuild without pip using a cached version, run:

pip install . --no-cache-dir --force-reinstall


Quick Start:-

Here is a quick example of how to use the vector in your Python scripts. Because it strictly follows C++ rules, remember that you cannot assign a value to an index that hasn't been created yet.

from pystl import vector

# Initialize a vector with 10 slots (filled with zeros)
vec = vector(10)

# Overwrite existing slots using normal Python syntax
vec[0] = 6
vec[1] = 7
vec[2] = 67
(😝)


# Append dynamically expands the array and attaches to the end
vec.append(6767)

# Access C++ specific memory diagnostics
print("Vector size: ", len(vec))
print("Memory capacity: ", vec.capacity())
print("Max size: ", vec.maxsize())

# Free up unused memory capacity
vec.shrinkFit()

Of course, this is just the beginning. I'll be working on this consistently and adding each and every essential thing from STL into this.

Contributing

If you want to help add more data structures or improve the pybind11 bindings, feel free to open a pull request.


