#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <stdexcept>
#include <execution>
#include <cstring>

namespace py = pybind11;

template <typename T>
void gilDrop(size_t size, T&& op) {
    if (size < 100000) {
        op();
    } else {
        py::gil_scoped_release release; 
        op();
    }
}

void rsort(int* raw, long long sz) {
    std::vector<int> buffer(sz);
    int* src = raw;
    int* dst = buffer.data();
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<int>> thread_counts(max_threads, std::vector<int>(256, 0));

    for (int shift = 0; shift < 32; shift += 8) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            for (int i = 0; i < 256; i++) thread_counts[tid][i] = 0;

            long long chunk = sz / num_threads;
            long long start = tid * chunk;
            long long end = (tid == num_threads - 1) ? sz : start + chunk;

            for (long long i = start; i < end; i++) {
                unsigned char byte = ((unsigned int)src[i] >> shift) & 0xFF;
                if (shift == 24) byte ^= 128;
                thread_counts[tid][byte]++;
            }
            #pragma omp barrier
            #pragma omp single
            {
                int total = 0;
                for (int bin = 0; bin < 256; bin++) {
                    for (int t = 0; t < num_threads; t++) {
                        int count = thread_counts[t][bin];
                        thread_counts[t][bin] = total;
                        total += count;
                    }
                }
            }
            for (long long i = start; i < end; i++) {
                unsigned char byte = ((unsigned int)src[i] >> shift) & 0xFF;
                if (shift == 24) byte ^= 128;
                int dest_idx = thread_counts[tid][byte]++;
                dst[dest_idx] = src[i];
            }
        }
        std::swap(src, dst);
    }
    if (src != raw) {
        std::memcpy(raw, buffer.data(), sz * sizeof(int));
    }
}
class Vector {
    std::vector<int> vec;
public:
    Vector() = default;
    Vector(int n) : vec(n) {}
    void set(int index, int val) { vec.at(index) = val; }
    void append(int val) { vec.push_back(val); }
    int get(int index) { return vec.at(index); }
    size_t size() const { return vec.size(); }
    size_t maxsize() const { return vec.max_size(); }
    void resize(int n) { vec.resize(n); }
    size_t capacity() const { return vec.capacity(); }
    bool isEmpty() const { return vec.empty(); }
    void reserve(int n) { vec.reserve(n); }
    void shrinkFit() { vec.shrink_to_fit(); }
    int* data() { return vec.data(); }
    
    void add_scalar(int val) {
        int* raw = vec.data();
        long long sz = static_cast<long long>(vec.size());
        gilDrop(sz, [&]() {
            #pragma omp parallel for
            for (long long i = 0; i < sz; i++) {
                raw[i] += val;
            }
        });
    }
    
    void fill(int val = 0) {
        int* raw = vec.data();
        long long sz = static_cast<long long>(vec.size());
        gilDrop(sz, [&]() {
            #pragma omp parallel for
            for (long long i = 0; i < sz; i++) {
                raw[i] = val;
            }
        });
    }
    
    void sort() {
        py::gil_scoped_release release;
        rsort(vec.data(), vec.size());
    }
    
    long long sum() {
        int* raw = vec.data();
        long long finalSum = 0;
        long long sz = static_cast<long long>(vec.size());
        
        gilDrop(sz, [&]() {
            long long localSum = 0; 
            
            #pragma omp parallel for reduction(+:localSum)
            for (long long i = 0; i < sz; i++) {
                localSum += raw[i];
            }
            
            finalSum = localSum; 
        });
        return finalSum;
    }
    
    int max() {
        if (vec.empty()) {
            throw std::runtime_error("Empty Vector"); 
        }
        int* raw = vec.data();
        long long sz = static_cast<long long>(vec.size());
        int finalMax = raw[0];

        gilDrop(sz, [&]() {
            int globalMax = finalMax; 
            
            #pragma omp parallel
            {
                int threadMax = globalMax;
                #pragma omp for
                for (long long i = 0; i < sz; i++) {
                    if (raw[i] > threadMax) {
                        threadMax = raw[i];
                    }}               
                #pragma omp critical
                {
                    if (threadMax > globalMax) {
                        globalMax = threadMax;
                    }
                }
            }
            finalMax = globalMax; 
        });
        return finalMax;
    }

    int min() {
        if (vec.empty()) {
            throw std::runtime_error("Empty Vector.");
        }
        int* raw = vec.data();
        long long sz = static_cast<long long>(vec.size());
        int finalMin = raw[0];
        
        gilDrop(sz, [&]() {
            int globalMin = finalMin; 
            
            #pragma omp parallel
            {
                int threadMin= globalMin;
                #pragma omp for
                for (long long i = 0; i < sz; i++) {
                    if (raw[i] < threadMin) {
                        threadMin = raw[i]; }}
                #pragma omp critical
                {
                    if (threadMin < globalMin) {
                        globalMin = threadMin;
                    }
                }
            }
            
            finalMin = globalMin; 
        });
        return finalMin;
    }
};

PYBIND11_MODULE(pystl, m) {
    py::class_<Vector>(m, "vector", py::buffer_protocol()) 
        .def(py::init<>())
        .def(py::init<int>())
        .def("set", &Vector::set)
        .def("append", &Vector::append)
        .def("get", &Vector::get)
        .def("__setitem__", &Vector::set)
        .def("__getitem__", &Vector::get)
        .def("size", &Vector::size)
        .def("__len__", &Vector::size)
        .def("maxsize", &Vector::maxsize)
        .def("resize", &Vector::resize)
        .def("capacity", &Vector::capacity)
        .def("isEmpty", &Vector::isEmpty)
        .def("reserve", &Vector::reserve)
        .def("shrinkFit", &Vector::shrinkFit)
        .def("add_scalar", &Vector::add_scalar)
        .def("sort", &Vector::sort)
        .def("sum", &Vector::sum)
        .def("fill", &Vector::fill, py::arg("val") = 0)
        .def("max", &Vector::max)
        .def("min", &Vector::min)
        .def_buffer([](Vector &v) -> py::buffer_info {
            return py::buffer_info(
                v.data(), sizeof(int), py::format_descriptor<int>::format(),
                1, { v.size() }, { sizeof(int) }
            );
        });
}