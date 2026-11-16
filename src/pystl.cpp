#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>
#include <vector>
#include <map>
#include "robin_hood.h"
#include <algorithm>
#include <omp.h>
#include <stdexcept>
#include <execution>
#include <cstring>

namespace nb = nanobind;

template <typename T>
void gilDrop(size_t size, T&& op) {
    if (size < 100000) {
        op();
    } else {
        nb::gil_scoped_release release; 
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
        nb::gil_scoped_release release;
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
                    }
                }               
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
                        threadMin = raw[i]; 
                    }
                }
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

class Map {
    std::map<int, int> map;
public:
    Map() = default;

    void set(int key, int val) { 
        map[key] = val; 
    }

    int get(int key) {
        auto it = map.find(key);
        if (it == map.end()) {
            throw nb::key_error("Key not found in pystl.map");
        }
        return it->second;
    }

    bool contains(int key) { 
        return map.find(key) != map.end(); 
    }

    void erase(int key) {
        if (map.erase(key) == 0) {
            throw nb::key_error("Key not found in pystl.map");
        }
    }

    size_t size() const { return map.size(); }
    bool isEmpty() const { return map.empty(); }
    void clear() { map.clear(); }

    void bulk_update(nb::ndarray<nb::numpy, int, nb::ro> keys, 
                     nb::ndarray<nb::numpy, int, nb::ro> vals) {
        
        if (keys.size() != vals.size()) {
            throw std::invalid_argument("Keys and values arrays must be the same size");
        }

        const int* k_ptr = keys.data();
        const int* v_ptr = vals.data();
        size_t sz = keys.size();

        if (sz > 1000) {
            nb::gil_scoped_release release;
            for (size_t i = 0; i < sz; i++) {
                map[k_ptr[i]] = v_ptr[i];
            }
        } else {
            for (size_t i = 0; i < sz; i++) {
                map[k_ptr[i]] = v_ptr[i];
            }
        }
    }
    std::pair<int, int> lower_bound(int key) {
        auto it = map.lower_bound(key);
        if (it == map.end()) {
            throw nb::key_error("No key found >= target");
        }
        return {it->first, it->second};
    }

    std::pair<int, int> upper_bound(int key) {
        auto it = map.upper_bound(key);
        if (it == map.end()) {
            throw nb::key_error("No key found > target");
        }
        return {it->first, it->second};
    }
    nb::ndarray<nb::numpy, int, nb::shape<-1>> keys_array() {
        size_t sz = map.size();
        int* buffer = new int[sz]; 
        
        size_t i = 0;
        for (const auto& item : map) {
            buffer[i++] = item.first;
        }

        nb::capsule deleter(buffer, [](void *p) noexcept {
            delete[] static_cast<int*>(p);
        });
        
        return nb::ndarray<nb::numpy, int, nb::shape<-1>>(
            buffer, 
            { sz }, 
            deleter
        );
    }

    std::map<int, int>& get_map() { return map; }
};

class UnorderedMap {
    // THE UPGRADE: Replaces the slow standard library hash table
    robin_hood::unordered_map<int, int> umap;
public:
    UnorderedMap() = default;

    void set(int key, int val) { 
        umap[key] = val; 
    }

    int get(int key) {
        auto it = umap.find(key);
        if (it == umap.end()) {
            throw nb::key_error("Key not found in pystl.unordered_map");
        }
        return it->second;
    }

    bool contains(int key) { 
        return umap.find(key) != umap.end(); 
    }

    void erase(int key) {
        if (umap.erase(key) == 0) {
            throw nb::key_error("Key not found in pystl.unordered_map");
        }
    }

    size_t size() const { return umap.size(); }
    bool isEmpty() const { return umap.empty(); }
    void clear() { umap.clear(); }
    
    void reserve(size_t n) { umap.reserve(n); }

    void bulk_update(nb::ndarray<nb::numpy, int, nb::ro> keys, 
                     nb::ndarray<nb::numpy, int, nb::ro> vals) {
        
        if (keys.size() != vals.size()) {
            throw std::invalid_argument("Keys and values arrays must be the same size");
        }

        const int* k_ptr = keys.data();
        const int* v_ptr = vals.data();
        long long sz = static_cast<long long>(keys.size());

        umap.reserve(umap.size() + sz);

        if (sz < 100000) {
            nb::gil_scoped_release release;
            for (long long i = 0; i < sz; i++) {
                umap[k_ptr[i]] = v_ptr[i];
            }
        } else {
            nb::gil_scoped_release release;
            int num_threads = omp_get_max_threads();
            
            std::vector<robin_hood::unordered_map<int, int>> local_maps(num_threads);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
             
                local_maps[tid].reserve(sz / num_threads + 100);

                #pragma omp for nowait
                for (long long i = 0; i < sz; i++) {
                    local_maps[tid][k_ptr[i]] = v_ptr[i];
                }
            }

            for (int t = 0; t < num_threads; t++) {
                for (const auto& kv : local_maps[t]) {
                    umap[kv.first] = kv.second;
                }
            }
        }
    }

    nb::ndarray<nb::numpy, int, nb::shape<-1>> keys_array() {
        size_t sz = umap.size();
        int* buffer = new int[sz]; 
        
        size_t i = 0;
        for (const auto& item : umap) {
            buffer[i++] = item.first;
        }
        
        nb::capsule deleter(buffer, [](void *p) noexcept {
            delete[] static_cast<int*>(p);
        });
        
        return nb::ndarray<nb::numpy, int, nb::shape<-1>>(
            buffer, { sz }, deleter
        );
    }

    robin_hood::unordered_map<int, int>& get_map() { return umap; }
};



NB_MODULE(pystl, m) {
    nb::class_<Vector>(m, "vector") 
        .def(nb::init<>())
        .def(nb::init<int>())
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
        .def("fill", &Vector::fill, nb::arg("val") = 0)
        .def("max", &Vector::max)
        .def("min", &Vector::min)
        .def("to_numpy", [](Vector &v) {
            return nb::ndarray<nb::numpy, int, nb::shape<-1>>(
                v.data(), 
                { v.size() }, 
                nb::handle() 
            );
        }); 

    nb::class_<Map>(m, "map")
        .def(nb::init<>())
        .def("set", &Map::set)
        .def("__setitem__", &Map::set)
        .def("get", &Map::get)
        .def("__getitem__", &Map::get)
        .def("__contains__", &Map::contains)
        .def("erase", &Map::erase)
        .def("__delitem__", &Map::erase)
        .def("size", &Map::size)
        .def("__len__", &Map::size)
        .def("isEmpty", &Map::isEmpty)
        .def("clear", &Map::clear)
        .def("bulk_update", &Map::bulk_update)
        .def("lower_bound", &Map::lower_bound)
        .def("upper_bound", &Map::upper_bound)
        .def("keys_array", &Map::keys_array) 
        .def("keys", [m](Map &self) {
            return nb::make_key_iterator(m, "key_iterator", 
                                         self.get_map().begin(), self.get_map().end());
        }, nb::keep_alive<0, 1>())
        .def("values", [m](Map &self) {
            return nb::make_value_iterator(m, "value_iterator", 
                                           self.get_map().begin(), self.get_map().end());
        }, nb::keep_alive<0, 1>())
        .def("items", [m](Map &self) {
            return nb::make_iterator(m, "item_iterator", 
                                     self.get_map().begin(), self.get_map().end());
        }, nb::keep_alive<0, 1>());        

    nb::class_<UnorderedMap>(m, "unordered_map")
        .def(nb::init<>())
        .def("set", &UnorderedMap::set)
        .def("__setitem__", &UnorderedMap::set)
        .def("get", &UnorderedMap::get)
        .def("__getitem__", &UnorderedMap::get)
        .def("__contains__", &UnorderedMap::contains)
        .def("erase", &UnorderedMap::erase)
        .def("__delitem__", &UnorderedMap::erase)
        .def("size", &UnorderedMap::size)
        .def("__len__", &UnorderedMap::size)
        .def("isEmpty", &UnorderedMap::isEmpty)
        .def("clear", &UnorderedMap::clear)
        .def("reserve", &UnorderedMap::reserve)
        .def("bulk_update", &UnorderedMap::bulk_update)
        .def("keys_array", &UnorderedMap::keys_array)
        .def("keys", [m](UnorderedMap &self) {
            return nb::make_key_iterator(m, "umap_key_iterator", 
                                         self.get_map().begin(), self.get_map().end());
        }, nb::keep_alive<0, 1>())
        .def("values", [m](UnorderedMap &self) {
            return nb::make_value_iterator(m, "umap_value_iterator", 
                                           self.get_map().begin(), self.get_map().end());
        }, nb::keep_alive<0, 1>())
        .def("items", [m](UnorderedMap &self) {
            return nb::make_iterator(m, "umap_item_iterator", 
                                     self.get_map().begin(), self.get_map().end());
        }, nb::keep_alive<0, 1>());
}