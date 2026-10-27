#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<vector>
#include<stack>
using namespace std;
namespace py = pybind11;

class Vector {
    vector<int> vec;
public:
    Vector() = default;
    Vector(int n) : vec(n) {}
    void set(int index, int val){vec.at(index) = val;}
    void append(int val){vec.push_back(val);}
    int get(int index){return vec.at(index);}
    size_t size() const{return vec.size();}
    size_t maxsize() const{return vec.max_size();}
    void resize(int n){vec.resize(n);}
    size_t capacity() const{return vec.capacity();}
    bool isEmpty() const{return vec.empty();}
    void reserve(int n){vec.reserve(n);}
    void shrinkFit(){vec.shrink_to_fit();}

};


PYBIND11_MODULE(pystl, m) {
    py::class_<Vector>(m, "vector")
        .def(py::init<>())
        .def(py::init<int>())
        .def("__setitem__", &Vector::set)
        .def("append", &Vector::append)
        .def("__getitem__", &Vector::get)
        .def("__len__", &Vector::size)
        .def("maxsize", &Vector::maxsize)
        .def("resize", &Vector::resize)
        .def("capacity", &Vector::capacity)
        .def("isEmpty", &Vector::isEmpty)
        .def("reserve", &Vector::reserve)
        .def("shrinkFit", &Vector::shrinkFit);
}
