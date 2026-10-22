#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

class FastVector {
    std::vector<int> data;
public:
    void append(int val) { data.push_back(val); }
    int get(int index) { return data.at(index); }
    int size() const { return data.size(); }
};

PYBIND11_MODULE(pystl, m) {
    py::class_<FastVector>(m, "Vector")
        .def(py::init<>())
        .def("append", &FastVector::append)
        .def("get", &FastVector::get)
        .def("__len__", &FastVector::size);
}