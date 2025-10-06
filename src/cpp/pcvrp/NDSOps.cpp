// cppimport

/**
 * PCVRP NDSOps Python Bindings
 * 
 * This file provides Python bindings for the PCVRP (Prize-Collecting Vehicle Routing Problem)
 * C++ implementation using pybind11. It exposes the core classes and functions
 * needed for solving PCVRP instances with prize collection.
 */


#include <vector>
#include <stdexcept>
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <tuple>
#include <unordered_set>
#include <iterator>
#include <numeric>
#include <fstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "Instance.h"
#include "Utils.h"
#include "Solution.h"
#include "Tour.h"
#include "Operations.h"

/**
 * Python module definition for PCVRP operations
 */
PYBIND11_MODULE(NDSOps, m) {
    m.doc() = "PCVRP NDSOps - Python bindings for Prize-Collecting Vehicle Routing Problem operations";

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

    // Instance class binding
    py::class_<Instance>(m, "Instance")
        .def(py::init<int, int, const std::vector<int>&, const std::vector<std::vector<float>>&, const std::vector<float>&>(),
             "Constructor for PCVRP instance",
             py::arg("numCustomers"), py::arg("vehicleCapacity"), 
             py::arg("demand"), py::arg("nodePositions"), py::arg("prizes"))
        .def_readwrite("numNodes", &Instance::numNodes, "Total number of nodes (customers + depot)")
        .def_readwrite("numCustomers", &Instance::numCustomers, "Number of customers (excluding depot)")
        .def_readwrite("vehicleCapacity", &Instance::vehicleCapacity, "Vehicle capacity constraint")
        .def_readwrite("distanceMatrix", &Instance::distanceMatrix, "Distance matrix between all nodes");

    // Solution class binding
    py::class_<Solution>(m, "Solution")
        .def(py::init<const Instance&, const std::vector<std::vector<int>>&>(),
             "Constructor for PCVRP solution",
             py::arg("instance"), py::arg("tours"))
        .def_readwrite("totalCosts", &Solution::totalCosts, "Total cost of the solution")
        .def("getTourList", &Solution::getTourList, "Get list of tours in the solution");

    // Function bindings
    m.def("create_starting_solution", &create_starting_solution, 
          "Create a starting solution for the given instance",
          py::arg("instance"), py::arg("nbImprovement"), py::arg("nbDestroy"),
          py::return_value_policy::take_ownership);
          
    m.def("remove_recreate_allImp", &remove_recreate_allImp, 
          "Remove and recreate operation with all improvements",
          py::arg("solution"), py::arg("A"), py::arg("beta"), py::arg("n"), py::arg("T"), py::arg("insertInNewToursOnly"),
          py::return_value_policy::take_ownership);
          
    m.def("remove_recreate_singleImp", &remove_recreate_singleImp, 
          "Remove and recreate operation with single improvement",
          py::arg("solution"), py::arg("A"), py::arg("beta"), py::arg("n"), py::arg("insertInNewToursOnly"),
          py::return_value_policy::take_ownership);
          
    m.def("heuristic_deconstruction_selection", &heuristic_deconstruction_selection, 
          "Heuristic selection of customers for deconstruction",
          py::arg("solution"), py::arg("c_bar"), py::arg("m"),
          py::return_value_policy::take_ownership);
}

/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-O2']
cfg['sources'] = ['Instance.cpp', 'Operations.cpp', 'Solution.cpp', 'Tour.cpp', 'Utils.cpp']
%>
*/