#include <torch/extension.h>

#define BASES "AUGC"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <deque>
#include <tuple>

#include "st.h"
#include "st.cpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("parse_dot_bracket", &parseDotBracket);
   m.def("st2pairmap", &st2pairmap);
   m.def("pairmap2tensor", &pairMap2Tensor, py::arg("pair_map"), py::arg("out_dim") = 3);
}
