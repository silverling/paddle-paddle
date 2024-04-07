/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include <features.h>

#include "paddle/fluid/framework/fleet/nccl_wrapper.h"
#include "paddle/fluid/pybind/nccl_wrapper_py.h"
#include "pybind11/detail/descr.h"
#include "pybind11/pytypes.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindNCCLWrapper(py::module* m) {
  py::class_<framework::NCCLWrapper>(*m, "Nccl")
      .def(py::init())
      .def("init_nccl", &framework::NCCLWrapper::InitNCCL)
      .def("set_nccl_id", &framework::NCCLWrapper::SetNCCLId)
      .def("set_rank_info", &framework::NCCLWrapper::SetRankInfo)
      .def("sync_var", &framework::NCCLWrapper::SyncVar);
}  // end NCCLWrapper
}  // end namespace pybind
}  // end namespace paddle
