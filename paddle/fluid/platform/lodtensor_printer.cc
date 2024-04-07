/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/lodtensor_printer.h"

#include <stdint.h>
#include <ostream>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dense_tensor.inl"

namespace paddle {
namespace platform {

void PrintVar(framework::Scope* scope,
              const std::string& var_name,
              const std::string& print_info,
              std::stringstream* sstream) {
  framework::Variable* var = scope->FindVar(var_name);
  if (var == nullptr) {
    VLOG(0) << "Variable Name " << var_name << " does not exist in your scope";
    return;
  }
  phi::DenseTensor* tensor = var->GetMutable<phi::DenseTensor>();
  if (tensor == nullptr) {
    VLOG(0) << "tensor of variable " << var_name
            << " does not exist in your scope";
    return;
  }
  if (!tensor->IsInitialized()) {
    VLOG(0) << "tensor of variable " << var_name
            << " does not initialized in your scope";
    return;
  }

  *sstream << print_info;

#define PrintTensorCallback(cpp_type, proto_type)                        \
  do {                                                                   \
    if (framework::TransToProtoVarType(tensor->dtype()) == proto_type) { \
      *sstream << "[";                                                   \
      const cpp_type* data = nullptr;                                    \
      phi::DenseTensor cpu_tensor;                                       \
      if (is_cpu_place(tensor->place())) {                               \
        data = tensor->data<cpp_type>();                                 \
      } else {                                                           \
        platform::CPUPlace cpu_place;                                    \
        paddle::framework::TensorCopy(*tensor, cpu_place, &cpu_tensor);  \
        data = cpu_tensor.data<cpp_type>();                              \
      }                                                                  \
      auto element_num = tensor->numel();                                \
      *sstream << element_num << "]:[";                                  \
      if (element_num > 0) {                                             \
        *sstream << data[0];                                             \
        for (int j = 1; j < element_num; ++j) {                          \
          *sstream << " " << data[j];                                    \
        }                                                                \
      }                                                                  \
      *sstream << "]";                                                   \
    }                                                                    \
  } while (0)

  _ForEachDataType_(PrintTensorCallback);
}

}  // end namespace platform
}  // end namespace paddle
