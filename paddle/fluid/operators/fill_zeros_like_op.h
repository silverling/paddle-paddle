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

#pragma once
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dense_tensor.inl"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class FillZerosLikeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Output<phi::DenseTensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    phi::funcs::SetConstant<DeviceContext, T> setter;
    setter(context.template device_context<DeviceContext>(),
           out,
           static_cast<T>(0));
  }
};

template <typename T, typename DeviceContext>
class FillZerosLikeKernel2 : public FillZerosLikeKernel<T, DeviceContext> {};

}  // namespace operators
}  // namespace paddle
