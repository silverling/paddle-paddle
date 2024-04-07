/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/unique_op.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dense_tensor.inl"
#include "paddle/utils/variant.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class UniqueWithCountsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto data_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));
    auto* x = context.Input<phi::DenseTensor>("X");
    auto* out = context.Output<phi::DenseTensor>("Out");
    auto* index = context.Output<phi::DenseTensor>("Index");
    auto* count = context.Output<phi::DenseTensor>("Count");
    framework::VisitDataType(data_type,
                             UniqueOpFunctor<T>(out, index, x, count));
  }
};

}  // namespace operators
}  // namespace paddle
