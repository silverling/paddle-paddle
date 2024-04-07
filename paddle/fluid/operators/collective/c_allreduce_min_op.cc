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

#include <stdint.h>
#include <algorithm>
#include <string>

#include "paddle/fluid/operators/collective/c_allreduce_op.h"
#include "paddle/fluid/framework/inplace_op_inference.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.inl"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
class CPUContext;
}  // namespace phi

namespace paddle {
namespace operators {

class CAllReduceMinOpMaker : public CAllReduceOpMaker {
 protected:
  std::string GetName() const override { return "Min"; }
};

DECLARE_INPLACE_OP_INFERER(AllreduceMinInplaceInferer, {"X", "Out"});

DEFINE_C_ALLREDUCE_CPU_KERNEL(CAllReduceMin, kRedMin)

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_allreduce_min,
                             ops::CAllReduceOp,
                             ops::CAllReduceMinOpMaker,
                             ops::AllreduceMinInplaceInferer)

PD_REGISTER_STRUCT_KERNEL(c_allreduce_min,
                          CPU,
                          ALL_LAYOUT,
                          ops::CAllReduceMinCPUKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          plat::float16) {}
