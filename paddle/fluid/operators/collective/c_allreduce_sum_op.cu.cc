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

#include "paddle/fluid/operators/collective/c_allreduce_op.h"
#include "cuda.h"
#include "nccl.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/core/dense_tensor.inl"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
class GPUContext;
}  // namespace phi

namespace paddle {
namespace operators {
DEFINE_C_ALLREDUCE_CUDA_KERNEL(CAllReduceSum, kRedSum)
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(c_allreduce_sum,
                          GPU,
                          ALL_LAYOUT,
                          ops::CAllReduceSumCUDAKernel,
                          float,
#if NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000
                          plat::bfloat16,
#endif
                          double,
                          int,
                          int64_t,
                          plat::float16) {
}
