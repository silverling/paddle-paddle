// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/reduce_any_kernel.h"

#include <cstdint>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_utils.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h"
#include "unsupported/Eigen/CXX11/src/util/CXX11Meta.h"

namespace phi {
class DenseTensor;
namespace funcs {
struct AnyFunctor;
}  // namespace funcs

template <typename T, typename Context>
void AnyRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  phi::BoolReduceKernel<CPUContext, T, phi::funcs::AnyFunctor>(
      dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(any_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::AnyRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
