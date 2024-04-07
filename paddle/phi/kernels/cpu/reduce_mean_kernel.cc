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

#include "paddle/phi/kernels/reduce_mean_kernel.h"

#include <cstdint>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "Eigen/src/Core/arch/AVX/PacketMath.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_utils.h"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDimensions.h"
#include "unsupported/Eigen/CXX11/src/util/CXX11Meta.h"

namespace phi {
namespace funcs {
struct MeanFunctor;
}  // namespace funcs

template <typename T, typename Context>
void MeanRawKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& dims,
                   bool keep_dim,
                   bool reduce_all,
                   DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  auto out_dtype = x.dtype();
  phi::Reduce<CPUContext, T, phi::funcs::MeanFunctor>(
      dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(mean_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::MeanRawKernel,
                   float,
                   double,
                   bool,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
