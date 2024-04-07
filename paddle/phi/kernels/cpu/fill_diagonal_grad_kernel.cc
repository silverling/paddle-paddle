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

#include "paddle/phi/kernels/fill_diagonal_grad_kernel.h"

#include <stdint.h>
#include <algorithm>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/common/ddim.h"
#include "paddle/common/macros.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void FillDiagonalGradKernel(const Context& ctx,
                            const DenseTensor& out_grad,
                            float value UNUSED,
                            int offset,
                            bool wrap,
                            DenseTensor* x_grad) {
  if (x_grad) {
    T* data = ctx.template Alloc<T>(x_grad);
    phi::Copy(ctx, out_grad, ctx.GetPlace(), false, x_grad);

    auto dx_dims = x_grad->dims();
    auto strides = funcs::CalStride(dx_dims);
    auto size = x_grad->numel();
    auto wrapsize = std::min(size, dx_dims[1] * dx_dims[1]);

    // The wrap mode supported only the dims equels to 2; In wrap mode, the
    // value will be filled in cycles
    if (wrap) {
      wrapsize = size;
    }

    for (int64_t i = 0; i < wrapsize; i += strides) {
      if (i % dx_dims[1] + offset >= 0 &&
          i % dx_dims[1] + offset < dx_dims[1]) {
        data[i + offset] = T(0);
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(fill_diagonal_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::FillDiagonalGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   bool) {}
