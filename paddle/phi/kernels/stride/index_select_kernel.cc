// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_select_kernel.h"

#include <cstdint>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/common/ddim.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dense_tensor.inl"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {
class CPUContext;
class GPUContext;

template <typename Context>
void IndexSelectStridedKernel(const Context& ctx,
                              const DenseTensor& x,
                              int64_t index,
                              int dim,
                              DenseTensor* output) {
  auto input_dim = x.dims();
  dim = dim >= 0 ? dim : dim + input_dim.size();

  std::vector<int64_t> shape = common::vectorize<int64_t>(x.dims());
  std::vector<int64_t> stride = common::vectorize<int64_t>(x.strides());
  int64_t offset = static_cast<int64_t>(x.offset());

  offset = static_cast<int64_t>(offset +
                                index * stride[dim] * SizeOf(output->dtype()));
  shape.erase(shape.begin() + dim);
  stride.erase(stride.begin() + dim);

  auto meta = output->meta();
  meta.offset = offset;
  auto tmp_dim = DDim(shape.data(), static_cast<int>(shape.size()));
  // if (product(meta.dims) > 0 && meta.dims != tmp_dim) {
  //   PADDLE_THROW(
  //       phi::errors::Fatal("Index_select kernel stride compute diff, infer "
  //                          "shape is %s, but compute is %s.",
  //                          meta.dims,
  //                          tmp_dim));
  // }
  meta.dims = tmp_dim;
  meta.strides = DDim(stride.data(), static_cast<int>(stride.size()));
  output->set_meta(meta);
  output->ResetHolder(x.Holder());
  output->ShareInplaceVersionCounterWith(x);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(index_select_strided,
                                         STRIDED,
                                         phi::IndexSelectStridedKernel) {}
