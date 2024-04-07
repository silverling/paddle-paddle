// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <vector>

#include "paddle/phi/kernels/flatten_grad_kernel.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "paddle/common/ddim.h"
#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dense_tensor.inl"
#include "paddle/phi/kernels/funcs/data_layout_transform.h"

namespace phi {
namespace dtype {
struct bfloat16;
}  // namespace dtype

template <typename T, typename Context>
void FlattenGradKernel(const Context& dev_ctx,
                       const DenseTensor& xshape,
                       const DenseTensor& out_grad,
                       DenseTensor* x_grad) {
  auto out_grad_vec_dims = out_grad.dims().size() != 0
                               ? common::vectorize(out_grad.dims())
                               : std::vector<int64_t>{1};

  auto out_grad_type = funcs::ToOneDNNDataType(out_grad.dtype());

  funcs::ReorderOneDNNHandler reorder_handler(
      out_grad_vec_dims, out_grad.dtype(), out_grad_type, dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      out_grad.mem_desc(), funcs::to_void_cast(out_grad.data<T>()));
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      x_grad,
      funcs::GetPlainOneDNNFormat(out_grad_vec_dims.size()),
      dev_ctx.GetPlace());
  auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                  reorder_src_memory_p);

  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();

  auto x_grad_dims = slice_ddim(xshape.dims(), 1, xshape.dims().size());
  x_grad->Resize(x_grad_dims);
  reorder_dst_memory_p->get_desc().reshape(common::vectorize(x_grad_dims));
}

}  // namespace phi

PD_REGISTER_KERNEL(flatten_grad,
                   OneDNN,
                   ONEDNN,
                   phi::FlattenGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
