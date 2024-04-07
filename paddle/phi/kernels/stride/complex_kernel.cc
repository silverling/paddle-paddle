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
#include "paddle/phi/kernels/complex_kernel.h"

#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.inl"
#include "paddle/phi/core/kernel_factory.h"

namespace phi {
class CPUContext;
class GPUContext;
namespace dtype {
template <typename T> struct __attribute__((aligned(sizeof(T) * 2))) complex;
}  // namespace dtype

template <typename T, typename Context>
void RealStridedKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out) {
  if (x.dtype() != DataType::COMPLEX64 && x.dtype() != DataType::COMPLEX128) {
    PADDLE_THROW(
        phi::errors::NotFound("paddle.real only support COMPLEX64 and "
                              "COMPLEX128, but the input dtype is %s",
                              x.dtype()));
  }
  DDim stride = x.strides();
  for (int i = 0; i < stride.size(); i++) {
    stride[i] = x.strides()[i] * 2;
  }
  out->set_offset(x.offset());
  out->set_strides(stride);
  out->ResetHolder(x.Holder());
  out->ShareInplaceVersionCounterWith(x);
}

template <typename T, typename Context>
void ImagStridedKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out) {
  if (x.dtype() != DataType::COMPLEX64 && x.dtype() != DataType::COMPLEX128) {
    PADDLE_THROW(
        phi::errors::NotFound("paddle.imag only support COMPLEX64 and "
                              "COMPLEX128, but the input dtype is %s",
                              x.dtype()));
  }
  DDim stride = x.strides();
  for (int i = 0; i < stride.size(); i++) {
    stride[i] = x.strides()[i] * 2;
  }
  out->set_strides(stride);
  out->set_offset(x.offset() + phi::SizeOf(out->dtype()));
  out->ResetHolder(x.Holder());
  out->ShareInplaceVersionCounterWith(x);
}

}  // namespace phi

PD_REGISTER_KERNEL(real,
                   CPU,
                   STRIDED,
                   phi::RealStridedKernel,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_KERNEL(imag,
                   CPU,
                   STRIDED,
                   phi::ImagStridedKernel,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(real,
                   GPU,
                   STRIDED,
                   phi::RealStridedKernel,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_KERNEL(imag,
                   GPU,
                   STRIDED,
                   phi::ImagStridedKernel,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
PD_REGISTER_KERNEL(real,
                   Custom,
                   STRIDED,
                   phi::RealStridedKernel,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_KERNEL(imag,
                   Custom,
                   STRIDED,
                   phi::ImagStridedKernel,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
#endif
