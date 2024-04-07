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

#include "paddle/phi/kernels/fft_grad_kernel.h"

#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/fft_grad_kernel_impl.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_factory.h"

PD_REGISTER_KERNEL(fft_c2c_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::FFTC2CGradKernel,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
PD_REGISTER_KERNEL(
    fft_c2r_grad, CPU, ALL_LAYOUT, phi::FFTC2RGradKernel, float, double) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToComplex(kernel_key.dtype()));
}
PD_REGISTER_KERNEL(fft_r2c_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::FFTR2CGradKernel,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
