//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/unbind_kernel.h"

#include <stdint.h>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/unbind_kernel_impl.h"
#include "paddle/phi/backends/cpu/cpu_context.h"

namespace phi {
namespace dtype {
struct bfloat16;
struct float16;
template <typename T> struct __attribute__((aligned(sizeof(T) * 2))) complex;
}  // namespace dtype
}  // namespace phi

PD_REGISTER_KERNEL(unbind,
                   CPU,
                   ALL_LAYOUT,
                   phi::UnbindKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
