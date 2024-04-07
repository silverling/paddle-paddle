/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/spp_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.inl"
#include "paddle/phi/core/kernel_registry.h"

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(spp, GPU, ALL_LAYOUT, ops::SppKernel, float, double) {
}
PD_REGISTER_STRUCT_KERNEL(
    spp_grad, GPU, ALL_LAYOUT, ops::SppGradKernel, float, double) {}
